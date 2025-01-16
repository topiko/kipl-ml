import json
import os

import configs as lasereak_configs
import dotenv
import hydra
import mlflow
import torch
from kipl_ml.data import assets
from kipl_ml.data.wf_dataset import get_train_valid_test
from kipl_ml.defences.utils import _forge_single_defence
from kipl_ml.logging.logger import get_logger
from kipl_ml.logging.utils import get_mlflow_expr
from kipl_ml.metrics.clf_metrics import Accuracy, ClassRecall, CrossEntropyLoss
from kipl_ml.model_eval.evaluate import evaluate_model
from kipl_ml.models.laserbeak import get_model, get_signature
from kipl_ml.trace.features import Feats, FeatureTrs
from kipl_ml.train.loops import train_model
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from torchtune.training.lr_schedulers import get_cosine_schedule_with_warmup

logger = get_logger(__name__)

dotenv.load_dotenv()

WORKING_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_DIR_PATH = os.path.join(WORKING_DIR, "config")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")

assert MLFLOW_TRACKING_URI is not None, "MLFLOW_TRACKING_URI must be set in .env file."
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

FEAT_NAME_MAP = {
    "time_dirs": Feats.TIME_DIRS,
    "times_norm": Feats.TIMES_MAX_NORMALIZED,
    "cumul_norm": Feats.CUM_SIZE_DIRS_MAX_NORMALIZED,
    "iat_dirs": Feats.IAT_DIRS,
    "inv_iat_log_dirs": Feats.LOG_INV_FLOW_IAT_DIRS,
    "running_rates": Feats.RUNNING_RATE_SIZES,
}


@hydra.main(config_path=CONFIG_DIR_PATH, config_name="test", version_base=None)
def main(cfg: DictConfig):
    dataset_name = cfg.dataset.name
    model_name = cfg.model.name
    n_packets = cfg.trace.n_packets
    experiment_name = cfg.mlflow.experiment_name + "->" + model_name

    if (defence_name := cfg.defences["name"]) != "no_defence":
        experiment_name += f" vs. {defence_name}"

    feature_names = cfg.features.features

    config_path = os.path.join(list(lasereak_configs.__path__)[0], model_name + ".json")

    with open(config_path, "r") as fi:
        model_config = json.load(fi)

    if model_config.get("input_size"):
        logger.warning(
            f"Trace len {n_packets} overwritten by model config -> {model_config['input_size']}"
        )

        n_packets = model_config["input_size"]

    if model_config.get("feature_list"):
        feature_names = [FEAT_NAME_MAP[feat] for feat in model_config["feature_list"]]

    feature_trs = FeatureTrs(feature_names=feature_names, n_packets=n_packets)

    defence = _forge_single_defence(**cfg.defences)

    ds_train, ds_valid, ds_test = get_train_valid_test(
        dataset=dataset_name,
        n_samples=(cfg.dataset.n_train_traces, 1000, 1000),
        random_state=cfg.dataset.random_state,
        feature_trs=feature_trs,
        defence=defence,
    )

    model = get_model(
        model_name,
        n_classes=ds_train.n_classes,
        inputs=ds_train.output_sizes,
        model_config=model_config,
    )

    bs = cfg.train.batch_size
    num_workers = 4
    train_loader = DataLoader(
        ds_train, batch_size=bs, shuffle=True, num_workers=num_workers
    )
    valid_loader = DataLoader(
        ds_valid, batch_size=bs, shuffle=False, num_workers=num_workers
    )
    test_loader = DataLoader(
        ds_test, batch_size=bs, shuffle=False, num_workers=num_workers
    )

    opt_betas = (0.9, 0.999)
    opt_wd = 0.001
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.train.lr, betas=opt_betas, weight_decay=opt_wd
    )
    if cfg.train.scheduler == "cosine":
        warmup_period = cfg.train.warmup_period
        epochs = cfg.train.epochs
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=len(train_loader) * warmup_period,
            num_training_steps=len(train_loader) * epochs,
            num_cycles=0.5,
            last_epoch=-1,
        )
    elif cfg.train.scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.3, patience=5
        )
    else:
        scheduler = None

    loss_fn = torch.nn.CrossEntropyLoss()
    metrics = [Accuracy(), ClassRecall(1)]
    early_stop_metric = "loss"
    patience = cfg.train.patience

    experiment_id = get_mlflow_expr(experiment_name=experiment_name)
    mlflow.set_experiment(experiment_id=experiment_id)
    run_name = model_name
    with mlflow.start_run(run_name=run_name):

        mlflow.log_params(
            {
                "feature_names": cfg.features.features,
                "n_packets": n_packets,
                "model_name": model_name,
                "dataset_name": dataset_name,
                "n_train_traces": cfg.dataset.n_train_traces,
                "batch_size": bs,
                "patience": patience,
                "early_stop_metric": early_stop_metric,
                "data_random_state": cfg.dataset.random_state,
                "scheduler": cfg.train.scheduler,
                "lr": cfg.train.lr,
            }
        )

        trained_model = train_model(
            model=model,
            train_loader=train_loader,
            valid_loader=valid_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            metrics=metrics,
            early_stop_metric=early_stop_metric,
            lr_scheduler=scheduler,
            patience=patience,
        )

        signature = get_signature(model=trained_model, ds=ds_train)
        mlflow.pytorch.log_model(trained_model, "model", signature=signature)
        mlflow.log_table(ds_test.meta_df, "test_df.json")
        mlflow.log_table({"features": cfg.features.features}, "features.json")

        # Evaluate on test set
        metrics_vals = evaluate_model(
            model=trained_model,
            dataloader=test_loader,
            loss_fn=loss_fn,
            metrics=metrics,
        )
        metrics_vals = {f"test_{k}": v for k, v in metrics_vals.items()}
        mlflow.log_metrics(metrics_vals, step=None)


if __name__ == "__main__":
    main()
