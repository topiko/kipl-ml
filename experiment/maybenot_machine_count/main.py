import os

import configs as lasereak_configs
import dotenv
import hydra
import mlflow
import numpy as np
import torch
from kipl_ml.data import assets
from kipl_ml.data.wf_dataset import get_train_valid_test
from kipl_ml.defences.defences import Defences, NoDefence
from kipl_ml.defences.maybenot import Maybenot
from kipl_ml.logging.logger import get_logger
from kipl_ml.logging.utils import get_mlflow_expr
from kipl_ml.metrics.clf_metrics import Accuracy, ClassRecall, CrossEntropyLoss
from kipl_ml.model_eval.evaluate import evaluate_model
from kipl_ml.models.laserbeak import get_model, get_signature
from kipl_ml.models.utils import get_laserbeak_model_config
from kipl_ml.trace.features import FEAT_NAME_MAP, Feats, FeatureTrs
from kipl_ml.train.loops import train_model
from mlflow.data.pandas_dataset import PandasDataset
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from torchtune.training.lr_schedulers import get_cosine_schedule_with_warmup

logger = get_logger(__name__)

dotenv.load_dotenv()

WORKING_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_DIR_PATH = os.path.join(WORKING_DIR, "config")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
STORE_DATA_COLS = ["label", "trace_id"]

assert MLFLOW_TRACKING_URI is not None, "MLFLOW_TRACKING_URI must be set in .env file."
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)


BASE_MODEL_RUNEXPIDS = {
    "df-multi": ("768f7d7e0a68496aaa73a28c6d7b5f78", "562836848803030619")
}


def _fetch_base_model(model_name: str) -> torch.nn.Module:
    run_id, expr_id = BASE_MODEL_RUNEXPIDS[model_name]
    mlflow.set_experiment(experiment_id=expr_id)

    run = mlflow.get_run(run_id)

    return mlflow.pytorch.load_model(run.info.artifact_uri + "/model")


@hydra.main(config_path=CONFIG_DIR_PATH, config_name="test", version_base=None)
def main(cfg: DictConfig):
    dataset_name = cfg.dataset.name
    model_name = cfg.model.name
    n_packets = cfg.trace.n_packets
    experiment_name = cfg.mlflow.experiment_name + "->" + model_name + " vs. maybenot"

    model_config = get_laserbeak_model_config(model_name)
    if model_config.get("input_size"):
        logger.warning(
            f"Trace len {n_packets} overwritten by model config -> {model_config['input_size']}"
        )
        n_packets = model_config["input_size"]

    feature_names = cfg.features.features
    if model_config.get("feature_list"):
        logger.warning("Feature list overwritten by model config")
        feature_names = [FEAT_NAME_MAP[feat] for feat in model_config["feature_list"]]

    feature_trs = FeatureTrs(feature_names=feature_names, n_packets=n_packets)

    maybenot_config = dict(cfg.defences)

    if (n_machines := maybenot_config.pop("n_machines")) == 0:
        defence = NoDefence()
    else:
        rng = np.random.default_rng()
        machine_idxs = list(rng.choice(10000, size=n_machines, replace=False))
        maybenot_config["machine_idxs"] = machine_idxs
        defence = Maybenot(**maybenot_config)

    defences = Defences(defences=[defence])

    ds_train, ds_valid, ds_test = get_train_valid_test(
        dataset=dataset_name,
        n_samples=(cfg.dataset.n_train_traces, 1000, 1000),
        random_state=cfg.dataset.random_state,
        feature_trs=feature_trs,
        defences=defences,
    )

    build_model = True
    if cfg.load_base_model:
        try:
            model = _fetch_base_model(model_name)
            build_model = False
        except Exception as e:
            logger.error(f"Failed to fetch base model: {e}")

    if build_model:
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
    run_name = f"{n_machines} state-machines"
    with mlflow.start_run(run_name=run_name):

        for ds in (ds_train, ds_valid, ds_test):
            ds_ = mlflow.data.from_pandas(
                ds.meta_df.loc[:, STORE_DATA_COLS],
                name=ds.name,
                targets="label",
            )
            # from_pandas(ds.meta_df, name=ds.name)
            mlflow.log_input(dataset=ds_, context=f"{ds.name}_df.json")

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
