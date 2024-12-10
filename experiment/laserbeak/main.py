import json
import os

import configs as lasereak_configs
import dotenv
import hydra
import mlflow
import torch
from kipl_ml.data import assets
from kipl_ml.data.wf_dataset import get_train_valid_test
from kipl_ml.defences.defences import Defences
from kipl_ml.logging.logger import get_logger
from kipl_ml.logging.utils import get_mlflow_expr
from kipl_ml.metrics.clf_metrics import Accuracy, ClassRecall, CrossEntropyLoss
from kipl_ml.model_eval.evaluate import evaluate_model
from kipl_ml.models.laserbeak import get_model, get_signature
from kipl_ml.trace.features import FeatureTrs
from kipl_ml.train.loops import train_model
from omegaconf import DictConfig
from torch.utils.data import DataLoader

logger = get_logger(__name__)

dotenv.load_dotenv()

WORKING_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_DIR_PATH = os.path.join(WORKING_DIR, "config")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")

assert MLFLOW_TRACKING_URI is not None, "MLFLOW_TRACKING_URI must be set in .env file."
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

FEAT_NAME_MAP = {
    "time_dirs": assets.TIME_DIRS,
    "times_norm": assets.TIMES_MAX_NORMALIZED,
    "cumul_norm": assets.MAX_NORMALIZED_CUM_SIZES,
    "iat_dirs": assets.IAT_DIRS,
    "inv_iat_log_dirs": assets.LOG_INV_IAT_DIRS,
    "running_rates": assets.RUNNING_RATE_SIZES,
}


@hydra.main(config_path=CONFIG_DIR_PATH, config_name="test", version_base=None)
def main(cfg: DictConfig):
    dataset_name = cfg.dataset.name
    model_name = cfg.model.name
    n_packets = cfg.trace.n_packets
    experiment_name = cfg.mlflow.experiment_name
    feature_names = cfg.features.features

    config_path = os.path.join(list(lasereak_configs.__path__)[0], model_name + ".json")

    with open(config_path, "r") as fi:
        model_config = json.load(fi)

    if model_config.get("input_size"):
        n_packets = model_config["input_size"]

    if model_config.get("feature_list"):
        feature_names = [FEAT_NAME_MAP[feat] for feat in model_config["feature_list"]]

    feature_trs = FeatureTrs(feature_names=feature_names, n_packets=n_packets)
    defences = Defences(defences=[dict(d) for d in cfg.defences])

    ds_train, ds_valid, ds_test = get_train_valid_test(
        dataset=dataset_name,
        n_samples=(cfg.dataset.n_train_traces, 1000, 1000),
        feature_trs=feature_trs,
        defences=defences,
    )

    model = get_model(
        model_name,
        n_classes=ds_train.n_classes,
        inputs=ds_train.output_sizes,
        model_config=model_config,
    )

    bs = cfg.train.batch_size
    train_loader = DataLoader(ds_train, batch_size=bs, shuffle=True)
    valid_loader = DataLoader(ds_valid, batch_size=bs, shuffle=False)
    test_loader = DataLoader(ds_test, batch_size=bs, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.lr)
    loss_fn = torch.nn.CrossEntropyLoss()
    metrics = [Accuracy(), CrossEntropyLoss(), ClassRecall(1)]
    early_stop_metric = "loss"
    patience = cfg.train.patience

    experiment_id = get_mlflow_expr(experiment_name=experiment_name)
    mlflow.set_experiment(experiment_id=experiment_id)
    run_name = model_name + "_" + cfg.features.name
    with mlflow.start_run(run_name=run_name):

        trained_model = train_model(
            model=model,
            train_loader=train_loader,
            valid_loader=valid_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            metrics=metrics,
            early_stop_metric=early_stop_metric,
            patience=patience,
        )

        signature = get_signature(model=trained_model, ds=ds_train)
        mlflow.pytorch.log_model(trained_model, "model", signature=signature)
        mlflow.log_table(ds_test.meta_df, "test_df.json")
        mlflow.log_table({"features": cfg.features.features}, "features.json")

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
            }
        )
        for key, loader in zip(["valid", "test"], [valid_loader, test_loader]):
            metrics_vals = evaluate_model(
                model=trained_model,
                dataloader=loader,
                loss_fn=loss_fn,
                metrics=metrics,
            )
            metrics_vals = {f"{key}_{k}": v for k, v in metrics_vals.items()}

            mlflow.log_metrics(metrics_vals, step=0)


if __name__ == "__main__":
    main()
