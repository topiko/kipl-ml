import os

import dotenv
import hydra
import mlflow
import torch
from kipl_ml.data.wf_dataset import WFDataset, get_train_valid_test
from kipl_ml.logging.logger import get_logger
from kipl_ml.logging.utils import get_mlflow_expr
from kipl_ml.metrics.clf_metrics import Accuracy, ClassRecall, CrossEntropyLoss
from kipl_ml.model_eval.evaluate import evaluate_model
from kipl_ml.models.laserbeak import get_model
from kipl_ml.trace.features import FeatureTrs, log_feature_trs_to_mlflow
from kipl_ml.train.loops import train_model
from omegaconf import DictConfig
from torch.utils.data import DataLoader

logger = get_logger(__name__)

dotenv.load_dotenv()

WORKING_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_DIR_PATH = os.path.join(WORKING_DIR, "config")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")

assert MLFLOW_TRACKING_URI is not None, "MLFLOW_TRACKING_URI must be set in .env file."


@hydra.main(config_path=CONFIG_DIR_PATH, config_name="config", version_base=None)
def main(cfg: DictConfig):
    dataset_name = cfg.dataset.name
    model_name = cfg.model.name
    n_packets = cfg.trace.n_packets

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    experiment_name = cfg.mlflow.experiment_name
    experiment_id = get_mlflow_expr(experiment_name=experiment_name)
    mlflow.set_experiment(experiment_id=experiment_id)
    run_name = model_name + "_" + cfg.features.name

    feature_trs = FeatureTrs(feature_names=cfg.features.features)

    ds_train, ds_valid, ds_test = get_train_valid_test(
        dataset=dataset_name,
        n_samples=(cfg.dataset.n_train_traces, 1000, 1000),
        feature_trs=feature_trs,
        n_packets=n_packets,
    )

    model = get_model(
        model_name,
        n_classes=ds_train.n_classes,
        inputs=ds_train.outputs,
    )

    train_loader = DataLoader(ds_train, batch_size=cfg.train.batch_size, shuffle=True)
    valid_loader = DataLoader(ds_valid, batch_size=32, shuffle=False)
    test_loader = DataLoader(ds_test, batch_size=32, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters())
    loss_fn = torch.nn.CrossEntropyLoss()
    metrics = [Accuracy(), CrossEntropyLoss(), ClassRecall(1)]
    early_stop_metric = "loss"
    patience = 2

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

        mlflow.log_params(dict(cfg))
        mlflow.pytorch.log_model(trained_model, "model")
        log_feature_trs_to_mlflow(feature_trs)

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
