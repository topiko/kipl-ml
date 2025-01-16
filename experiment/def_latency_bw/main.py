import os

import dotenv
import hydra
import mlflow
import torch
from kipl_ml.data.wf_dataset import get_train_valid_test
from kipl_ml.defences.defences import StackedDefence
from kipl_ml.defences.naive import Chi2Delays, RandomPadding
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


FRACTIONS = [0, 0.01, 0.02, 0.05, 0.1, 0.2]
KS = [0, 1, 2, 3, 4, 5]


@hydra.main(config_path=CONFIG_DIR_PATH, config_name="config", version_base=None)
def main(cfg: DictConfig):
    dataset_name = cfg.dataset.name
    model_name = cfg.model.name
    n_packets = cfg.trace.n_packets
    experiment_name = cfg.mlflow.experiment_name

    experiment_id = get_mlflow_expr(experiment_name=experiment_name)
    mlflow.set_experiment(experiment_id=experiment_id)
    run_name = model_name + "_" + cfg.features.name

    def single_run(defences: StackedDefence):
        feature_trs = FeatureTrs(
            feature_names=cfg.features.features,
            n_packets=int(n_packets * (1 + fraction)),
        )

        ds_train, ds_valid, ds_test = get_train_valid_test(
            dataset=dataset_name,
            n_samples=(cfg.dataset.n_train_traces, 1000, 1000),
            random_state=cfg.dataset.random_state,
            feature_trs=feature_trs,
            defences=defences,
        )

        model = get_model(
            model_name,
            n_classes=ds_train.n_classes,
            inputs=ds_train.output_sizes,
        )

        bs = cfg.train.batch_size
        train_loader = DataLoader(ds_train, batch_size=bs, shuffle=True)
        valid_loader = DataLoader(ds_valid, batch_size=bs, shuffle=False)
        test_loader = DataLoader(ds_test, batch_size=bs, shuffle=False)

        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.lr)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5
        )
        loss_fn = torch.nn.CrossEntropyLoss()
        metrics = [Accuracy()]
        early_stop_metric = "loss"
        patience = cfg.train.patience

        trained_model = train_model(
            model=model,
            train_loader=train_loader,
            valid_loader=valid_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            metrics=metrics,
            lr_scheduler=lr_scheduler,
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
                "def_padding_fraction": fraction,
                "def_chi2_df": k,
            }
        )
        for key, loader in zip(["valid", "test"], [valid_loader, test_loader]):
            metrics_vals = evaluate_model(
                model=trained_model,
                dataloader=loader,
                loss_fn=loss_fn,
                metrics=metrics,
            )
            metrics_vals = {f"final:{key}_{k}": v for k, v in metrics_vals.items()}

            mlflow.log_metrics(metrics_vals)

    with mlflow.start_run(run_name=run_name):
        for fraction in FRACTIONS:
            for k in KS:
                defences = StackedDefence(
                    [Chi2Delays(k=k), RandomPadding(fraction=fraction)]
                )

                run_name = f"chi2={k}_pad={fraction}"
                with mlflow.start_run(nested=True, run_name=run_name):
                    single_run(defences)


if __name__ == "__main__":
    main()
