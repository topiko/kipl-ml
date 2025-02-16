from __future__ import annotations

import os

import dotenv
import hydra
import mlflow
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
from torchtune.training.lr_schedulers import get_cosine_schedule_with_warmup

from experiment.ephemeral_defences import defence_builder
from kipl_ml.data import assets
from kipl_ml.data.wf_dataset import WFDataset, get_train_valid_test
from kipl_ml.defences.base import _Def
from kipl_ml.logging.logger import get_logger
from kipl_ml.logging.utils import get_mlflow_expr, key_val_fmt, log_multiline
from kipl_ml.metrics.clf_metrics import Accuracy, ClassRecall
from kipl_ml.model_eval.evaluate import evaluate_model
from kipl_ml.models.laserbeak import get_model, get_signature
from kipl_ml.models.utils import get_laserbeak_model_config
from kipl_ml.tools.mlflow_utils import (
    list_runs,
    log_dataset,
    log_hydra_conf,
)
from kipl_ml.trace.features import FEAT_NAME_MAP, FeatureTrs
from kipl_ml.trace.params import MAX_TRACE_LENGTH
from kipl_ml.train.loops import train_model

logger = get_logger(__name__)

dotenv.load_dotenv()

WORKING_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_DIR_PATH = os.path.join(WORKING_DIR, "config")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")

assert MLFLOW_TRACKING_URI is not None, "MLFLOW_TRACKING_URI must be set in .env file."
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)


def _get_parent(
    experiment_name: str, run_name: str, only_finished: bool = False
) -> str | None:
    df = list_runs(experiment_name, only_finished=only_finished)

    mask = df.loc[:, "tags.mlflow.runName"] == run_name

    if mask.sum() == 0:
        return None

    parent_run_id = df.loc[mask, "tags.mlflow.parentRunId"].iloc[0]

    return parent_run_id


def _get_lr_scheduler(
    cfg: OmegaConf, optimizer: torch.optim.Optimizer
) -> tuple[ReduceLROnPlateau | LambdaLR | None, dict]:

    params = dict(cfg.train)
    try:
        scheduler = cfg.train.scheduler
    except AttributeError:
        return None, params

    if scheduler == "cosine":
        warmup_period = cfg.train.warmup_period
        epochs = cfg.train.epochs
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_period,
            num_training_steps=epochs,
            num_cycles=0.5,
            last_epoch=-1,
        )

    elif scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=cfg.train.factor,
            patience=cfg.train.lr_patience,
        )

    return scheduler, params


def _get_defence(cfg: OmegaConf, netwk_delay: tuple[int, int]) -> dict[str, _Def]:

    def_type = cfg.defence.type

    if def_type == "no-defence":
        return defence_builder.no_def(netwk_delay)
    if def_type == "maybenot":
        return defence_builder.maybenot(cfg, netwk_delay)
    if def_type == "front":
        return defence_builder.front(cfg, netwk_delay)
    if def_type == "interspace":
        return defence_builder.interspace(cfg, netwk_delay)
    if def_type == "breakpad":
        return defence_builder.breakpad(netwk_delay)

    raise NotImplementedError("no builder for defence '{def_type}'")


def _get_dl(ds: Dataset, bs: int, shuffle: bool = False) -> DataLoader:
    pin_memory = True
    num_workers = 24
    return DataLoader(
        ds,
        batch_size=bs,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


def _get_target(target: str) -> str:

    if target in (assets.PAGE_LABEL, assets.SUB_PAGE_LABEL):
        return target

    raise KeyError(f"provided target '{target}' is not valid!")


def _parse_experiment_name(cfg: OmegaConf) -> str:

    return f"{cfg.mlflow.experiment_name}->{cfg.model.name} vs. {cfg.defence.type}"


def _parse_run_name(cfg: OmegaConf) -> str:

    def_type = cfg.defence.type
    aug = cfg.dataset.defence_augmentation
    if def_type == "no-defence":
        return f"NoDefence, aug={aug}"
    if def_type == "maybenot":
        return f"Maybenot w. {cfg.defence.n_machines:04d}, aug={aug}"
    if def_type == "front":
        return f"Front, aug={aug}"
    if def_type == "interspace":
        return f"Interspace, aug={aug}"
    if def_type == "breakpad":
        return f"Breakpad, aug={aug}"

    raise KeyError(f"Defence : '{def_type}' is unavailable.")


def _get_bw_overhead(cfg: OmegaConf, undefended_trace_len: int) -> int:
    if cfg.defence.type == "maybenot":
        return undefended_trace_len * float(1 / (1 - cfg.defence.max_padding_frac))
    if cfg.defence.type == "no-defence":
        return undefended_trace_len
    if cfg.defence.type == "front":
        max_extra_packets = (
            cfg.defence.padding_budget_max_client
            + cfg.defence.padding_budget_max_server
        )

        return undefended_trace_len + max_extra_packets
    if cfg.defence.type == "interspace":
        logger.warning("What is the bw overhead estimate for Interspace?")
        return undefended_trace_len * 2
    if cfg.defence.type == "breakpad":
        logger.warning("What is the bw overhead estimate for Breakpad?")
        return undefended_trace_len * 2

    raise NotImplementedError("Only maybenot and no-defence known.")


def _run_xv(
    cfg: OmegaConf, parent_run_name: str, test_xv: int, nested_run: bool = True
):

    run_name = f"{parent_run_name}_xv={test_xv:02d}"

    df = list_runs(_parse_experiment_name(cfg), only_finished=True)

    if len(df) > 0:
        mask = run_name == df.loc[:, "tags.mlflow.runName"]
        if mask.sum() > 0:
            logger.info(f"Found finished run for: {run_name} -> exiting.")
            return

    logger.info("Starting run w. config:")
    log_multiline(OmegaConf.to_yaml(cfg))

    dataset_name = cfg.dataset.name
    model_name = cfg.model.name
    target = _get_target(cfg.dataset.target)

    model_config = get_laserbeak_model_config(model_name)
    model_config["input_size"] = _get_bw_overhead(cfg, int(model_config["input_size"]))

    if (n_packets := model_config["input_size"]) > MAX_TRACE_LENGTH:
        raise ValueError("Inpu len larger than MAX_TRACE_LENGTH...")

    feature_names = cfg.features.features
    if model_config.get("feature_list"):
        logger.warning("Feature list overwritten by model config")
        feature_names = [FEAT_NAME_MAP[feat] for feat in model_config["feature_list"]]

    feature_trs = FeatureTrs(feature_names=feature_names, n_packets=n_packets)

    netwk_delay = (
        cfg.network.network_delay_millis.min,
        cfg.network.network_delay_millis.max,
    )

    def _get_datasets(test_xv: int) -> tuple[WFDataset, WFDataset, WFDataset]:
        return get_train_valid_test(
            dataset=dataset_name,
            n_splits=cfg.dataset.n_splits,
            label=target,
            test_xv=test_xv,
            random_state=cfg.dataset.random_state,
            feature_trs=feature_trs,
            defence_aug=cfg.dataset.defence_augmentation,
            defence_aug_valid=cfg.dataset.defence_augmentation_valid,
            **_get_defence(cfg, netwk_delay),
        )

    STORE_DATA_COLS = [target, assets.TRACE_ID]

    loss_fn = torch.nn.CrossEntropyLoss(
        reduction="mean", label_smoothing=cfg.train.label_smoothing
    )
    metrics = [Accuracy(), ClassRecall(1)]
    early_stop_metric = "loss"
    patience = cfg.train.patience

    ds_train, ds_valid, ds_test = _get_datasets(test_xv)

    model = get_model(
        model_name,
        n_classes=ds_train.n_classes,
        inputs=ds_train.output_sizes,
        model_config=model_config,
    )

    train_loader = _get_dl(ds_train, cfg.train.batch_size, True)
    valid_loader = _get_dl(ds_valid, 128)
    test_loader = _get_dl(ds_test, 128)

    opt_betas = (0.9, 0.999)
    opt_wd = 0.001
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.train.lr,
        betas=opt_betas,
        weight_decay=opt_wd,
    )

    lr_scheduler, scheduler_params = _get_lr_scheduler(cfg, optimizer)

    with mlflow.start_run(run_name=run_name, nested=nested_run):

        # Log the datasets
        for ds in (ds_train, ds_valid, ds_test):
            log_dataset(ds, STORE_DATA_COLS, target)

        # Log the config file as an artifact
        log_hydra_conf(cfg)

        # Log the most interesting hyp params.
        mlflow.log_params(
            {
                "feature_names": cfg.features.features,
                "n_packets": n_packets,
                "model_name": model_name,
                "dataset_name": dataset_name,
                "n_train_traces": ds_train.n_orig_traces,
                "batch_size": cfg.train.batch_size,
                "patience": patience,
                "early_stop_metric": early_stop_metric,
                "data_random_state": cfg.dataset.random_state,
                "scheduler": cfg.train.scheduler,
                "lr": cfg.train.lr,
                "defence_augmentation": cfg.dataset.defence_augmentation,
                "n_epochs": cfg.train.n_epochs,
                "test_xv": test_xv,
            }
        )

        # Log the defence params:
        mlflow.log_params(ds_train.defence.mlflow_log_params())

        # Netwk params:
        mlflow.log_params(ds_train.defence.network_delay_millis.mlflow_log_params())

        # scheduler params:
        mlflow.log_params(scheduler_params)

        # Train model.
        trained_model = train_model(
            model=model,
            train_loader=train_loader,
            valid_loader=valid_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            metrics=metrics,
            early_stop_metric=early_stop_metric,
            lr_scheduler=lr_scheduler,
            patience=patience,
            n_epochs=None if cfg.train.n_epochs == -1 else cfg.train.n_epochs,
        )

        # Evaluate on test set and log.
        metrics_vals = evaluate_model(
            model=trained_model,
            dataloader=test_loader,
            loss_fn=loss_fn,
            metrics=metrics,
        )
        for k, v in metrics_vals.items():
            logger.info(key_val_fmt(k, f"{v:1.4f}", suffix=""))
        metrics_vals = {f"test_{k}": v for k, v in metrics_vals.items()}
        mlflow.log_metrics(metrics_vals, step=None)

        # log model.
        signature = get_signature(model=trained_model, ds=ds_train)
        mlflow.pytorch.log_model(trained_model, "model", signature=signature)
        mlflow.log_table({"features": cfg.features.features}, "features.json")


@hydra.main(config_path=CONFIG_DIR_PATH, config_name="test", version_base=None)
def main(cfg: DictConfig):
    experiment_name = _parse_experiment_name(cfg)

    # Set seeds
    seed = cfg.seed + cfg.dataset.test_xv
    torch.manual_seed(seed)
    # torch.use_deterministic_algorithms(True)
    np.random.seed(seed)

    experiment_id = get_mlflow_expr(experiment_name=experiment_name)
    mlflow.set_experiment(experiment_id=experiment_id)
    run_name = _parse_run_name(cfg)

    if cfg.dataset.test_xv == -1:
        with mlflow.start_run(run_name=run_name):
            mlflow.set_tag("project", "ephemeral_defences")
            for test_xv in range(10):
                _run_xv(cfg, run_name, test_xv)

    else:
        _run_xv(cfg, run_name, cfg.dataset.test_xv, nested_run=False)


if __name__ == "__main__":
    main()
