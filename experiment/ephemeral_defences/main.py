from __future__ import annotations

import os

import dotenv
import hydra
import mlflow
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
from torchtune.training.lr_schedulers import get_cosine_schedule_with_warmup

from experiment.ephemeral_defences import defence_builder
from kipl_ml.data import assets
from kipl_ml.data.wf_dataset import get_train_valid_test
from kipl_ml.defences.base import NoDefence
from kipl_ml.defences.breakpad import Breakpad
from kipl_ml.defences.front import FRONT
from kipl_ml.defences.interspace import Interspace
from kipl_ml.defences.maybenot import Maybenot
from kipl_ml.logging.logger import get_logger
from kipl_ml.logging.utils import get_mlflow_expr, key_val_fmt, log_multiline
from kipl_ml.metrics.clf_metrics import Accuracy, ClassRecall
from kipl_ml.metrics.defence_netwk_metrics import get_overheads
from kipl_ml.model_eval.evaluate import evaluate_model
from kipl_ml.models.models import get_model
from kipl_ml.models.rf import RFLRScheduler
from kipl_ml.models.utils import get_laserbeak_model_config, get_signature
from kipl_ml.tools.mlflow_utils import (
    get_parent_run_id,
    list_runs,
    log_dataset,
    log_hydra_conf,
    run_exists,
)
from kipl_ml.trace.features import FEAT_NAME_MAP, FeatureTrs
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
    params = {f"lr_scheduler.{k}": v for k, v in dict(cfg.lr_scheduler).items()}

    match cfg.lr_scheduler.scheduler:
        case "cosine":
            return (
                get_cosine_schedule_with_warmup(
                    optimizer,
                    num_warmup_steps=cfg.lr_scheduler.warmup_period,
                    num_training_steps=cfg.lr_scheduler.epochs,
                    num_cycles=0.5,
                    last_epoch=-1,
                ),
                params,
            )

        case "plateau":
            return (
                torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    mode="min",
                    factor=cfg.lr_scheduler.factor,
                    patience=cfg.lr_scheduler.lr_patience,
                ),
                params,
            )
        case "rf-native":
            return (
                RFLRScheduler(
                    optimizer=optimizer,
                    n_epochs=cfg.lr_scheduler.epochs,
                ),
                params,
            )

        case "none":
            return None, params
        case _:
            raise NotImplementedError()


def _get_optimizer(cfg: OmegaConf, model: nn.Module) -> torch.optim.Optimizer:
    match cfg.optimizer.type:
        case "adamw":
            opt_betas = (0.9, 0.999)
            opt_wd = 0.001
            return torch.optim.AdamW(
                model.parameters(),
                lr=cfg.optimizer.lr,
                betas=opt_betas,
                weight_decay=opt_wd,
            )
        case "adamax":
            return torch.optim.Adamax(params=model.parameters(), lr=cfg.optimizer.lr)
        case "adam":
            return torch.optim.Adam(
                params=model.parameters(),
                lr=cfg.optimizer.lr,
                weight_decay=cfg.optimizer.weight_decay,
            )
        case _:
            raise NotImplementedError


def _get_loss(cfg: OmegaConf):
    match cfg.loss.type:
        case "crossentropyloss":
            return torch.nn.CrossEntropyLoss(
                reduction="mean", label_smoothing=cfg.loss.label_smoothing
            )
        case _:
            raise NotImplementedError()


def _get_defence(
    cfg: OmegaConf,
) -> dict[str, Maybenot | FRONT | Interspace | Breakpad | NoDefence]:
    def_type = cfg.defence.type

    match def_type:
        case "no-defence":
            return defence_builder.no_def(cfg)
        case "ephemeral":
            return defence_builder.ephemeral(cfg)
        case "front":
            return defence_builder.front(cfg)
        case "interspace":
            return defence_builder.interspace(cfg)
        case "breakpad":
            return defence_builder.breakpad(cfg)
        case "tamaraw":
            return defence_builder.tamaraw(cfg)
        case "regulator":
            return defence_builder.regulator(cfg)
        case _:
            raise NotImplementedError("no builder for defence '{def_type}'")


def _get_dl(ds: Dataset, bs: int, shuffle: bool = False) -> DataLoader:
    pin_memory = True
    ncpus = os.cpu_count() or 1
    if ncpus == 1:
        logger.warning("Using only 1 CPU for dataloading.")

    num_workers = min(24, ncpus)
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
    return f"{cfg.misc.mlflow.experiment_name}"


def _parse_run_name(cfg: OmegaConf) -> str:
    aug = cfg.train.defence_augmentation

    defence_str = cfg.defence.type

    try:
        defence_str += f"_{cfg.defence.flavor}"
    except AttributeError:
        pass

    if cfg.defence.type == "ephemeral":
        defence_str = (
            f"{defence_str} | {cfg.defence.deck_name} | sc={cfg.defence.scale:.02f}"
        )

    return f"{defence_str} vs. {cfg.model.name} | aug={aug}"


def _get_bw_overhead(cfg: OmegaConf, undefended_trace_len: int) -> int:
    match cfg.defence.type:
        case "ephemeral":
            return int(
                undefended_trace_len * float(1 / (1 - cfg.defence.max_padding_frac))
            )
        case "no-defence":
            return undefended_trace_len
        case "front":
            max_extra_packets = int(
                cfg.defence.padding_budget_max_client
                + cfg.defence.padding_budget_max_server
            )
            return undefended_trace_len + max_extra_packets
        case "interspace":
            logger.warning("What is the bw overhead estimate for Interspace?")
            return undefended_trace_len * 2
        case "breakpad":
            logger.warning("What is the bw overhead estimate for Breakpad?")
            return undefended_trace_len * 2
        case _:
            raise NotImplementedError("Defence type not implemented.")


def _get_model_config(
    cfg: OmegaConf, ret_feature_names: bool = False
) -> dict | list[str]:
    model_name = cfg.model.name
    match cfg.model.source:
        case "local":
            feature_names = cfg.model.features
            trace_len = cfg.model.trace_len
            model_config = {}
            if model_name == "march":
                model_config = {
                    "input_len": cfg.model.trace_len,
                    "step_len": cfg.model.step_len,
                    "step_stride": cfg.model.step_stride,
                    "embed_dim": cfg.model.embed_dim,
                    "vmap": cfg.model.vmap,
                    "rnn_kwargs": {
                        "hidden_size": cfg.model.rnn.hidden_size,
                        "num_layers": cfg.model.rnn.num_layers,
                        "dropout": cfg.model.rnn.dropout,
                    },
                    "cnn_kwargs": {
                        "dropout": cfg.model.cnn.dropout,
                    },
                }
        case "lb":
            model_config = get_laserbeak_model_config(model_name)

            if (trace_len := cfg.model.trace_len) != model_config["input_size"]:
                logger.warning(
                    f"Overriding input_size w. our local trace len for {model_name}"
                )
            model_config["input_size"] = trace_len

            if model_config.get("feature_list"):
                logger.warning("Feature list overwritten by model config")

                feature_names = []
                for feat in model_config["feature_list"]:
                    if feat == "running_rates":
                        logger.warning(
                            "Replacing 'running_rates' with max normalized ones for stability."
                        )
                        feat = "max_normalized_running_rates"

                    feature_names.append(FEAT_NAME_MAP[feat])

        case _:
            raise NotImplementedError

    if ret_feature_names:
        return feature_names

    return model_config


def _get_feature_names(cfg: OmegaConf) -> list[str]:
    return _get_model_config(cfg, ret_feature_names=True)


def _run_xv(
    cfg: OmegaConf,
    experiment_name: str,
    parent_run_name: str,
    test_xv: int,
    nested_run: bool = True,
):
    # Set seeds
    cfg.misc.seed += test_xv
    torch.manual_seed(cfg.misc.seed)
    torch.use_deterministic_algorithms(True)
    np.random.seed(cfg.misc.seed)

    run_name = f"{parent_run_name}_xv={test_xv:02d}"

    if run_exists(experiment_name, run_name, parent_run_name=parent_run_name) and (
        not cfg.misc.ignore_existing
    ):
        logger.info("Found finished run for: %s -> exiting.", run_name)
        return

    logger.info(run_name)
    logger.info("Starting run w. config:")
    log_multiline(OmegaConf.to_yaml(cfg))

    dataset_name = cfg.dataset.name
    model_name = cfg.model.name
    target = _get_target(cfg.dataset.target)
    STORE_DATA_COLS = [target, assets.TRACE_ID]

    feature_names = _get_feature_names(cfg)
    trace_len = cfg.model.trace_len
    feature_trs = FeatureTrs(feature_names=feature_names, n_packets=trace_len)

    loss_fn = _get_loss(cfg)
    metrics = [Accuracy(), ClassRecall(1)]
    early_stop_metric = "loss"
    patience = cfg.train.patience

    ds_train, ds_valid, ds_test = get_train_valid_test(
        dataset=dataset_name,
        n_splits=cfg.dataset.n_splits,
        label=target,
        test_xv=test_xv,
        random_state=cfg.dataset.random_state,
        feature_trs=feature_trs,
        defence_aug=cfg.train.defence_augmentation,
        defence_aug_valid=cfg.train.defence_augmentation_valid,
        n_min_packets=cfg.dataset.n_min_packets,
        **_get_defence(cfg),
    )

    model_config = _get_model_config(cfg)
    model = get_model(
        cfg.model.source,
        model_name,
        n_classes=ds_train.n_classes,
        inputs=ds_train.output_sizes,
        model_config=model_config,
    )

    train_loader = _get_dl(ds_train, cfg.train.batch_size, True)
    valid_loader = _get_dl(ds_valid, 128)
    test_loader = _get_dl(ds_test, 128)

    optimizer = _get_optimizer(cfg, model)
    lr_scheduler, scheduler_params = _get_lr_scheduler(cfg, optimizer)

    with mlflow.start_run(run_name=run_name, nested=nested_run):
        # Log the config file as an artifact
        log_hydra_conf(cfg)

        # Log the most interesting hyp params.
        mlflow.log_params(
            {
                "n_packets": trace_len,
                "model_name": model_name,
                "dataset_name": dataset_name,
                "n_train_traces": ds_train.n_orig_traces,
                "train.early_stop_metric": early_stop_metric,
                "data_random_state": cfg.dataset.random_state,
                "defence_augmentation": cfg.train.defence_augmentation,
                "defence_augmentation_valid": cfg.train.defence_augmentation_valid,
                "test_xv": test_xv,
                "network_state": cfg.network.type,
            }
        )

        try:
            mlflow.log_param("defence.flavor", cfg.defence.flavor)
        except AttributeError:
            pass

        # Log the defence params:
        mlflow.log_params(ds_train.defence.mlflow_log_params())

        # Netwk params:
        mlflow.log_params(ds_train.defence.network_delay_millis.mlflow_log_params())

        # scheduler params:
        mlflow.log_params(scheduler_params)

        # Log the datasets
        for ds in (ds_train, ds_valid, ds_test):
            log_dataset(ds, STORE_DATA_COLS, target)

        # Log the overheads
        defence_overheads = get_overheads(
            test_loader.dataset.defence, test_loader.dataset.meta_df
        )

        if (missing := defence_overheads["sim.missing"]) > 0.0001:
            logger.warning(f"Missing packets ({missing}) in simulation!")

        mlflow.log_metrics(defence_overheads, step=None)

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


def _run_xvs(experiment_name: str, run_name: str, cfg: OmegaConf):
    test_splits = OmegaConf.to_object(cfg.dataset.test_splits)
    orig_seed = cfg.misc.seed
    for test_xv in test_splits:
        _run_xv(cfg, experiment_name, run_name, test_xv)
        # Seed is modified inside _run_xv for each xv split.
        # For consistency, we restore here the original seed.
        cfg.misc.seed = orig_seed


@hydra.main(config_path=CONFIG_DIR_PATH, config_name="config", version_base=None)
def main(cfg: DictConfig):
    experiment_name = _parse_experiment_name(cfg)

    experiment_id = get_mlflow_expr(experiment_name=experiment_name)
    mlflow.set_experiment(experiment_id=experiment_id)
    run_name = _parse_run_name(cfg)

    logger.info("Run: %s", run_name)

    if run_exists(experiment_name, parent_run_name=run_name):
        runids = get_parent_run_id(experiment_name, None, parent_run_name=run_name)
        if (runids is None) or (len(runids) != 1):
            raise ValueError(
                f"Several runids found (for {run_name}) - cannot initiate."
            )

        with mlflow.start_run(run_id=runids[0]):
            _run_xvs(experiment_name, run_name, cfg)
    else:
        with mlflow.start_run(run_name=run_name):
            mlflow.set_tag("project", "ephemeral_defences")
            _run_xvs(experiment_name, run_name, cfg)


if __name__ == "__main__":
    main()
