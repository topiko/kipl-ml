import os

import dotenv
import hydra
import mlflow
import numpy as np
import torch
from kipl_ml.data import assets
from kipl_ml.data.wf_dataset import get_train_valid_test
from kipl_ml.defences.base import NoDefence
from kipl_ml.defences.maybenot import Deck, DeckStats, Maybenot
from kipl_ml.logging.logger import get_logger
from kipl_ml.logging.utils import get_mlflow_expr, key_val_fmt, log_multiline
from kipl_ml.metrics.clf_metrics import Accuracy, ClassRecall
from kipl_ml.model_eval.evaluate import evaluate_model
from kipl_ml.models.laserbeak import get_model, get_signature
from kipl_ml.models.utils import get_laserbeak_model_config
from kipl_ml.tools.mlflow_utils import hydra_run_exists, log_dataset, log_hydra_conf
from kipl_ml.trace.features import FEAT_NAME_MAP, FeatureTrs
from kipl_ml.train.loops import train_model
from omegaconf import DictConfig, OmegaConf
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
from torchtune.training.lr_schedulers import get_cosine_schedule_with_warmup

logger = get_logger(__name__)

dotenv.load_dotenv()

WORKING_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_DIR_PATH = os.path.join(WORKING_DIR, "config")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")

assert MLFLOW_TRACKING_URI is not None, "MLFLOW_TRACKING_URI must be set in .env file."
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)


BASE_MODEL_RUNEXPIDS = {
    "df-multi": ("09fd713aebed4450a7ea43d492b00c3d", "581319079988312243")
}


def _fetch_base_model(model_name: str) -> torch.nn.Module:
    run_id, expr_id = BASE_MODEL_RUNEXPIDS[model_name]
    mlflow.set_experiment(experiment_id=expr_id)

    run = mlflow.get_run(run_id)

    return mlflow.pytorch.load_model(run.info.artifact_uri + "/model")


def _get_lr_scheduler(
    cfg: OmegaConf, optimizer: torch.optim.Optimizer, train_loader: DataLoader
) -> ReduceLROnPlateau | LambdaLR | None:

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
            optimizer, mode="min", factor=0.2, patience=5 * len(train_loader)
        )
    else:
        scheduler = None

    return scheduler


def _get_dl(ds: Dataset, bs: int, shuffle: bool = False) -> DataLoader:
    pin_memory = True
    num_workers = 16
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

    raise KeyError(f"provided target '{target}' nno valid!")


@hydra.main(config_path=CONFIG_DIR_PATH, config_name="test", version_base=None)
def main(cfg: DictConfig):
    logger.info("Starting run w. config:")
    log_multiline(OmegaConf.to_yaml(cfg))

    dataset_name = cfg.dataset.name
    model_name = cfg.model.name
    n_packets = cfg.trace.n_packets
    experiment_name = cfg.mlflow.experiment_name + "->" + model_name + " vs. maybenot"
    target = _get_target(cfg.dataset.target)

    STORE_DATA_COLS = [target, assets.TRACE_ID]

    if run_id := hydra_run_exists(experiment_name, cfg):
        logger.info("Run ('%s') already exists and is finished for. Skipping.", run_id)
        return

    # Set seeds
    torch.manual_seed(cfg.seed)
    # torch.use_deterministic_algorithms(True)
    np.random.seed(cfg.seed)

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
    ds_train, ds_valid, ds_test = get_train_valid_test(
        dataset=dataset_name,
        n_splits=cfg.dataset.n_splits,
        label=target,
        test_xv=cfg.dataset.test_xv,
        random_state=cfg.dataset.random_state,
        feature_trs=feature_trs,
        defence_aug=cfg.dataset.defence_augmentation,
    )

    netwk_delay = (
        cfg.network.network_delay_millis.min,
        cfg.network.network_delay_millis.max,
    )

    maybenot_config = dict(cfg.defences)
    maybenot_config["network_delay_millis"] = netwk_delay
    deck_path = maybenot_config.pop("deck_path")
    if maybenot_config.pop("name") != "maybenot":
        raise ValueError("Defence must be 'maybenot'")

    if (n_machines := maybenot_config.pop("n_machines")) == 0:
        defence_train = NoDefence()
        defence_valid = defence_train
        defence_test = defence_train
    else:
        rng = np.random.default_rng(seed=cfg.seed)

        deck_stats = DeckStats.load(deck_path)
        machine_idxs = list(
            rng.choice(deck_stats.n_machines, size=n_machines, replace=False)
        )

        maybenot_config["deck"] = Deck(deck_stats, machine_idxs)

        defence_train = Maybenot(**maybenot_config)

        if len(machine_idxs) > ds_train.n_orig_traces:
            logger.info(
                f"Number of machines ({len(machine_idxs)}) exceeds number of training traces ({cfg.dataset.n_train_traces}) --> consider as infinite machine limit."
            )

            test_machines = list(
                set(range(deck_stats.n_machines)).difference(set(machine_idxs))
            )

            maybenot_config["deck"] = Deck(deck_stats, test_machines)
            defence_valid = Maybenot(**maybenot_config)
            defence_test = Maybenot(**maybenot_config)
        else:
            defence_valid = defence_train
            defence_test = defence_train

    for ds, def_ in zip(
        (ds_train, ds_valid, ds_test), (defence_train, defence_valid, defence_test)
    ):
        ds.defence = def_
        ds.report(to_log=True)

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
        model.parameters(), lr=cfg.train.lr, betas=opt_betas, weight_decay=opt_wd
    )

    lr_scheduler = _get_lr_scheduler(cfg, optimizer, train_loader)

    loss_fn = torch.nn.CrossEntropyLoss(
        reduction="mean", label_smoothing=cfg.train.label_smoothing
    )
    metrics = [Accuracy(), ClassRecall(1)]
    early_stop_metric = "loss"
    patience = cfg.train.patience

    experiment_id = get_mlflow_expr(experiment_name=experiment_name)
    mlflow.set_experiment(experiment_id=experiment_id)
    run_name = f"{n_machines} state-machines"
    with mlflow.start_run(run_name=run_name):

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
                "n_maybenot_machines": n_machines,
                "max_padding_frac": cfg.defences.max_padding_frac,
                "network_delay_millis_min": netwk_delay[0],
                "network_delay_millis_max": netwk_delay[1],
            }
        )

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


if __name__ == "__main__":
    main()
