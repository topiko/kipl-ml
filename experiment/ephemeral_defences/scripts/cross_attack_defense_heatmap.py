from __future__ import annotations

import argparse
import os
from itertools import product

import dotenv
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from hydra import compose, initialize
from omegaconf import OmegaConf
from torch import nn

from experiment.ephemeral_defences.main import (
    _get_defence,
    _get_dl,
    _get_feature_names,
    _get_target,
    _parse_run_name,
)
from kipl_ml.data import assets
from kipl_ml.data.utils import load_dataset_meta_df
from kipl_ml.data.wf_dataset import WFDataset
from kipl_ml.logging.logger import get_logger
from kipl_ml.logging.utils import get_mlflow_expr
from kipl_ml.metrics.clf_metrics import Accuracy
from kipl_ml.model_eval.evaluate import evaluate_model
from kipl_ml.tools.mlflow_utils import (
    get_parent_run_id,
    list_runs,
)
from kipl_ml.trace.features import FeatureTrs

logger = get_logger(__name__)

dotenv.load_dotenv()
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)


def _get_run_id(cfg: OmegaConf, test_xv: int) -> str:

    experiment_name = cfg.misc.mlflow.experiment_name
    experiment_id = get_mlflow_expr(experiment_name=experiment_name)
    mlflow.set_experiment(experiment_id=experiment_id)

    run_name = _parse_run_name(cfg)
    xv_run_name = f"{run_name}_xv={test_xv:02d}"
    parent_runids = get_parent_run_id(
        experiment_name, xv_run_name, parent_run_name=run_name
    )

    logger.info("Run name: %s", run_name)
    logger.info("Xv run name: %s", xv_run_name)
    if (parent_runids is None) or (len(parent_runids) == 0):
        raise ValueError("No run found!")

    df = list_runs(experiment_name, only_finished=False, raise_on_empty=False)
    mask = df.loc[:, "tags.mlflow.runName"] == xv_run_name

    if mask.sum() == 0:
        raise ValueError(f"No runs found for name: {xv_run_name}")
    if mask.sum() != 1:
        raise ValueError(f"Several runs found for name: {xv_run_name}")

    run_id = df.loc[mask, "run_id"].values[0]

    logger.info("Run ID: %s", run_id)

    return run_id


def _checks(cfg: OmegaConf, test_xv: int, feature_names: list[str]):
    # Loading the dataset's source

    try:
        run_id = _get_run_id(cfg, test_xv)
    except ValueError as e:
        logger.warning("For test set generation cannot verify orig. conf params.")
        logger.warning(e)

        return

    run = mlflow.get_run(run_id=run_id)
    logged_dataset = run.inputs.dataset_inputs[2].dataset

    try:
        dataset_source = mlflow.data.get_source(logged_dataset)
        local_dataset = dataset_source.load()
        print(local_dataset)
        # Work w. this local dataset to expose the original asset.TRACE_ID s
    except NotImplementedError:
        logger.warning("Cannot verify that test sets are matching.")

    config = mlflow.artifacts.load_dict(run.info.artifact_uri + "/hydra_config.json")

    try:
        feature_names_remote = config["model"]["features"]

        if any(n1 != n2 for n1, n2 in zip(feature_names, feature_names_remote)):
            logger.warning("Feature names do not match")
            logger.warning("\t" + str(feature_names))
            logger.warning("\t" + str(feature_names_remote))
    except KeyError:
        pass


def get_model(cfg: OmegaConf, test_xv: int) -> nn.Module:

    run_id = _get_run_id(cfg, test_xv)
    run = mlflow.get_run(run_id=run_id)

    artifact_uri = run.info.artifact_uri

    logger.info("Loading model")
    model = mlflow.pytorch.load_model(artifact_uri + "/model")

    return model


def get_test_set(cfg: OmegaConf, test_xv: int) -> WFDataset:

    dataset = cfg.dataset.name
    meta_df = load_dataset_meta_df(dataset)
    n_splits = cfg.dataset.n_splits
    label = _get_target(cfg.dataset.target)

    col = assets.XV_SPLIT(n_splits, label)

    test_df = meta_df[meta_df[col] == test_xv]

    feature_names = _get_feature_names(cfg)

    _checks(cfg, test_xv, feature_names)

    trace_len = cfg.model.trace_len
    feature_trs = FeatureTrs(feature_names=feature_names, n_packets=trace_len)
    defence_test = _get_defence(cfg)["defence_test"]
    test_ds = WFDataset(
        dataset=f"{dataset}-test",
        label=label,
        meta_df=test_df,
        defence=defence_test,
        feature_trs=feature_trs,
        defence_aug=cfg.train.defence_augmentation_valid,
    )

    return test_ds


def get_metrics_for_xv(
    model_name: str,
    trained_defence: str,
    trained_netwk: str,
    test_defence: str,
    test_netwk: str,
    overrides: list[str],
    test_xv: int,
    experiment_name: str,
) -> dict:

    expr_name = "{}-{}"
    with initialize(version_base=None, config_path="../config/"):
        overrides_ = overrides + [
            f"defence={trained_defence}",
            f"misc.mlflow.experiment_name={experiment_name}-{trained_netwk}",
            f"network={trained_netwk}",
        ]
        cfg_attack = compose(config_name=model_name, overrides=overrides_)

    trained_model = get_model(cfg_attack, test_xv=test_xv)

    with initialize(version_base=None, config_path="../config/"):
        expr_name = expr_name.format(experiment_name, test_defence)
        overrides_ = overrides + [
            f"defence={test_defence}",
            f"misc.mlflow.experiment_name={experiment_name}-{test_netwk}",
            f"network={test_netwk}",
        ]
        cfg_defence = compose(config_name=model_name, overrides=overrides_)

    # In the original scripts the seed is modified per xv.
    cfg_defence.misc.seed += test_xv

    torch.manual_seed(cfg_defence.misc.seed)
    torch.use_deterministic_algorithms(True)
    np.random.seed(cfg_defence.misc.seed)

    test_ds = get_test_set(cfg_defence, test_xv=test_xv)

    test_ds.report()
    test_loader = _get_dl(test_ds, 128)

    metrics = [Accuracy()]
    metrics_vals = evaluate_model(
        model=trained_model,
        dataloader=test_loader,
        metrics=metrics,
    )
    if (trained_defence == test_defence) and (trained_netwk == test_netwk):
        run_id = _get_run_id(cfg_attack, test_xv=test_xv)
        run = mlflow.get_run(run_id=run_id)
        recorded_acc = run.data.metrics["test_accuracy"]
        current_acc = metrics_vals["accuracy"]
        print(rf"recorded acc = {recorded_acc:.5f} ~ {current_acc:.5f} = current acc ?")
        print()
    return metrics_vals


def _to_pivotet(df: pd.DataFrame) -> pd.DataFrame:
    acc_mean = df.groupby(
        ["trained_defence", "trained_netwk", "test_defence", "test_netwk"]
    ).agg({"accuracy": ["mean", "std"]})
    acc_mean.columns = [c[0] + " " + c[1] for c in acc_mean.columns]
    acc_mean = acc_mean.reset_index()

    print(acc_mean)

    acc_mean.loc[:, "tr-def-netwk"] = acc_mean.loc[
        :, ["trained_defence", "trained_netwk"]
    ].apply(lambda x: f"{x.iloc[0]}-{x.iloc[1]}", axis=1)
    acc_mean.loc[:, "ts-def-netwk"] = acc_mean.loc[
        :, ["test_defence", "test_netwk"]
    ].apply(lambda x: f"{x.iloc[0]}-{x.iloc[1]}", axis=1)

    acc_mean = acc_mean.pivot_table(
        index="tr-def-netwk",
        columns="ts-def-netwk",
        values="accuracy mean",
    )
    return acc_mean


def main():
    parser = argparse.ArgumentParser(description="Cross Attack Script")
    parser.add_argument(
        "--network",
        type=str,
        required=False,
        choices=["infinite", "bottleneck", "both"],
        default="bottleneck",
        help="Network type",
    )
    parser.add_argument("--experiment-name", type=str, required=True)
    parser.add_argument(
        "--model",
        type=str,
        required=False,
        default="df",
        choices=["df", "df-multi", "rf", "laserbeak"],
        help="Model type",
    )

    args = parser.parse_args()

    defences = [
        "no_defence",
        "breakpad",
        "front",
        "interspace",
        "ephemeral-pad",
        "tamaraw",
        "regulator",
        "ephemeral-block",
    ]

    networks = [args.network] if args.network != "both" else ["infinite", "bottleneck"]

    overrides = []
    if "inftrain" in args.experiment_name:
        overrides += ["train.defence_augmentation=0"]

    def _defence_name_map(defence: str, network: str) -> str:
        match network:
            case "infinite":
                match defence:
                    case "no_defence" | "breakpad" | "interspace":
                        return defence
                    case "front" | "tamaraw" | "regulator":
                        return f"{defence}-infinite"
                    case "ephemeral-pad":
                        return "ephemeral-pad-inf-sc0.75"
                    case "ephemeral-block":
                        return "ephemeral-block-inf-sc0.75"
                    case _:
                        raise ValueError(f"Invalid defence {defence}")
            case "bottleneck":
                match defence:
                    case "no_defence" | "breakpad" | "interspace":
                        return defence
                    case "front" | "tamaraw" | "regulator":
                        return f"{defence}-bottleneck"
                    case "ephemeral-pad":
                        return "ephemeral-pad-bottle-sc0.5"
                    case "ephemeral-block":
                        return "ephemeral-block-bottle-sc0.75"
                    case _:
                        raise ValueError(f"Invalid defence {defence}")
            case _:
                raise ValueError("invalid network")

    table_name = f"cross_attack_{args.experiment_name}-{args.model}-{args.network}.csv"
    try:
        df = pd.read_csv("tables/" + table_name)
    except FileNotFoundError as e:
        logger.info("No previous results found, starting fresh.")
        logger.warning(e)
        df = None

    for xv in range(5):
        for trained_defence_, trained_netwk in product(defences, networks):
            for test_defence, test_netwk in product(defences, networks):
                trained_defence = _defence_name_map(trained_defence_, trained_netwk)
                test_defence = _defence_name_map(test_defence, test_netwk)

                logger.info(
                    f"Now running: {trained_defence} - {trained_netwk} | {test_defence} - {test_netwk}"
                )
                if df is not None:
                    mask = (
                        (df["trained_defence"] == trained_defence)
                        & (df["trained_netwk"] == trained_netwk)
                        & (df["test_defence"] == test_defence)
                        & (df["test_netwk"] == test_netwk)
                        & (df["xv"] == xv)
                    )
                    if mask.sum() > 0:
                        logger.info("Already done!")
                        continue

                logger.info("Evaluating %s for xv=%d", test_defence, xv)

                try:
                    metrics_vals = get_metrics_for_xv(
                        model_name=args.model,
                        trained_defence=trained_defence,
                        trained_netwk=trained_netwk,
                        test_defence=test_defence,
                        test_netwk=test_netwk,
                        overrides=overrides,
                        test_xv=xv,
                        experiment_name=args.experiment_name,
                    )
                except ValueError as e:
                    logger.warning(e)
                    continue

                df_ = pd.Series(metrics_vals).to_frame().T
                df_.loc[:, "xv"] = xv
                df_.loc[:, "trained_defence"] = trained_defence
                df_.loc[:, "trained_netwk"] = trained_netwk
                df_.loc[:, "test_defence"] = test_defence
                df_.loc[:, "test_netwk"] = test_netwk

                if df is None:
                    df = df_
                else:
                    df = pd.concat([df, df_], axis=0)

                acc_mean = _to_pivotet(df)
                print(acc_mean)

                df.to_csv("tables/" + table_name, index=False)

    acc_mean = _to_pivotet(df)
    print("==============================")
    print(acc_mean)

    _, ax = plt.subplots(figsize=(12, 12))
    sns.heatmap(acc_mean, annot=True, ax=ax, annot_kws={"fontsize": 8})
    plt.suptitle(f"{args.experiment_name}\nntwk={args.network} | model={args.model}")
    plt.tight_layout()
    plt.savefig(f"figs/{table_name.replace('.csv', '.png')}")


if __name__ == "__main__":
    main()
