from __future__ import annotations

import argparse
import os

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
    run_id = _get_run_id(cfg, test_xv)
    run = mlflow.get_run(run_id=run_id)

    feature_names = _get_feature_names(cfg)

    # Loading the dataset's source
    logged_dataset = run.inputs.dataset_inputs[2].dataset
    dataset_source = mlflow.data.get_source(logged_dataset)

    try:
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
            logger.warning("\t" + feature_names)
            logger.warning("\t" + feature_names_remote)
    except KeyError:
        pass

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
    test_defence: str,
    overrides: list[str],
    test_xv: int,
) -> dict:

    with initialize(version_base=None, config_path="./config/"):
        cfg_attack = compose(
            config_name=model_name, overrides=overrides + [f"defence={trained_defence}"]
        )
    trained_model = get_model(cfg_attack, test_xv=test_xv)

    with initialize(version_base=None, config_path="./config/"):
        cfg_defence = compose(
            config_name=model_name,
            overrides=overrides + [f"defence={test_defence}"],
        )

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
    if trained_defence == test_defence:
        run_id = _get_run_id(cfg_attack, test_xv=test_xv)
        run = mlflow.get_run(run_id=run_id)
        recorded_acc = run.data.metrics["test_accuracy"]
        current_acc = metrics_vals["accuracy"]
        print(rf"recorded acc = {recorded_acc:.5f} ~ {current_acc:.5f} = current acc ?")
        print()
    return metrics_vals


def _to_pivotet(df: pd.DataFrame) -> pd.DataFrame:
    acc_mean = df.groupby(["trained_defence", "test_defence"]).agg(
        {"accuracy": ["mean", "std"]}
    )
    acc_mean.columns = [c[0] + " " + c[1] for c in acc_mean.columns]
    acc_mean = acc_mean.reset_index()

    acc_mean = acc_mean.pivot_table(
        index="trained_defence",
        columns="test_defence",
        values="accuracy mean",
    )
    return acc_mean


def main():
    parser = argparse.ArgumentParser(description="Cross Attack Script")
    parser.add_argument(
        "--network",
        type=str,
        required=False,
        choices=["infinite", "bottleneck"],
        default="bottleneck",
        help="Network type",
    )
    parser.add_argument("--experiment-name", type=str, required=True)
    parser.add_argument(
        "--model",
        type=str,
        required=False,
        default="df",
        choices=["df", "df-multi", "rf"],
        help="Model type",
    )

    args = parser.parse_args()
    if args.network == "bottleneck":
        defences = [
            "no_defence",
            "breakpad",
            "front-bottleneck",
            "interspace",
            "ephemeral-pad-bottle-sc0.5",
            "tamaraw-bottleneck",
            "regulator-bottleneck",
            "ephemeral-block-bottle-sc0.75",
        ]
    elif args.network == "infinite":
        defences = [
            "no_defence",
            "breakpad",
            "front-infinite",
            "interspace",
            "ephemeral-pad-inf-sc0.75",
            "tamaraw-infinite",
            "regulator-infinite",
            "ephemeral-block-inf-sc0.75",
        ]

    overrides = [
        f"misc.mlflow.experiment_name={args.experiment_name}-{args.network}",
        f"network={args.network}",
    ]
    if "inftrain" in args.experiment_name:
        overrides += ["train.defence_augmentation=0"]

    dfs = []

    table_name = f"cross_attack_{args.experiment_name}-{args.model}-{args.network}.csv"
    try:
        df = pd.read_csv("tables/" + table_name)
    except FileNotFoundError:
        df = None

    for xv in range(5):
        for trained_defence in defences:
            for defence in defences:
                if df is not None:
                    mask = (
                        (df["trained_defence"] == trained_defence)
                        & (df["test_defence"] == defence)
                        & (df["xv"] == xv)
                    )
                    if mask.sum() > 0:
                        continue

                logger.info("Evaluating %s for xv=%d", defence, xv)
                try:
                    metrics_vals = get_metrics_for_xv(
                        model_name=args.model,
                        trained_defence=trained_defence,
                        test_defence=defence,
                        overrides=overrides,
                        test_xv=xv,
                    )
                except ValueError:
                    continue

                df_ = pd.Series(metrics_vals).to_frame().T
                df_.loc[:, "xv"] = xv
                df_.loc[:, "trained_defence"] = trained_defence
                df_.loc[:, "test_defence"] = defence

                dfs.append(df_)

                if df is None:
                    df = pd.concat(dfs, axis=0)
                else:
                    df = pd.concat([df, df_], axis=0)

                acc_mean = _to_pivotet(df)
                print(acc_mean)

                df.to_csv("tables/" + table_name, index=False)

    tr_def = [d for d in defences if d in df.trained_defence.unique()]
    test_def = [d for d in defences if d in df.test_defence.unique()]
    acc_mean = _to_pivotet(df).loc[tr_def, test_def]
    print("==============================")
    print(acc_mean)

    sns.heatmap(acc_mean, annot=True)
    plt.suptitle(f"{args.experiment_name}\nntwk={args.network} | model={args.model}")
    plt.tight_layout()
    plt.savefig(f"figs/{table_name.replace('.csv', '.png')}")


if __name__ == "__main__":
    main()
