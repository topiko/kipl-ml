from __future__ import annotations

import argparse
import os

import dotenv
import mlflow
import pandas as pd
from hydra import compose, initialize
from omegaconf import OmegaConf
from torch import nn
from torch.utils.data import DataLoader, Dataset

from experiment.ephemeral_defences.main import (
    _get_defence,
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

    if (parent_runids is None) or (len(parent_runids) == 0):
        raise ValueError("No run found!")

    df = list_runs(experiment_name, only_finished=False, raise_on_empty=False)
    mask = df.loc[:, "tags.mlflow.runName"] == xv_run_name

    if mask.sum() != 1:
        raise ValueError(f"Several or no runs found for name: {xv_run_name}")

    run_id = df.loc[mask, "run_id"].values[0]

    logger.info("Run name: %s -> %s", run_name, parent_runids[0])
    logger.info("Xv run name: %s", xv_run_name)
    logger.info("Run ID: %s", run_id)

    return run_id


def get_model(cfg: OmegaConf, test_xv: int) -> nn.Module:

    run_id = _get_run_id(cfg, test_xv)
    run = mlflow.get_run(run_id=run_id)

    artifact_uri = run.info.artifact_uri

    logger.info("Loading model")
    model = mlflow.pytorch.load_model(artifact_uri + "/model")

    return model


def get_test_set(cfg: OmegaConf, test_xv: int) -> Dataset:

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

    feature_names_remote = config["model"]["features"]

    if any(n1 != n2 for n1, n2 in zip(feature_names, feature_names_remote)):
        raise ValueError("Feature names do not match")

    trace_len = cfg.model.trace_len
    feature_trs = FeatureTrs(feature_names=feature_names, n_packets=trace_len)
    defence_test = _get_defence(cfg)["defence_test"]
    test_ds = WFDataset(
        dataset=f"{dataset}-test",
        label=label,
        meta_df=test_df,
        defence=defence_test,
        feature_trs=feature_trs,
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

    test_ds = get_test_set(cfg_defence, test_xv=test_xv)
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False)

    metrics = [Accuracy()]
    metrics_vals = evaluate_model(
        model=trained_model,
        dataloader=test_loader,
        metrics=metrics,
    )
    return metrics_vals


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
            "breakpad",
            "front-bottleneck",
            "interspace",
            "ephemeral-pad-bottle-sc0.5",
            "no_defence",
            "tamaraw-bottleneck",
            "regulator-bottleneck",
            "ephemeral-block-bottle-sc0.5",
            "ephemeral-block-bottle-sc0.75",
        ]
    elif args.network == "infinite":
        defences = [
            "breakpad",
            "front-infinite",
            "interspace",
            "ephemeral-pad-inf-sc0.75",
            "no_defence",
            "tamaraw-infinite",
            "regulator-infinite",
            "ephemeral-block-inf-sc0.75",
        ]

    defences = ["no_defence", "breakpad"]
    overrides = [f"misc.mlflow.experiment_name={args.experiment_name}-{args.network}"]

    dfs = []

    table_name = f"tables/cross_attack_{args.network}.csv"
    try:
        df = pd.read_csv(table_name)
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
                        logger.info("Already computed for %s", mask.sum())
                        continue

                logger.info("Evaluating %s for xv=%d", defence, xv)
                metrics_vals = get_metrics_for_xv(
                    model_name=args.model,
                    trained_defence=trained_defence,
                    test_defence=defence,
                    overrides=overrides,
                    test_xv=xv,
                )

                df_ = pd.Series(metrics_vals).to_frame().T
                df_.loc[:, "xv"] = xv
                df_.loc[:, "trained_defence"] = trained_defence
                df_.loc[:, "test_defence"] = defence

                dfs.append(df_)

                df = pd.concat(dfs, axis=0)

                print(
                    df.groupby(["trained_defence", "test_defence"]).agg(
                        {"accuracy": ["mean", "std"]}
                    )
                )

                df.to_csv(table_name, index=False)
                print(df)


if __name__ == "__main__":
    main()
