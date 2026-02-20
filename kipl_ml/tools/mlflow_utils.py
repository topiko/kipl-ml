import os
import re

import dotenv
import mlflow
import numpy as np
import pandas as pd
from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient
from omegaconf import OmegaConf

from kipl_ml.data.wf_dataset import WFDataset
from kipl_ml.logging.logger import get_logger

dotenv.load_dotenv()
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
if MLFLOW_TRACKING_URI:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

logger = get_logger(__name__)


def set_tracking_uri_from_env() -> None:
    dotenv.load_dotenv()
    if tracking_uri := os.getenv("MLFLOW_TRACKING_URI"):
        mlflow.set_tracking_uri(tracking_uri)


def list_runs(
    experiment_names: list[str] | str | None = None,
    only_finished: bool = True,
    raise_on_empty: bool = True,
    parents: bool = True,
) -> pd.DataFrame:
    if experiment_names is None:
        experiments = mlflow.search_experiments(view_type=ViewType.ALL)
        experiment_names = [ex.name for ex in experiments]
    elif isinstance(experiment_names, str):
        experiment_names = [experiment_names]

    runs = mlflow.search_runs(
        search_all_experiments=True,
        run_view_type=ViewType.ACTIVE_ONLY,
        experiment_names=experiment_names,
        output_format="pandas",
    )

    assert isinstance(runs, pd.DataFrame)

    if only_finished:
        mask = runs.loc[:, "status"] == "FINISHED"
        runs = runs[mask]

    if raise_on_empty and (len(runs) == 0):
        raise ValueError("Empty df.")

    return runs.reset_index(drop=True)


def log_dataset(
    ds: WFDataset,
    store_cols: list[str],
    target: str,
    predictions: np.ndarray | None = None,
):
    df = ds.meta_df.loc[:, store_cols]

    pred_col = "predictions" if predictions is not None else None
    if predictions is not None:
        df.loc[:, pred_col] = predictions

    ds_ = mlflow.data.from_pandas(
        df,
        name=ds.name,
        targets=target,
        predictions=pred_col,
    )
    mlflow.log_input(dataset=ds_, context=f"{ds.name}_df.json")


def log_hydra_conf(cfg: OmegaConf):
    d = OmegaConf.to_container(cfg, resolve=True)
    mlflow.log_dict(d, artifact_file="hydra_config.json")


def get_parent_run_id(
    experiment_name: str, run_name: str | None, parent_run_name: str | None
) -> list[str] | None:
    df = list_runs(experiment_name, only_finished=False, raise_on_empty=False)

    if run_name is not None:
        mask = df.loc[:, "tags.mlflow.runName"] == run_name
    else:
        mask = np.ones(len(df), dtype=bool)

    parents = df.loc[mask, "tags.mlflow.parentRunId"].unique()
    parents = parents[parents != None]

    if parent_run_name is not None:
        mask = df.loc[:, "tags.mlflow.runName"] == parent_run_name
        parents = df.loc[mask, "run_id"]

    if len(parents) == 0:
        return None

    return list(parents)


def run_exists(
    experiment_name: str,
    run_name: str | None = None,
    parent_run_name: str | None = None,
    ignore_existing: bool = False,
) -> bool:
    df = list_runs(experiment_name, only_finished=False, raise_on_empty=False)

    if len(df) == 0:
        return False

    if run_name is not None:
        mask = run_name == df.loc[:, "tags.mlflow.runName"]
        if mask.sum() == 0:
            return False

        if parent_run_name is None:
            if mask.sum() == 1:
                return True
            if mask.sum() > 1:
                raise ValueError(f"Several runs found for name: {run_name}")
            return False

        if (
            parents := get_parent_run_id(experiment_name, run_name, parent_run_name)
        ) is None:
            return False
        if (len(parents) == 1) and (mask.sum() == 1):
            return True
        if (len(parents) == 1) and (mask.sum() > 1):
            if not ignore_existing:
                raise ValueError("Same run several time for single parent")
            else:
                logger.warning(
                    f"Repeated run name '{run_name}' for parent '{parent_run_name}'"
                )

        logger.warning(
            "Run %s has several parents in experiment: %s", run_name, experiment_name
        )
        return True

    if run_name is None:
        if parent_run_name is None:
            raise ValueError("Both run_name and parent_run_name are None")

        if (
            parents := get_parent_run_id(experiment_name, run_name, parent_run_name)
        ) is None:
            return False

        parent_names = (
            df[df.loc[:, "run_id"].isin(parents)].loc[:, "tags.mlflow.runName"].values
        )
        return parent_run_name in parent_names


def get_mlflow_expr(experiment_name: str) -> str:
    """
    Retrieve the ID of an existing MLflow experiment or
    create a new one if it doesn't exist.

    If it does, the function returns its ID. If not,
    it creates a new experiment with the provided name and returns its ID.

    Args:
        experiment_name (str): Name of the MLflow experiment.

    Returns:
        str: ID of the existing or newly created MLflow experiment.

    """

    if experiment := mlflow.get_experiment_by_name(experiment_name):
        logger.info("Experiment '%s' already exists; using that.", experiment_name)
        return str(experiment.experiment_id)

    logger.info("Experiment '%s' does not exist --> create.", experiment_name)
    return mlflow.create_experiment(experiment_name)


def require_experiment_id(experiment_name: str) -> str:
    """Return experiment id for name; raise if missing."""

    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"Unknown MLflow experiment: {experiment_name}")
    return str(experiment.experiment_id)


def _escape_filter_value(s: str) -> str:
    # MLflow filter strings use single quotes.
    return s.replace("'", "\\'")


def find_parent_run_id(
    experiment_id: str,
    parent_run_name: str,
    client: MlflowClient | None = None,
) -> str | None:
    """Find the run_id of a parent run by run_name.

    Parent run is defined as having tag.mlflow.runName == parent_run_name and
    no tag.mlflow.parentRunId.
    """

    if client is None:
        client = MlflowClient()

    runs = client.search_runs(
        [experiment_id],
        filter_string=f"tag.mlflow.runName = '{_escape_filter_value(parent_run_name)}'",
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=10_000,
    )
    parent_runs = [r for r in runs if r.data.tags.get("mlflow.parentRunId") is None]
    if len(parent_runs) == 0:
        return None
    if len(parent_runs) > 1:
        ids = [r.info.run_id for r in parent_runs]
        raise ValueError(
            f"Several parent runs found for run_name='{parent_run_name}': {ids}"
        )
    return parent_runs[0].info.run_id


def list_child_runs(
    experiment_id: str,
    parent_run_id: str,
    client: MlflowClient | None = None,
):
    """List MLflow runs that have the given parent_run_id."""

    if client is None:
        client = MlflowClient()

    return client.search_runs(
        [experiment_id],
        filter_string=f"tag.mlflow.parentRunId = '{_escape_filter_value(parent_run_id)}'",
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=10_000,
    )


def parse_child_idx(
    run,
    *,
    idx_tag: str = "sisyphus.child_idx",
    run_name_re: str = r"(\d{3})",
) -> int | None:
    """Parse a numeric child index from a run.

    Order of preference:
      1) tag idx_tag
      2) run_name matching run_name_re (default: exactly 3 digits)
    """

    tag_val = run.data.tags.get(idx_tag)
    if tag_val is not None:
        try:
            return int(tag_val)
        except Exception:
            pass

    rn = getattr(run.info, "run_name", None)
    if isinstance(rn, str):
        m = re.fullmatch(run_name_re, rn.strip())
        if m:
            return int(m.group(1))
    return None


def next_child_idx(
    child_runs,
    *,
    idx_tag: str = "sisyphus.child_idx",
    run_name_re: str = r"(\d{3})",
) -> int:
    idxs = [
        i
        for r in child_runs
        if (i := parse_child_idx(r, idx_tag=idx_tag, run_name_re=run_name_re))
        is not None
    ]
    return (max(idxs) + 1) if idxs else 1
