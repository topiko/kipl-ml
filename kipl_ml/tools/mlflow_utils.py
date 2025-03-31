import os

import dotenv
import mlflow
import numpy as np
import pandas as pd
from mlflow.entities import ViewType
from omegaconf import OmegaConf

from kipl_ml.data.wf_dataset import WFDataset
from kipl_ml.logging.logger import get_logger

dotenv.load_dotenv()
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

logger = get_logger(__name__)


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

    if predictions is not None:
        df.loc[:, "predictions"] = predictions

    ds_ = mlflow.data.from_pandas(
        df,
        name=ds.name,
        targets=target,
        predictions="predictions",
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
            raise ValueError("Same run several time for single parent")

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
