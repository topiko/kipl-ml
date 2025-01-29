import os

import dotenv
import mlflow
import pandas as pd
from kipl_ml.data.wf_dataset import WFDataset
from kipl_ml.logging.logger import get_logger
from mlflow.entities import ViewType
from omegaconf import OmegaConf

dotenv.load_dotenv()
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

logger = get_logger(__name__)


def list_runs(
    experiment_names: list[str] | str | None = None, only_finished: bool = True
) -> pd.DataFrame:

    if experiment_names is None:
        experiments = mlflow.search_experiments(view_type=ViewType.ALL)
        experiment_names = [ex.name for ex in experiments]
    elif isinstance(experiment_names, str):
        experiment_names = [experiment_names]

    logger.info(f"Searching for runs in experiments: {experiment_names}")
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

    return runs.reset_index(drop=True)


def log_dataset(ds: WFDataset, store_cols: list[str]):

    ds_ = mlflow.data.from_pandas(
        ds.meta_df.loc[:, store_cols],
        name=ds.name,
        targets="label",
    )
    mlflow.log_input(dataset=ds_, context=f"{ds.name}_df.json")


def log_hydra_conf(cfg: OmegaConf):

    d = OmegaConf.to_container(cfg, resolve=True)
    mlflow.log_dict(d, artifact_file="hydra_config.json")


def load_hydra_conf(run_id: str) -> OmegaConf:
    run = mlflow.get_run(run_id)

    artifact_uri = run.info.artifact_uri

    d = mlflow.artifacts.load_dict(artifact_uri + "/hydra_config.json")

    return OmegaConf.create(d)


def hydra_run_exists(experiment_name: str, cfg: OmegaConf) -> str | None:
    runs = list_runs(experiment_name, only_finished=True)

    logger.warning("run exists fun is broken")

    return None
    # for i, row in runs.iterrows():
    #     d = load_hydra_conf(row["run_id"])
    #     if OmegaConf.to_container(cfg) == d:
    #         return row["run_id"]

    # return None
