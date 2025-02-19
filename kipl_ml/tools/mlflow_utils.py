import os

import dotenv
import mlflow
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


def log_dataset(ds: WFDataset, store_cols: list[str], target: str):

    ds_ = mlflow.data.from_pandas(
        ds.meta_df.loc[:, store_cols],
        name=ds.name,
        targets=target,
    )
    mlflow.log_input(dataset=ds_, context=f"{ds.name}_df.json")


def log_hydra_conf(cfg: OmegaConf):

    d = OmegaConf.to_container(cfg, resolve=True)
    mlflow.log_dict(d, artifact_file="hydra_config.json")
