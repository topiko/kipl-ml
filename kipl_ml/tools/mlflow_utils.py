import os

import dotenv
import mlflow
import pandas as pd
from kipl_ml.logging.logger import get_logger
from mlflow.entities import ViewType

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
