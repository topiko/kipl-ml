import os

import dotenv
import mlflow
import pandas as pd
from kipl_ml.data.wf_dataset import WFDataset
from kipl_ml.logging.logger import get_logger
from kipl_ml.trace.features import FeatureTrs
from mlflow.entities import ViewType

dotenv.load_dotenv()
logger = get_logger(__name__)

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
assert MLFLOW_TRACKING_URI is not None, "MLFLOW_TRACKING_URI must be set in .env file."
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

SHOW = [
    "run_id",
    "start_time",
    "tags.mlflow.runName",
    "metrics.test_accuracy",
    "experiment_id",
]


def list_runs(experiment_names: list[str] | None = None) -> pd.DataFrame:
    experiment_names = experiment_names or ["laserbeak", "laserbeak_tune"]
    runs = mlflow.search_runs(
        search_all_experiments=True,
        run_view_type=ViewType.ALL,
        experiment_names=experiment_names,
        output_format="pandas",
    )

    assert isinstance(runs, pd.DataFrame)
    mask = runs.loc[:, "status"] == "FINISHED"
    runs = runs.loc[mask, SHOW].reset_index(drop=True)

    return runs


def main():
    runs = list_runs()

    print(runs)

    idx = int(input("Your choice (idx): "))

    run_id = runs.loc[idx, "run_id"]
    experiment_id = runs.loc[idx, "experiment_id"]

    # It is imperative to set the experiment_id correctly:
    mlflow.set_experiment(experiment_id=experiment_id)

    run = mlflow.get_run(run_id)

    features = mlflow.load_table(artifact_file="features.json", run_ids=[run_id])
    feature_names = features.features.tolist()
    n_packets = int(run.data.params["n_packets"])

    feature_trs = FeatureTrs(feature_names=feature_names, n_packets=n_packets)

    dataset_name = run.data.params["dataset_name"]
    artifact_uri = run.info.artifact_uri

    test_meta_df = mlflow.load_table(artifact_file="test_df.json", run_ids=[run_id])

    dataset = WFDataset(
        dataset=dataset_name, meta_df=test_meta_df, feature_trs=feature_trs
    )
    model = mlflow.pyfunc.load_model(artifact_uri + "/model")


if __name__ == "__main__":
    main()
