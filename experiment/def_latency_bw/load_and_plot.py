import os

import dotenv
import matplotlib.pyplot as plt
import mlflow
import pandas as pd
import seaborn as sns
import yaml
from kipl_ml.logging.logger import get_logger
from kipl_ml.tools.mlflow_utils import list_runs

dotenv.load_dotenv()
logger = get_logger(__name__)

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
assert MLFLOW_TRACKING_URI is not None, "MLFLOW_TRACKING_URI must be set in .env file."
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

SHOW = [
    "start_time",
    "tags.mlflow.runName",
    "run_id",
    "params.def_padding_fraction",
    "params.def_chi2_df",
    "metrics.final:test_accuracy",
]

with open("config/config.yaml", "r") as f:
    PARAMS = yaml.load(f, Loader=yaml.FullLoader)
    EXPERIMENT_NAME = PARAMS["mlflow"]["experiment_name"]


def heatmap(res_df: pd.DataFrame) -> pd.DataFrame:
    res_df = res_df.pivot(
        index="params.def_padding_fraction",
        columns="params.def_chi2_df",
        values="metrics.final:test_accuracy",
    ).iloc[::-1]


    sns.heatmap(res_df, annot=True, fmt=".2f")
    plt.show()


def main():

    runs_df = list_runs(experiment_names=EXPERIMENT_NAME, only_finished=True)

    data = runs_df.loc[
        :,
        [
            "metrics.final:test_accuracy",
            "params.def_padding_fraction",
            "params.def_chi2_df",
        ],
    ]

    heatmap(data)

if __name__ == "__main__":
    main()
