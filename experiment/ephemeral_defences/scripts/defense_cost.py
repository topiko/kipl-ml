import argparse

import matplotlib.pyplot as plt
import seaborn as sns

from kipl_ml.tools.mlflow_utils import list_runs

FLAVOR_COL = "def-flavor"
COST = "bw+delay"
METRIC = "accuracy"
COLS = [
    FLAVOR_COL,
    "params.defence.scale",
    "metrics.test_accuracy",
    "metrics.def.bandwidth",
    "metrics.def.delay",
    "metrics.sim.missing",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-en",
        "--experiment-name",
        type=str,
        nargs="+",
        required=True,
        help="Which experiment(s) to consider",
    )
    parser.add_argument("--missing", action="store_true")
    parser.add_argument("--timings", action="store_true")

    args = parser.parse_args()

    df = list_runs(args.experiment_name)
    df = df[~df.loc[:, "params.test_xv"].isnull()]

    if "params.defence.flavor" not in df.columns:
        df[FLAVOR_COL] = None

    df = df.rename(
        columns={
            "params.defence.flavor": FLAVOR_COL,
            "metrics.test_accuracy": "accuracy",
            "metrics.def.bandwidth": "bw",
            "metrics.def.delay": "delay",
            "params.defence.scale": "def-scale",
        }
    )

    df.loc[:, COST] = df.loc[:, "delay"] + df.loc[:, "bw"]
    df.loc[df.loc[:, FLAVOR_COL].isnull(), FLAVOR_COL] = "default"

    sns.scatterplot(df, x=COST, y=METRIC, hue=FLAVOR_COL)
    # plt.semilogx()
    plt.show()


if __name__ == "__main__":
    main()
