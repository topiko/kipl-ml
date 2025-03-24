import argparse

import matplotlib.pyplot as plt
import seaborn as sns

from kipl_ml.tools.mlflow_utils import list_runs


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

    args = parser.parse_args()

    df = list_runs(args.experiment_name)

    metric = "accuracy"
    df = df[~df.loc[:, "params.test_xv"].isnull()]
    df = df.infer_objects()
    df.loc[:, "params.defence_augmentation"] = df.loc[
        :, "params.defence_augmentation"
    ].astype(int)

    df.rename(
        columns={
            "params.defence_augmentation": "def-aug",
            "params.defence.defence-type": "def-type",
            "params.model_name": "model",
            "params.network_state": "netwk-state",
            "metrics.test_accuracy": "accuracy",
            "params.defence.fixed_per_trace": "fixed-per-trace",
        },
        inplace=True,
    )

    df.loc[:, "fixed-per-trace"] = df.loc[:, "fixed-per-trace"].apply(
        lambda x: x == "True"
    )

    fgrid = sns.relplot(
        df,
        x="def-aug",
        y=metric,
        hue="model",
        row="def-type",
        col="fixed-per-trace",
        kind="line",
        height=2,
        aspect=2,
        legend="brief",
        facet_kws={"margin_titles": True, "sharey": True, "sharex": True},
    )

    fgrid.set_titles(row_template="{row_name}", col_template="{col_var}={col_name}")
    fgrid.set_ylabels(metric, clear_inner=False)
    plt.savefig("figs/aug_curve.png")
    plt.show()


if __name__ == "__main__":
    main()
