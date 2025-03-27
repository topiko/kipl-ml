import argparse

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from kipl_ml.tools.mlflow_utils import list_runs

pd.set_option("future.no_silent_downcasting", True)


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
    parser.add_argument(
        "--defence-type",
        type=str,
        nargs="+",
        default=["front", "maybenot", "interspace"],
    )

    args = parser.parse_args()

    df = list_runs(args.experiment_name)

    metric = "accuracy"
    df = df[~df.loc[:, "params.test_xv"].isnull()]
    df = df.infer_objects()

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
    df = df[df.loc[:, "def-type"].isin(args.defence_type)]

    df.loc[:, "def-aug"] = df.loc[:, "def-aug"].astype(int)
    augs = df.loc[:, "def-aug"].unique()
    max_aug = max(augs)
    df.loc[:, "def-aug"] = df.loc[:, "def-aug"].replace(0, 2 * max_aug)
    augs = sorted(df.loc[:, "def-aug"].unique())

    df.loc[:, "fixed-per-trace"] = df.loc[:, "fixed-per-trace"].apply(
        lambda x: x == "True"
    )

    fgrid = sns.relplot(
        df,
        x="def-aug",
        y=metric,
        hue="model",
        col="def-type",
        row="fixed-per-trace",
        kind="line",
        height=3,
        aspect=0.8,
        legend="brief",
        facet_kws={"margin_titles": True, "sharey": True, "sharex": True},
    )

    fgrid.set_titles(col_template="{col_name}", row_template="{row_var}={row_name}")
    fgrid.set_ylabels(metric, clear_inner=False)
    sns.move_legend(fgrid, "upper left", bbox_to_anchor=(0.25, 0.55))

    for ax in fgrid.axes.flatten():
        ax.set_xscale("log", base=2)
        ax.set_yscale("log", base=10)
        ticks = augs
        ax.set_xticks(ticks)
        labels = [str(t) for t in ticks]
        labels[-1] = "∞"
        ax.set_xticklabels(labels)

    plt.tight_layout()
    plt.savefig("figs/aug_curve.png")
    plt.show()


if __name__ == "__main__":
    main()
