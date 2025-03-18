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

    metric = "metrics.test_accuracy"
    df = df[~df.loc[:, "params.test_xv"].isnull()]
    df = df.infer_objects()
    df.loc[:, "params.defence_augmentation"] = df.loc[
        :, "params.defence_augmentation"
    ].astype(int)

    sns.relplot(
        df,
        x="params.defence_augmentation",
        y=metric,
        hue="params.model_name",
        row="params.defence.defence-type",
        kind="line",
        height=2,
        aspect=2,
    )
    plt.savefig("figs/aug_curve.png")
    plt.show()


if __name__ == "__main__":
    main()
