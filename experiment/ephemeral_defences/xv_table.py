import argparse

from kipl_ml.tools.mlflow_utils import list_runs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-en",
        "--experiment-name",
        type=str,
        required=True,
        help="Which experiment to consider",
    )

    args = parser.parse_args()

    df = list_runs(args.experiment_name)

    groupby = ["params.dataset_name", "params.model_name", "params.defence-type"]

    metric = "metrics.test_accuracy"
    res = (
        df.groupby(groupby)
        .apply(
            lambda x: f"{x.loc[:, metric].mean():.3f} \u00b1 {x.loc[:, metric].std():.3f}",
            include_groups=False,
        )
        .to_frame()
        .reset_index()
        .pivot(
            index="params.defence-type",
            columns=["params.dataset_name", "params.model_name"],
            values=0,
        )
        .rename_axis(index="", columns=["", ""])
    )

    print(res)

    res.to_latex(f"{args.experiment_name}_table.tex")


if __name__ == "__main__":
    main()
