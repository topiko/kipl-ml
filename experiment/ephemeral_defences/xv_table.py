import argparse

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

    groupby = [
        "params.dataset_name",
        "params.model_name",
        "params.defence.defence-type",
        "params.network_state",
    ]

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
            index="params.defence.defence-type",
            columns=[
                "params.dataset_name",
                "params.network_state",
                "params.model_name",
            ],
            values=0,
        )
        .rename_axis(index="", columns=["", "", ""])
        .sort_index(axis=1)
    )

    print(res)

    with open(f"tables/{args.experiment_name}_table.txt", "w", encoding="utf-8") as f:
        for line in str(res).split("\n"):
            f.write(line + "\n")

    res.to_latex(f"tables/{args.experiment_name}_table.tex")


if __name__ == "__main__":
    main()
