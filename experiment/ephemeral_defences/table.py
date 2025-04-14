import argparse
import unicodedata

import pandas as pd

from kipl_ml.tools.mlflow_utils import list_runs


def _parse_df(df: pd.DataFrame, metric: str) -> pd.DataFrame:

    def_type = "params.defence.defence-type"
    mbnt_mask = df.loc[:, def_type].isin(["maybenot", "ephemeral"])
    df.loc[mbnt_mask, def_type] = df.loc[mbnt_mask, :].apply(
        lambda x: f"{x.loc[def_type]} | sc.={x.loc['params.defence.scale']} | {x.loc['params.defence.deck']}",
        axis=1,
    )

    groupby = [
        "params.dataset_name",
        "params.model_name",
        "params.defence.defence-type",
        "params.network_state",
    ]

    metric = f"metrics.{metric}"
    df.loc[:, metric] *= 100

    infty = unicodedata.lookup("Rocket")
    bottleneck = unicodedata.lookup("Hourglass")
    res = (
        df.groupby(groupby)
        .apply(
            lambda x: f"{x.loc[:, metric].mean():.1f}\u00b1{x.loc[:, metric].std():.1f}",
            include_groups=False,
        )
        .to_frame()
        .reset_index()
        .pivot(
            index=["params.defence.defence-type", "params.network_state"],
            columns=["params.dataset_name", "params.model_name"],
            values=0,
        )
        .rename_axis(index=["", ""], columns=["", ""])
        .sort_index(axis=1)
        .rename(
            index={"infinite": infty, "bottleneck": bottleneck},
            level=1,
        )
    )

    if metric.startswith("metrics.def."):
        res = res.loc[:, (slice(None), "df")].rename(
            columns={"df": metric.replace("metrics.def.", "")}, level=1
        )
    elif metric.startswith("metrics.sim."):
        res = res.loc[:, (slice(None), "df")].rename(
            columns={"df": metric.replace("metrics.", "")}, level=1
        )
    elif metric == "metrics.test_accuracy":
        res = res.rename(columns=lambda x: f"acc-{x} %", level=1)

    return res


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
    df = df[~df.loc[:, "params.test_xv"].isnull()]

    res_acc = _parse_df(df, "test_accuracy")
    res_bw = _parse_df(df, "def.bandwidth")
    res_delay = _parse_df(df, "def.delay")
    res_missing = _parse_df(df, "sim.missing")

    res = pd.concat((res_acc, res_bw, res_delay, res_missing), axis=1)

    print(res)
    with open(f"tables/{args.experiment_name}_table.txt", "w", encoding="utf-8") as f:
        for line in str(res).split("\n"):
            f.write(line + "\n")

    res.to_latex(f"tables/{args.experiment_name}_table.tex")


if __name__ == "__main__":
    main()
