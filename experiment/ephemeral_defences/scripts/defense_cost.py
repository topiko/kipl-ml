import argparse

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

from kipl_ml.tools.mlflow_utils import list_runs

FLAVOR_COL = "def-flavor"
COST = "Cost (bandwidth+delay)"
METRIC = "Accuracy"
COLS = [
    FLAVOR_COL,
    "params.defence.scale",
    "metrics.test_accuracy",
    "metrics.def.bandwidth_median",
    "metrics.def.delay_median",
    "metrics.sim.missing",
]

fs = 6
mpl.rcParams["xtick.labelsize"] = fs
mpl.rcParams["ytick.labelsize"] = fs
mpl.rcParams["axes.labelsize"] = fs
mpl.rcParams["legend.fontsize"] = fs
mpl.rcParams["figure.figsize"] = (3.2, 2.3)
mpl.rcParams["axes.spines.right"] = False
mpl.rcParams["axes.spines.top"] = False
mpl.rcParams["axes.spines.left"] = False
mpl.rcParams["axes.spines.bottom"] = False
mpl.rcParams["axes.grid"] = True
mpl.rcParams["grid.color"] = "white"
mpl.rcParams["axes.facecolor"] = "whitesmoke"
mpl.rcParams["axes.linewidth"] = 0.5
mpl.rcParams["lines.linewidth"] = 0.5
mpl.rcParams["lines.markersize"] = 2
mpl.rcParams["errorbar.capsize"] = 1


def _get_undefended(attack: str, undef_expr: str, xvs: list[int]) -> pd.DataFrame:

    df = list_runs(undef_expr)

    df = df[df.loc[:, "params.model_name"] == attack]
    df = df[df.loc[:, "params.defence.defence-type"] == "nodefence"]
    df.loc[:, "params.test_xv"] = df.loc[:, "params.test_xv"].astype(int)
    df = df[df.loc[:, "params.test_xv"].isin(xvs)]

    if len(df) != len(xvs):
        raise ValueError(f"Expected {len(xvs)} undefended runs, but got {len(df)}.")

    df.loc[:, "params.defence.scale"] = 0
    df.loc[:, "params.defence.flavor"] = "undefended"

    return df


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
        "-en-undef",
        "--undef-experiment-name",
        type=str,
        nargs="+",
        required=True,
        help="Where to get undef experiment to consider",
    )

    parser.add_argument(
        "--xv",
        type=int,
        nargs="+",
        default=[0],
        help="Cross-validation folds to consider",
    )
    parser.add_argument("--missing", action="store_true")
    parser.add_argument("--model", type=str, default="laserbeak_wo_attention")

    args = parser.parse_args()

    df = list_runs(args.experiment_name)
    df = df[~df.loc[:, "params.test_xv"].isnull()]
    df = df[df.loc[:, "params.model_name"] == args.model]

    attack = args.model

    df = pd.concat(
        (df, _get_undefended(attack, args.undef_experiment_name, args.xv)), axis=0
    )

    if "params.defence.flavor" not in df.columns:
        df[FLAVOR_COL] = None

    df = df.rename(
        columns={
            "params.defence.flavor": FLAVOR_COL,
            "metrics.test_accuracy": "Accuracy",
            "metrics.def.bandwidth_median": "bw",
            "metrics.def.delay_median": "delay",
            "params.defence.scale": "def-scale",
            "params.test_xv": "xv",
        }
    )

    df.loc[:, "xv"] = df.loc[:, "xv"].astype(int)
    df = df[df.loc[:, "xv"].isin(args.xv)]

    df.loc[:, COST] = df.loc[:, "delay"] + df.loc[:, "bw"]
    df.loc[df.loc[:, FLAVOR_COL].isnull(), FLAVOR_COL] = "default"

    df.loc[
        :,
        [
            FLAVOR_COL,
            METRIC,
            "bw",
            "delay",
            "xv",
            "def-scale",
            "params.model_name",
            "params.defence.defence-type",
        ],
    ].to_csv("tables/cost_curve_raw.csv", index=False)

    df_ = (
        df.groupby([FLAVOR_COL, "def-scale"])
        .agg(
            {
                "Accuracy": ["mean", "std"],
                COST: ["mean", "std"],
            }
        )
        .reset_index()
    )

    df_.columns = [f"{c[0]}-{c[1]}" if c[1] != "" else c[0] for c in df_.columns]

    df_.to_csv("tables/cost_curve_formatted.csv", index=False)

    _, ax = plt.subplots()

    for i, flavor in enumerate(sorted(df.loc[:, FLAVOR_COL].unique())):

        c = f"C{i}"
        fmt = ["-o", "-s", "-^", "-*"][i]

        mask2 = df_.loc[:, FLAVOR_COL] == flavor

        # mask1 = df.loc[:, FLAVOR_COL] == flavor
        # ax.scatter(x=df.loc[mask1, COST], y=df.loc[mask1, METRIC], c=c)

        df_tmp = df_.loc[mask2, :].copy()

        ax.errorbar(
            df_tmp.loc[:, f"{COST}-mean"].values,
            df_tmp.loc[:, f"{METRIC}-mean"].values,
            xerr=df_tmp.loc[:, f"{COST}-std"].values,
            yerr=df_tmp.loc[:, f"{METRIC}-std"].values,
            fmt=fmt,
            color=c,
            label=flavor,
        )

    plt.xlabel(COST)
    plt.ylabel(f"{METRIC} | {attack}")
    plt.legend(loc=3)
    plt.tight_layout()
    plt.savefig("figs/cost_curve.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    main()
