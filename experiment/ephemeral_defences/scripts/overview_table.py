import argparse

import pandas as pd

from experiment.ephemeral_defences.scripts.name_maps import DEFENCE_NAME_MAP
from kipl_ml.logging.logger import get_logger
from kipl_ml.tools.mlflow_utils import list_runs

logger = get_logger(__name__)


def _parse_df(df: pd.DataFrame, metric: str, make_bold: bool = False) -> pd.DataFrame:

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

    if metric in {"sim.missing", "def.bandwidth", "def.delay", "test_accuracy"}:
        metric = f"metrics.{metric}"
        df.loc[:, metric] *= 100
        unit = "\\%"
    elif metric == "timings":
        df.loc[:, metric] = (
            df.loc[:, ["end_time", "start_time"]].apply(
                lambda x: (x.end_time - x.start_time).total_seconds(), axis=1
            )
            / 60
            / 60
        )
        unit = "h"
    else:
        print("Available cols")
        for c in df.columns:
            print(c)
        raise NotImplementedError(f"{metric}")

    dataset = df.loc[:, "params.dataset_name"].unique()
    if len(dataset) != 1:
        raise ValueError(f"Multiple datasets detected: {dataset}")

    dataset = dataset[0]

    if len(
        df.drop_duplicates(
            [col for col in df.columns if col.split(".")[0] in {"params", "mlflow"}]
        )
    ) != len(df):
        raise ValueError("Duplicates in df!")

    infty = ""
    bottleneck = "\\bottleneck"

    df.to_csv("tables/overview_table_raw.csv", index=False)

    res = df.groupby(groupby).agg({metric: ["mean", "std"]}).reset_index()
    res.columns = [c[0] if c[1] == "" else f"{c[0]}-{c[1]}" for c in res.columns]

    def _pivot_table(df, values) -> pd.DataFrame:
        return (
            df.pivot(
                index=["params.defence.defence-type", "params.network_state"],
                columns=["params.dataset_name", "params.model_name"],
                values=values,
            )
            .rename_axis(index=["", ""], columns=["", ""])
            .sort_index(axis=1)
            .rename(
                index={"infinite": infty, "bottleneck": bottleneck},
                level=1,
            )
        )

    def _to_str(means: pd.DataFrame, stds: pd.DataFrame) -> pd.DataFrame:
        df = pd.DataFrame(index=means.index, columns=means.columns)
        for row_idx, row_vals in means.iterrows():
            max_idx = row_vals.idxmax()
            for col_idx in row_vals.index:
                m_val = max(0, means.loc[row_idx, col_idx])
                m = f"{m_val:.1f}"
                s = f"{stds.loc[row_idx, col_idx]:.1f}"

                if (col_idx == max_idx) and make_bold:
                    str_ = rf"$\mathbf{{{m}^{{\pm {s}}}}}$"
                else:
                    str_ = rf"${m}^{{\pm {s}}}$"
                df.loc[row_idx, col_idx] = str_

        return df

    means = _pivot_table(res, f"{metric}-mean")

    if metric == "timings":
        print(f"Total runtime: {df.loc[:, metric].sum()} {unit}")

    stds = _pivot_table(res, values=f"{metric}-std")

    res = _to_str(means, stds)

    idx = res.index
    res.index = [DEFENCE_NAME_MAP[id_[0]] + id_[1] for id_ in idx]
    res.index.name = dataset

    if metric.startswith("metrics.def."):
        res = (
            res.loc[:, (slice(None), "df")]
            .rename(columns={"df": f"{metric.split(".")[-1]}"}, level=1)
            .rename(columns={dataset: f"overhead {unit}"}, level=0)
        )
    elif metric.startswith("metrics.sim."):
        res = (
            res.loc[:, (slice(None), "df")]
            .rename(columns={"df": metric.split(".")[-1]}, level=1)
            .rename(columns={dataset: f"sim {unit}"}, level=0)
        )
    elif metric == "timings":
        res = res.rename(columns={dataset: f"timing {unit}"}, level=0)
    elif metric == "metrics.test_accuracy":
        res = res.rename(columns={dataset: f"accuracy {unit}"}, level=0)

    index = [
        "Undefended",
        "Break-Pad\\bottleneck",
        "Break-Pad",
        "Ephemeral padding-only | sc.=0.5\\bottleneck",
        "Ephemeral padding-only | sc.=0.75",
        "FRONT\\bottleneck",
        "FRONT",
        "Interspace\\bottleneck",
        "Interspace",
        "Ephemeral blocking | sc.=0.75\\bottleneck",
        "Ephemeral blocking | sc.=0.75",
        "RegulaTor\\bottleneck",
        "RegulaTor",
        "Tamaraw\\bottleneck",
        "Tamaraw",
    ]

    index = [idx for idx in index if idx in res.index]
    res = res.loc[index]
    res.index = [
        idx.replace(" | sc.=0.5", "")
        .replace(" | sc.=0.75", "")
        .replace("padding-only", "Pad")
        .replace("blocking", "Block")
        for idx in res.index
    ]

    res.to_csv("tables/overview_table_formatted.csv", index=True)

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
    parser.add_argument("--missing", action="store_true")
    parser.add_argument("--timings", action="store_true")

    args = parser.parse_args()

    df = list_runs(args.experiment_name)
    # Remove the parent runs.
    df = df[~df.loc[:, "params.test_xv"].isnull()]

    mask = df.loc[:, ["params.model_name", "params.test_xv"]].apply(
        lambda x: (
            x.loc["params.model_name"] in {"laserbeak", "laserbeak_wo_attention"}
        )
        and (int(x.loc["params.test_xv"]) >= 2),
        axis=1,
    )

    if sum(mask) > 0:
        logger.warning("Removing lb runs over xv 2.")
        df = df[~mask]

    res_acc = _parse_df(df, "test_accuracy", make_bold=True)
    res_bw = _parse_df(df, "def.bandwidth")
    res_delay = _parse_df(df, "def.delay")

    res = pd.concat((res_acc, res_bw, res_delay), axis=1)

    if args.missing:
        res_missing = _parse_df(df, "sim.missing")
        res = pd.concat((res, res_missing), axis=1)
    if args.timings:
        res_timings = _parse_df(df, "timings")
        res = res_timings

    print(res)
    with open(f"tables/{args.experiment_name}_table.txt", "w", encoding="utf-8") as f:
        for line in str(res).split("\n"):
            f.write(line + "\n")

    res.to_latex("tables/table.tex")


if __name__ == "__main__":
    main()
