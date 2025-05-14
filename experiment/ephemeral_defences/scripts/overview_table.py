import argparse

import pandas as pd

from experiment.ephemeral_defences.scripts.name_maps import DEFENCE_NAME_MAP
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
        )
        unit = "min"
    else:
        print("Available cols")
        for c in df.columns:
            print(c)
        raise NotImplementedError(f"{metric}")

    dataset = df.loc[:, "params.dataset_name"].unique()
    if len(dataset) != 1:
        raise ValueError(f"Multiple datasets detected: {dataset}")

    dataset = dataset[0]

    infty = ""
    bottleneck = "\\bottleneck"
    res = (
        df.groupby(groupby)
        .apply(
            lambda x: rf"${x.loc[:, metric].mean():.1f}^{{\pm {x.loc[:, metric].std():.1f}}}$",
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
    df = df[~df.loc[:, "params.test_xv"].isnull()]

    res_acc = _parse_df(df, "test_accuracy")
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
