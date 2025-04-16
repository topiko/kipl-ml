import argparse

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

    infty = ""  # "\\infinity"  # unicodedata.lookup("Rocket")
    bottleneck = "\\bottleneck"  # unicodedata.lookup("Hourglass")
    res = (
        df.groupby(groupby)
        .apply(
            lambda x: f"{x.loc[:, metric].mean():.1f}\u00b1{x.loc[:, metric].std():.1f}\\%",
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

    defence_name_map = {
        "breakpad": "Break-Pad",
        "front": "FRONT",
        "interspace": "Interspace",
        "regulator": "RegulaTor",
        "maybenot | sc.=0.5 | eph-blocking-bottle-1k-c2-100k-april13-use-scale0.75": "Ephemeral blocking | sc.=0.5",
        "maybenot | sc.=0.75 | eph-blocking-bottle-1k-c2-100k-april13-use-scale0.75": "Ephemeral blocking | sc.=0.75",
        "maybenot | sc.=0.75 | eph-blocking-1k-c2-100k-april8-use-scale0.75": "Ephemeral blocking | sc.=0.75",
        "maybenot | sc.=0.5 | eph-padding-bottle-1k-c2-100k-april13-use-scale0.5": "Ephemeral padding-only | sc.=0.5",
        "maybenot | sc.=0.75 | eph-padding-1k-c2-100k-april8-use-scale0.75": "Ephemeral padding-only | sc.=0.75",
        "maybenot | sc.=0.5 | def-padding-only-bottleneck-100k-2025-03-03": "Ephemeral padding-only 3.3. 100k | sc.=0.5",
        "maybenot | sc.=0.5 | def-padding-only-infinite-100k-2025-03-03": "Ephemeral padding-only 3.3. 100k | sc.=0.5",
        "nodefence": "Undefended",
        "tamaraw": "Tamaraw",
    }

    # Apr  8 17:04 eph-blocking-1k-c2-100k-april8-use-scale0.75
    # Apr 13 23:14 eph-blocking-bottle-1k-c2-100k-april13-use-scale0.75
    # Apr  8 16:46 eph-padding-1k-c2-100k-april8-use-scale0.75
    # Apr 13 23:16 eph-padding-bottle-1k-c2-100k-april13-use-scale0.5

    res.index = [defence_name_map[id_[0]] + id_[1] for id_ in idx]

    if metric.startswith("metrics.def."):
        res = res.loc[:, (slice(None), "df")].rename(
            columns={"df": metric.replace("metrics.def.", "")}, level=1
        )
    elif metric.startswith("metrics.sim."):
        res = res.loc[:, (slice(None), "df")].rename(
            columns={"df": metric.replace("metrics.", "")}, level=1
        )
    elif metric == "metrics.test_accuracy":
        res = res.rename(columns=lambda x: f"acc-{x}", level=1)

    index = [
        "Break-Pad\\bottleneck",
        "Break-Pad",
        "FRONT\\bottleneck",
        "FRONT",
        "Interspace\\bottleneck",
        "Interspace",
        "Ephemeral padding-only | sc.=0.5\\bottleneck",
        "Ephemeral padding-only | sc.=0.75",
        "Undefended\\bottleneck",
        "Undefended",
        "Tamaraw\\bottleneck",
        "Tamaraw",
        "RegulaTor\\bottleneck",
        "RegulaTor",
        "Ephemeral blocking | sc.=0.75\\bottleneck",
        "Ephemeral blocking | sc.=0.5\\bottleneck",
        "Ephemeral blocking | sc.=0.75",
    ]

    index = [idx for idx in index if idx in res.index]
    res = res.loc[index]
    res.index = [idx.replace("|", "\\vert") for idx in res.index]

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

    print(res)
    with open(f"tables/{args.experiment_name}_table.txt", "w", encoding="utf-8") as f:
        for line in str(res).split("\n"):
            f.write(line + "\n")

    res.to_latex("tables/table.tex")


if __name__ == "__main__":
    main()
