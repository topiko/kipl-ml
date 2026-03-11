"""Plot duration quantiles at packet-count checkpoints.

Example checkpoints: 1000, 2000, 3000 packets.
For each checkpoint k, we measure elapsed time until packet k for all traces
that have at least k packets (after trimming) and plot quantile summaries.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from kipl_ml.data.utils import Datasets, assets, get_std_trace_dict, load_dataset_meta_df
from kipl_ml.trace.enums import Feats


def _subset_meta(meta: pd.DataFrame, max_traces: int, seed: int) -> pd.DataFrame:
    if max_traces <= 0 or max_traces >= len(meta):
        return meta
    return meta.sample(n=max_traces, random_state=seed).reset_index(drop=True)


def _parse_limits(limits_s: str, step: int, n_packets: int) -> list[int]:
    if limits_s.strip():
        vals = [int(x.strip()) for x in limits_s.split(",") if x.strip()]
    else:
        vals = list(range(int(step), int(n_packets) + 1, int(step)))

    vals = sorted(set(v for v in vals if v > 0))
    if not vals:
        raise ValueError("No valid packet checkpoints provided")
    return vals


def _collect_elapsed_by_limit(
    meta: pd.DataFrame,
    *,
    limits: list[int],
    n_packets: int,
    trim_beginning: int,
) -> dict[int, list[float]]:
    out = {k: [] for k in limits}

    for _, row in tqdm(meta.iterrows(), total=len(meta), desc="Collecting quantiles"):
        path = Path(row[assets.TRACE_F_PATH])
        trace = get_std_trace_dict(path)

        t = trace[Feats.TIMES]
        d = trace[Feats.DIRS]
        m = d != 0
        t = t[m]

        if trim_beginning > 0:
            t = t[trim_beginning:]
        if n_packets > 0:
            t = t[:n_packets]

        n = int(t.numel())
        if n == 0:
            continue

        elapsed = (t - t[0]).detach().cpu().numpy().astype(np.float64)
        for k in limits:
            if n >= k:
                out[k].append(float(elapsed[k - 1]))

    return out


def _summary_df(values: dict[int, list[float]]) -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []
    for k in sorted(values):
        arr = np.asarray(values[k], dtype=np.float64)
        if arr.size == 0:
            rows.append(
                {
                    "packet_limit": int(k),
                    "n_traces": 0,
                    "q00_s": np.nan,
                    "q05_s": np.nan,
                    "q25_s": np.nan,
                    "q50_s": np.nan,
                    "q75_s": np.nan,
                    "q95_s": np.nan,
                    "q100_s": np.nan,
                    "mean_s": np.nan,
                    "std_s": np.nan,
                }
            )
            continue

        q00, q05, q25, q50, q75, q95, q100 = np.quantile(
            arr, [0.00, 0.05, 0.25, 0.50, 0.75, 0.95, 1.00]
        )
        rows.append(
            {
                "packet_limit": int(k),
                "n_traces": int(arr.size),
                "q00_s": float(q00),
                "q05_s": float(q05),
                "q25_s": float(q25),
                "q50_s": float(q50),
                "q75_s": float(q75),
                "q95_s": float(q95),
                "q100_s": float(q100),
                "mean_s": float(arr.mean()),
                "std_s": float(arr.std()),
            }
        )

    return pd.DataFrame(rows).sort_values("packet_limit").reset_index(drop=True)


def _plot_quantiles(
    summary: pd.DataFrame,
    values: dict[int, list[float]],
    *,
    out: Path,
    title: str,
    dpi: int,
) -> None:
    df = summary[summary["n_traces"] > 0].copy()
    if df.empty:
        raise RuntimeError("No traces contributed to any packet checkpoint")

    x = df["packet_limit"].to_numpy(dtype=np.float64)
    q00 = df["q00_s"].to_numpy(dtype=np.float64)
    q05 = df["q05_s"].to_numpy(dtype=np.float64)
    q25 = df["q25_s"].to_numpy(dtype=np.float64)
    q50 = df["q50_s"].to_numpy(dtype=np.float64)
    q75 = df["q75_s"].to_numpy(dtype=np.float64)
    q95 = df["q95_s"].to_numpy(dtype=np.float64)
    q100 = df["q100_s"].to_numpy(dtype=np.float64)

    data = [np.asarray(values[int(k)], dtype=np.float64) for k in x.astype(int)]

    fig, (ax0, ax1) = plt.subplots(
        2,
        1,
        figsize=(14, 9),
        sharex=True,
        gridspec_kw={"height_ratios": [2, 3]},
    )

    ax0.fill_between(x, q00, q100, color="#cbd5e1", alpha=0.25, label="q00-q100")
    ax0.fill_between(x, q05, q95, color="#a5b4fc", alpha=0.30, label="q05-q95")
    ax0.fill_between(x, q25, q75, color="#6366f1", alpha=0.35, label="q25-q75")
    ax0.plot(x, q00, color="#334155", lw=1.0, ls="--", alpha=0.8, label="q00")
    ax0.plot(x, q100, color="#334155", lw=1.0, ls=":", alpha=0.8, label="q100")
    ax0.plot(x, q50, color="#1e1b4b", lw=2.0, marker="o", ms=4, label="q50")
    ax0.set_ylabel("Duration [s]")
    ax0.grid(alpha=0.2)
    ax0.legend(frameon=False, loc="upper left")

    if len(x) >= 2:
        min_dx = np.min(np.diff(x))
        widths = max(10.0, 0.55 * min_dx)
    else:
        widths = max(10.0, 0.25 * float(x[0]))

    bp = ax1.boxplot(
        data,
        positions=x,
        widths=widths,
        showfliers=False,
        whis=(0, 100),
        patch_artist=True,
    )
    for box in bp["boxes"]:
        box.set(facecolor="#93c5fd", alpha=0.45, edgecolor="#1d4ed8")
    for med in bp["medians"]:
        med.set(color="#1e3a8a", linewidth=2)
    for whisk in bp["whiskers"]:
        whisk.set(color="#1d4ed8", linewidth=1)
    for cap in bp["caps"]:
        cap.set(color="#1d4ed8", linewidth=1)

    ax1.set_ylabel("Duration [s]")
    ax1.set_xlabel("Packet checkpoint")
    ax1.grid(alpha=0.2)

    axn = ax1.twinx()
    axn.plot(x, df["n_traces"].to_numpy(), color="#475569", lw=1.2, alpha=0.9)
    axn.set_ylabel("Trace count", color="#475569")
    axn.tick_params(axis="y", colors="#475569")

    fig.suptitle(title)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=int(dpi))
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=str(Datasets.BIGENOUGH))
    ap.add_argument("--n_packets", type=int, default=5000)
    ap.add_argument("--trim_beginning", type=int, default=10)
    ap.add_argument("--limits", default="", help="Comma-separated packet checkpoints")
    ap.add_argument("--step", type=int, default=1000)
    ap.add_argument("--max_traces", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--out",
        default="experiment/obsrl/codex_debug/trace_duration_quantiles.png",
    )
    ap.add_argument(
        "--summary_csv",
        default="experiment/obsrl/codex_debug/trace_duration_quantiles_summary.csv",
    )
    ap.add_argument("--dpi", type=int, default=280)
    args = ap.parse_args()

    limits = _parse_limits(args.limits, step=int(args.step), n_packets=int(args.n_packets))

    meta = load_dataset_meta_df(args.dataset, include_xv_cols=False)
    meta = _subset_meta(meta, max_traces=int(args.max_traces), seed=int(args.seed))

    values = _collect_elapsed_by_limit(
        meta,
        limits=limits,
        n_packets=int(args.n_packets),
        trim_beginning=int(args.trim_beginning),
    )

    summary = _summary_df(values)
    summary_csv = Path(args.summary_csv)
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_csv, index=False)

    out = Path(args.out)
    title = (
        f"Duration quantiles by packet checkpoint ({len(meta)} traces, trim={args.trim_beginning})"
    )
    _plot_quantiles(summary, values, out=out, title=title, dpi=int(args.dpi))

    print(f"OK: saved quantile plot -> {out}")
    print(f"OK: saved quantile summary -> {summary_csv}")
    print("Checkpoints:", limits)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
