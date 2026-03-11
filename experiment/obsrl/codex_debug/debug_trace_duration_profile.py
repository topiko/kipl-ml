"""Profile trace duration growth versus packet index.

This helps detect traces with unusually long silent periods that can slow down
delay-enabled NNDef rollouts despite moderate packet-count limits.
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


def _compute_profile(
    meta: pd.DataFrame,
    *,
    n_packets: int,
    trim_beginning: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, float | int | str]]]:
    sums = np.zeros((n_packets,), dtype=np.float64)
    sums2 = np.zeros((n_packets,), dtype=np.float64)
    counts = np.zeros((n_packets,), dtype=np.int64)

    per_trace: list[dict[str, float | int | str]] = []

    for _, row in tqdm(meta.iterrows(), total=len(meta), desc="Profiling traces"):
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
            per_trace.append(
                {
                    "trace_path": str(path),
                    "n_packets": 0,
                    "duration_s": 0.0,
                    "max_gap_s": 0.0,
                    "mean_gap_s": 0.0,
                }
            )
            continue

        elapsed = (t - t[0]).to(dtype=t.dtype)
        elapsed_np = elapsed.detach().cpu().numpy().astype(np.float64)

        sums[:n] += elapsed_np
        sums2[:n] += elapsed_np * elapsed_np
        counts[:n] += 1

        if n >= 2:
            gaps = np.diff(t.detach().cpu().numpy().astype(np.float64))
            max_gap = float(np.max(gaps))
            mean_gap = float(np.mean(gaps))
        else:
            max_gap = 0.0
            mean_gap = 0.0

        per_trace.append(
            {
                "trace_path": str(path),
                "n_packets": n,
                "duration_s": float(elapsed_np[n - 1]),
                "max_gap_s": max_gap,
                "mean_gap_s": mean_gap,
            }
        )

    mean = np.zeros_like(sums)
    std = np.zeros_like(sums)

    m = counts > 0
    mean[m] = sums[m] / counts[m]
    var = np.zeros_like(sums)
    var[m] = sums2[m] / counts[m] - mean[m] * mean[m]
    var = np.maximum(var, 0.0)
    std[m] = np.sqrt(var[m])

    return mean, std, counts, per_trace


def _plot_profile(
    *,
    mean: np.ndarray,
    std: np.ndarray,
    counts: np.ndarray,
    out: Path,
    title: str,
) -> None:
    x = np.arange(1, len(mean) + 1)
    m = counts > 0

    fig, (ax0, ax1) = plt.subplots(
        2,
        1,
        figsize=(12, 8),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )

    y = mean[m]
    s = std[m]
    xx = x[m]

    ax0.plot(xx, y, color="#0f766e", lw=2.0, label="mean duration")
    ax0.fill_between(
        xx,
        np.maximum(0.0, y - s),
        y + s,
        color="#14b8a6",
        alpha=0.25,
        label="mean ± std",
    )
    ax0.set_ylabel("Duration until packet [s]")
    ax0.legend(frameon=False, loc="upper left")
    ax0.grid(alpha=0.2)

    ax1.plot(xx, counts[m], color="#475569", lw=1.5)
    ax1.set_ylabel("Trace count")
    ax1.set_xlabel("Packet index")
    ax1.grid(alpha=0.2)

    fig.suptitle(title)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=170)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=str(Datasets.BIGENOUGH))
    ap.add_argument("--n_packets", type=int, default=3000)
    ap.add_argument("--trim_beginning", type=int, default=10)
    ap.add_argument("--max_traces", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--out",
        default="experiment/obsrl/codex_debug/trace_duration_profile.png",
    )
    ap.add_argument(
        "--profile_csv",
        default="experiment/obsrl/codex_debug/trace_duration_profile.csv",
    )
    ap.add_argument(
        "--per_trace_csv",
        default="experiment/obsrl/codex_debug/trace_duration_per_trace.csv",
    )
    ap.add_argument("--gap_threshold_s", type=float, default=2.0)
    ap.add_argument(
        "--blacklist_csv",
        default="experiment/obsrl/codex_debug/trace_blacklist_by_gap.csv",
    )
    args = ap.parse_args()

    meta = load_dataset_meta_df(args.dataset, include_xv_cols=False)
    meta = _subset_meta(meta, max_traces=int(args.max_traces), seed=int(args.seed))

    mean, std, counts, per_trace = _compute_profile(
        meta,
        n_packets=int(args.n_packets),
        trim_beginning=int(args.trim_beginning),
    )

    out = Path(args.out)
    title = (
        f"Duration profile ({len(meta)} traces, n_packets={args.n_packets}, "
        + f"trim={args.trim_beginning})"
    )
    _plot_profile(mean=mean, std=std, counts=counts, out=out, title=title)

    x = np.arange(1, len(mean) + 1)
    prof_df = pd.DataFrame(
        {
            "packet_index": x,
            "mean_duration_s": mean,
            "std_duration_s": std,
            "n_traces": counts,
        }
    )
    profile_csv = Path(args.profile_csv)
    profile_csv.parent.mkdir(parents=True, exist_ok=True)
    prof_df.to_csv(profile_csv, index=False)

    per_trace_df = pd.DataFrame(per_trace).sort_values("max_gap_s", ascending=False)
    per_trace_csv = Path(args.per_trace_csv)
    per_trace_csv.parent.mkdir(parents=True, exist_ok=True)
    per_trace_df.to_csv(per_trace_csv, index=False)

    th = float(args.gap_threshold_s)
    blacklist = per_trace_df[per_trace_df["max_gap_s"] >= th].copy()
    blacklist_csv = Path(args.blacklist_csv)
    blacklist_csv.parent.mkdir(parents=True, exist_ok=True)
    blacklist.to_csv(blacklist_csv, index=False)

    print(f"OK: saved profile plot -> {out}")
    print(f"OK: saved packet profile csv -> {profile_csv}")
    print(f"OK: saved per-trace metrics -> {per_trace_csv}")
    print(
        "OK: saved blacklist candidates "
        + f"(max_gap_s >= {th:.3f}, n={len(blacklist)}) -> {blacklist_csv}"
    )
    if len(per_trace_df) > 0:
        top = per_trace_df.head(5)
        print("Top-5 traces by max_gap_s:")
        for _, r in top.iterrows():
            print(
                "- "
                + f"max_gap_s={float(r['max_gap_s']):.3f}, "
                + f"duration_s={float(r['duration_s']):.3f}, "
                + f"n_packets={int(r['n_packets'])}, path={r['trace_path']}"
            )


if __name__ == "__main__":
    main()
