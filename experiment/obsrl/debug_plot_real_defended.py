"""Real-data plotting + execution consistency check for obsrl delay mode.

This script runs a defended rollout on a real trace, generates a debug plot,
and verifies invariants that help diagnose potential TraceExecState.finalize bugs.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from kipl_ml.data.utils import DOWNLOAD, UPLOAD, Datasets, assets
from kipl_ml.data.wf_dataset import get_train_valid_test
from kipl_ml.models.trgen import AGENT1
from kipl_ml.rl.action import send_exec
from kipl_ml.rl.enums import Actions
from kipl_ml.rl.simulate import policy_rollout_streaming
from kipl_ml.tools.plottr import plot_actions, plot_obs_features, plot_trace
from kipl_ml.trace.enums import Feats
from kipl_ml.trace.features import FeatureTrs


class PatternSelector(torch.nn.Module):
    """Deterministic selector logits from a repeating pattern."""

    def __init__(self, pattern: list[int], n_actions: int, high: float = 40.0):
        super().__init__()
        self.pattern = [int(x) for x in pattern]
        self.n_actions = int(n_actions)
        self.high = float(high)
        self._step = 0

    def reset(self) -> None:
        self._step = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, l, _ = x.shape
        logits = torch.zeros((b, l, self.n_actions), device=x.device, dtype=x.dtype)
        idx = self.pattern[self._step % len(self.pattern)]
        logits[..., idx] = self.high
        self._step += 1
        return logits


def _pattern(name: str) -> list[int]:
    # selector indices: 0=DO_NOTHING, 1=SEND_UP, 2=SEND_DOWN, 3=SEND_BOTH, 4=DELAY
    if name == "delay_only":
        return [4]
    if name == "send_and_delay_cycle":
        return [1, 4, 2, 4, 3, 4, 0, 4]
    if name == "do_nothing":
        return [0]
    raise ValueError(f"Unknown pattern: {name}")


def _sorted_rows(X: dict[Feats, torch.Tensor], n: int) -> np.ndarray:
    t = X[Feats.TIMES][0, :n].detach().cpu().numpy().astype(np.float64)
    d = X[Feats.DIRS][0, :n].detach().cpu().numpy().astype(np.float64)
    p = X[Feats.PADDING][0, :n].detach().cpu().numpy().astype(np.float64)
    t = np.round(t, 6)
    rows = np.stack([t, d, p], axis=1)
    order = np.lexsort((rows[:, 2], rows[:, 1], rows[:, 0]))
    return rows[order]


def _check_finalize_matches_send_exec(
    X_base: dict[Feats, torch.Tensor],
    act_times: torch.Tensor,
    actions: dict[Actions, torch.Tensor],
    X_obs: dict[Feats, torch.Tensor],
) -> None:
    X_ref = send_exec(
        {
            Feats.TIMES: X_base[Feats.TIMES].clone(),
            Feats.DIRS: X_base[Feats.DIRS].clone(),
            Feats.PADDING: X_base[Feats.PADDING].clone(),
        },
        act_times,
        actions,
    )

    n_obs = int((X_obs[Feats.DIRS][0] != 0).sum().item())
    n_ref = int((X_ref[Feats.DIRS][0] != 0).sum().item())
    if n_obs != n_ref:
        raise AssertionError(f"Packet count mismatch finalize vs send_exec: {n_obs} vs {n_ref}")

    a = _sorted_rows(X_obs, n_obs)
    b = _sorted_rows(X_ref, n_ref)
    if a.shape != b.shape or not np.array_equal(a, b):
        raise AssertionError("TraceExecState.finalize output differs from send_exec (multiset)")


def _check_fd_counts_match_X_obs_nonpadding(
    fd: dict[Feats, torch.Tensor], X_obs: dict[Feats, torch.Tensor], dt: float
) -> list[tuple[int, float, float, int, int, int, int]]:
    w_t = fd[Feats.TIMES][0]
    w_dt = fd[Feats.Dt][0]
    up_fd = fd[Feats.UP_COUNT][0]
    down_fd = fd[Feats.DOWN_COUNT][0]
    m_w = torch.isfinite(w_t)

    pkt_t = X_obs[Feats.TIMES][0]
    pkt_d = X_obs[Feats.DIRS][0]
    pkt_p = X_obs[Feats.PADDING][0] != 0
    m_pkt = (pkt_d != 0) & (~pkt_p) & torch.isfinite(pkt_t)
    pkt_t = pkt_t[m_pkt]
    pkt_d = pkt_d[m_pkt]

    bad: list[tuple[int, float, float, int, int, int, int]] = []
    for k in torch.where(m_w)[0].tolist():
        t0 = float(w_t[k].item())
        # UP/DOWN counts are per base time bin [t, t+dt), not over Dt.
        # Dt is the gap to the next emitted action window.
        t1 = t0 + float(dt)
        m = (pkt_t >= t0) & (pkt_t < t1)
        up = int((pkt_d[m] == UPLOAD).sum().item())
        down = int((pkt_d[m] == DOWNLOAD).sum().item())
        up0 = int(round(float(up_fd[k].item())))
        down0 = int(round(float(down_fd[k].item())))
        if up != up0 or down != down0:
            bad.append((k, t0, t1, up0, down0, up, down))
            if len(bad) >= 10:
                break

    # Note: per-window mismatches against final X_obs can happen when packets are
    # delayed multiple times before they are eventually emitted. These are useful
    # diagnostics, but not necessarily a correctness violation.
    return bad


def _check_no_packets_inside_delay(
    act_times: torch.Tensor,
    actions: dict[Actions, torch.Tensor],
    X_obs: dict[Feats, torch.Tensor],
) -> tuple[int, int]:
    if Actions.DELAY not in actions:
        return 0, 0

    t_act = act_times[0]
    d_act = actions[Actions.DELAY][0]
    m = torch.isfinite(t_act) & (d_act > 0)
    if not bool(m.any().item()):
        return 0, 0

    pkt_t = X_obs[Feats.TIMES][0]
    pkt_d = X_obs[Feats.DIRS][0]
    pkt_p = X_obs[Feats.PADDING][0] != 0
    m_pkt = (pkt_d != 0) & torch.isfinite(pkt_t)

    t_all = pkt_t[m_pkt]
    t_nopad = pkt_t[m_pkt & (~pkt_p)]

    bad_all = 0
    bad_nopad = 0
    for t0, dd in zip(t_act[m], d_act[m]):
        t1 = t0 + dd
        bad_all += int(((t_all >= t0) & (t_all < t1)).sum().item())
        bad_nopad += int(((t_nopad >= t0) & (t_nopad < t1)).sum().item())
    return bad_all, bad_nopad


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=str(Datasets.BIGENOUGH))
    ap.add_argument("--idx", type=int, default=0)
    ap.add_argument("--trace_len", type=int, default=1000)
    ap.add_argument("--trim_beginning", type=int, default=10)
    ap.add_argument("--dt", type=float, default=0.02)
    ap.add_argument("--max_silence_s", type=float, default=0.1)
    ap.add_argument("--extend_end_s", type=float, default=0.0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument(
        "--selector_pattern",
        choices=["delay_only", "send_and_delay_cycle", "do_nothing"],
        default="send_and_delay_cycle",
    )
    ap.add_argument("--out", default="experiment/obsrl/plot_real_defended_debug.png")
    args = ap.parse_args()

    device = torch.device(args.device)

    _, ds_valid, _ = get_train_valid_test(
        dataset=args.dataset,
        label=assets.PAGE_LABEL,
        n_splits=5,
        test_xv=0,
        feature_trs=FeatureTrs(feature_names=[Feats.DIRS, Feats.TIMES], n_packets=args.trace_len),
        trim_raw=args.trim_beginning,
    )

    X, y = ds_valid[int(args.idx)]
    y = int(y.item())
    Xb = {k: v.unsqueeze(0).to(device) for k, v in X.items()}
    if Feats.PADDING not in Xb:
        Xb[Feats.PADDING] = torch.zeros_like(Xb[Feats.TIMES])

    obs = AGENT1(
        time_step=float(args.dt),
        max_silence_s=float(args.max_silence_s),
        enable_delay=True,
        send_mode="fixed",
        prob_eps=0.0,
    ).to(device)
    obs.eval()

    selector = PatternSelector(pattern=_pattern(args.selector_pattern), n_actions=5)
    obs.actor["action_selection"] = selector.to(device)

    with torch.no_grad():
        fd, act_times, actions, _, _, _, _, X_obs = policy_rollout_streaming(
            obs,
            {k: v.clone() for k, v in Xb.items()},
            sample=False,
            extend_end_s=float(args.extend_end_s),
            max_packets=None,
        )

    # Invariants / diagnostics.
    _check_finalize_matches_send_exec(Xb, act_times, actions, X_obs)
    bad_windows = _check_fd_counts_match_X_obs_nonpadding(fd, X_obs, dt=float(args.dt))

    # Strong global check: totals should still match.
    up_total_fd = float(fd[Feats.UP_COUNT][0].nan_to_num(nan=0.0).sum().item())
    down_total_fd = float(fd[Feats.DOWN_COUNT][0].nan_to_num(nan=0.0).sum().item())
    dirs = X_obs[Feats.DIRS][0]
    pad = X_obs[Feats.PADDING][0] != 0
    up_total_x = float(((dirs == UPLOAD) & (~pad)).sum().item())
    down_total_x = float(((dirs == DOWNLOAD) & (~pad)).sum().item())
    if abs(up_total_fd - up_total_x) > 1e-3 or abs(down_total_fd - down_total_x) > 1e-3:
        raise AssertionError(
            "Global UP/DOWN totals mismatch: "
            + f"fd=({up_total_fd},{down_total_fd}) X_obs=({up_total_x},{down_total_x})"
        )
    bad_all, bad_nopad = _check_no_packets_inside_delay(act_times, actions, X_obs)

    # Plot with real data.
    fig, axes = plt.subplots(4, 1, figsize=(18, 11), sharex=True)
    fig.suptitle(
        f"real defended trace idx={args.idx} label={y} pattern={args.selector_pattern} "
        + f"inside_delay(all={bad_all}, nonpad={bad_nopad})"
    )

    plot_trace(Xb, idx=0, ax=axes[0])
    axes[0].set_title("Base trace")
    plot_obs_features(fd, idx=0, ax=axes[1])
    axes[1].set_title("Obs features")
    plot_actions(act_times, actions, idx=0, ax=axes[2])
    axes[2].set_title("Actions")
    plot_trace(X_obs, idx=0, ax=axes[3])
    axes[3].set_title("Defended trace (X_obs)")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)

    print(f"OK: plotted defended real trace -> {out}")
    print("OK: finalize matches send_exec")
    if bad_windows:
        print(
            "NOTE: per-window fd vs final X_obs mismatches found "
            + f"(expected with repeated delays), n={len(bad_windows)}"
        )
        for k, t0, t1, u0, d0, u1, d1 in bad_windows[:10]:
            print(
                f"  k={k} [{t0:.4f},{t1:.4f}) fd=({u0},{d0}) X_obs=({u1},{d1})"
            )
    print("OK: global fd totals match X_obs (non-padding)")
    print(f"OK: packets inside delay windows all={bad_all}, nonpad={bad_nopad}")


if __name__ == "__main__":
    main()
