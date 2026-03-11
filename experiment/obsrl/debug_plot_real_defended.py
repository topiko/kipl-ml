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

from experiment.obsrl.invariants import (
    check_fd_matches_recomputed_nonpadding,
    check_row2_equals_row4_minus_padding,
    count_packets_inside_delay_windows,
    format_delay_leak_report,
    format_recomputed_fd_report,
    format_row24_report,
)
from kipl_ml.data.utils import Datasets, assets
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
    dt_s: float,
) -> None:
    X_ref = send_exec(
        {
            Feats.TIMES: X_base[Feats.TIMES].clone(),
            Feats.DIRS: X_base[Feats.DIRS].clone(),
            Feats.PADDING: X_base[Feats.PADDING].clone(),
        },
        act_times,
        actions,
        time_step_s=dt_s,
    )

    n_obs = int((X_obs[Feats.DIRS][0] != 0).sum().item())
    n_ref = int((X_ref[Feats.DIRS][0] != 0).sum().item())
    if n_obs != n_ref:
        raise AssertionError(f"Packet count mismatch finalize vs send_exec: {n_obs} vs {n_ref}")

    a = _sorted_rows(X_obs, n_obs)
    b = _sorted_rows(X_ref, n_ref)
    if a.shape != b.shape or not np.array_equal(a, b):
        raise AssertionError("TraceExecState.finalize output differs from send_exec (multiset)")


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
    ap.add_argument("--show", action="store_true")
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
    _check_finalize_matches_send_exec(Xb, act_times, actions, X_obs, dt_s=float(args.dt))
    row24 = check_row2_equals_row4_minus_padding(
        fd,
        X_obs,
        dt_s=float(args.dt),
        idx=0,
        max_report=25,
    )
    if not row24.ok:
        raise AssertionError(format_row24_report(row24))

    delay_leak = count_packets_inside_delay_windows(
        act_times,
        actions,
        X_obs,
        dt_s=float(args.dt),
        idx=0,
        max_report=25,
    )
    if delay_leak.bad_all > 0:
        raise AssertionError(format_delay_leak_report(delay_leak))

    fd_recomputed = check_fd_matches_recomputed_nonpadding(
        fd,
        X_obs,
        dt_s=float(args.dt),
        max_silence_s=float(args.max_silence_s),
        idx=0,
        max_report=25,
    )
    if not fd_recomputed.ok:
        raise AssertionError(format_recomputed_fd_report(fd_recomputed))

    # Plot with real data.
    fig, axes = plt.subplots(4, 1, figsize=(18, 11), sharex=True)
    fig.suptitle(
        f"real defended trace idx={args.idx} label={y} pattern={args.selector_pattern} "
        + (
            "inside_delay(all="
            + f"{delay_leak.bad_all}, nonpad={delay_leak.bad_nonpadding})"
        )
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
    if args.show:
        plt.show()
    plt.close(fig)

    print(f"OK: plotted defended real trace -> {out}")
    print("OK: finalize matches send_exec")
    print("OK: row2(fd) == row4(X_obs)-padding")
    print(
        "OK: totals fd(up,down)="
        + f"({row24.total_fd_up},{row24.total_fd_down}) "
        + f"X_obs=({row24.total_x_up},{row24.total_x_down})"
    )
    print(
        "OK: packets inside delay windows all/nonpad="
        + f"{delay_leak.bad_all}/{delay_leak.bad_nonpadding}"
    )
    print("OK: fd == recomputed non-padding windows")


if __name__ == "__main__":
    main()
