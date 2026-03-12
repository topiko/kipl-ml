"""Real-data plotting + execution consistency check for obsrl delay mode.

This script runs a defended rollout on a real trace, generates a debug plot,
and verifies invariants that help diagnose potential TraceExecState.finalize bugs.
"""

from __future__ import annotations

import argparse
from pathlib import Path

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
from kipl_ml.data.utils import DOWNLOAD, UPLOAD, Datasets, assets
from kipl_ml.data.wf_dataset import get_train_valid_test
from kipl_ml.models.trgen import AGENT1
from kipl_ml.rl.enums import Actions
from kipl_ml.rl.simulate import policy_rollout_streaming
from kipl_ml.tools.plottr import plot_actions, plot_tam
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


def _time_to_bin_idx(times: torch.Tensor, dt: float) -> torch.Tensor:
    dt_us = max(1, int(round(float(dt) * 1e6)))
    t_us = torch.round(times * 1e6).to(torch.long)
    return torch.div(t_us, dt_us, rounding_mode="floor")


def _tam_from_trace(
    X: dict[Feats, torch.Tensor],
    dt: float,
    *,
    idx: int = 0,
) -> dict[Feats, torch.Tensor]:
    t = X[Feats.TIMES][idx]
    d = X[Feats.DIRS][idx]
    p = X[Feats.PADDING][idx] != 0

    m = (d != 0) & torch.isfinite(t)
    if not bool(m.any().item()):
        z = torch.zeros((1,), device=t.device, dtype=t.dtype)
        return {
            Feats.TAM_TIMES: z,
            Feats.TAM_UP_COUNTS: z,
            Feats.TAM_DOWN_COUNTS: z,
            Feats.TAM_UP_PAD: z,
            Feats.TAM_DOWN_PAD: z,
        }

    t = t[m]
    d = d[m]
    p = p[m]
    bins = _time_to_bin_idx(t, dt)
    n_bins = int(bins.max().item()) + 1

    up = torch.zeros((n_bins,), device=t.device, dtype=t.dtype)
    down = torch.zeros((n_bins,), device=t.device, dtype=t.dtype)
    up_pad = torch.zeros((n_bins,), device=t.device, dtype=t.dtype)
    down_pad = torch.zeros((n_bins,), device=t.device, dtype=t.dtype)

    m_up = d == int(UPLOAD)
    m_down = d == int(DOWNLOAD)

    up.scatter_add_(0, bins, m_up.to(t.dtype))
    down.scatter_add_(0, bins, m_down.to(t.dtype))
    up_pad.scatter_add_(0, bins, (m_up & p).to(t.dtype))
    down_pad.scatter_add_(0, bins, (m_down & p).to(t.dtype))

    tb = torch.arange(n_bins, device=t.device, dtype=t.dtype)
    tt = tb * float(dt)

    return {
        Feats.TAM_TIMES: tt,
        Feats.TAM_UP_COUNTS: up,
        Feats.TAM_DOWN_COUNTS: down,
        Feats.TAM_UP_PAD: up_pad,
        Feats.TAM_DOWN_PAD: down_pad,
    }


def _tam_from_fd(fd: dict[Feats, torch.Tensor], *, idx: int = 0) -> dict[Feats, torch.Tensor]:
    t = fd[Feats.TIMES][idx]
    m = torch.isfinite(t)
    if not bool(m.any().item()):
        z = torch.zeros((1,), device=t.device, dtype=t.dtype)
        return {
            Feats.TAM_TIMES: z,
            Feats.TAM_UP_COUNTS: z,
            Feats.TAM_DOWN_COUNTS: z,
            Feats.TAM_UP_PAD: z,
            Feats.TAM_DOWN_PAD: z,
        }

    tt = t[m]
    up = fd[Feats.UP_COUNT][idx][m]
    down = fd[Feats.DOWN_COUNT][idx][m]
    z = torch.zeros_like(up)
    return {
        Feats.TAM_TIMES: tt,
        Feats.TAM_UP_COUNTS: up,
        Feats.TAM_DOWN_COUNTS: down,
        Feats.TAM_UP_PAD: z,
        Feats.TAM_DOWN_PAD: z,
    }


def _maybe_enable_interactive_backend(show: bool, backend: str | None) -> None:
    if backend:
        try:
            plt.switch_backend(backend)
            print(f"Using matplotlib backend: {plt.get_backend()}")
        except Exception as exc:
            raise RuntimeError(
                f"Could not switch to backend '{backend}': {exc}"
            ) from exc

    if not show:
        return

    current = str(plt.get_backend()).lower()
    if current not in {"agg", "module://matplotlib_inline.backend_inline", "inline"}:
        return

    for cand in ("QtAgg", "TkAgg", "GTK3Agg", "WXAgg", "MacOSX"):
        try:
            plt.switch_backend(cand)
            print(f"Using matplotlib backend: {plt.get_backend()}")
            return
        except Exception:
            continue


def _sorted_rows(X: dict[Feats, torch.Tensor], n: int) -> np.ndarray:
    t = X[Feats.TIMES][0, :n].detach().cpu().numpy().astype(np.float64)
    d = X[Feats.DIRS][0, :n].detach().cpu().numpy().astype(np.float64)
    p = X[Feats.PADDING][0, :n].detach().cpu().numpy().astype(np.float64)
    t = np.round(t, 6)
    rows = np.stack([t, d, p], axis=1)
    order = np.lexsort((rows[:, 2], rows[:, 1], rows[:, 0]))
    return rows[order]


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
    ap.add_argument("--backend", default="")
    args = ap.parse_args()

    _maybe_enable_interactive_backend(args.show, args.backend or None)

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

    tam_base = _tam_from_trace(Xb, float(args.dt), idx=0)
    tam_fd = _tam_from_fd(fd, idx=0)
    tam_obs = _tam_from_trace(X_obs, float(args.dt), idx=0)

    plot_tam(tam_base, window_width=float(args.dt), ax=axes[0])
    axes[0].set_title("Base trace (TAM-like)")
    plot_tam(tam_fd, window_width=float(args.dt), ax=axes[1])
    axes[1].set_title("Obs features (TAM-like)")
    plot_actions(act_times, actions, idx=0, ax=axes[2])
    axes[2].set_title("Actions")
    plot_tam(tam_obs, window_width=float(args.dt), ax=axes[3])
    axes[3].set_title("Defended trace (X_obs, TAM-like)")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    if args.show:
        backend = plt.get_backend().lower()
        if backend in {"agg", "module://matplotlib_inline.backend_inline", "inline"}:
            print(
                "NOTE: backend "
                + f"'{backend}' is non-interactive; pass --backend TkAgg/QtAgg to show"
            )
        else:
            try:
                plt.show()
            except Exception as exc:
                print(f"NOTE: backend '{backend}' could not open window: {exc}")
    plt.close(fig)

    print(f"OK: plotted defended real trace -> {out}")
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
