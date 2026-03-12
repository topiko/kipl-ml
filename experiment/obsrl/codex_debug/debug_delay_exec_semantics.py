"""Minimal regression checks for DELAY execution semantics.

This script intentionally avoids policy/streaming logic and tests the core
execution behavior implemented in send_exec.
"""

from __future__ import annotations

import torch

from kipl_ml.data.utils import DOWNLOAD, UPLOAD
from kipl_ml.rl.action import TraceExecState
from kipl_ml.rl.enums import Actions
from kipl_ml.rl.utils import _boundary_time_to_bin_idx, _duration_to_bin_offsets
from kipl_ml.trace.enums import Feats


def _assert_close(a: torch.Tensor, b: torch.Tensor, *, msg: str) -> None:
    if not torch.allclose(a, b, atol=1e-5, rtol=0):
        raise AssertionError(f"{msg}:\n{a}\n!=\n{b}")


DT = 0.02


def _exec(
    X: dict[Feats, torch.Tensor],
    act_times: torch.Tensor,
    actions: dict[Actions, torch.Tensor],
) -> dict[Feats, torch.Tensor]:
    """Execute actions via TraceExecState with float-to-int-bin conversion."""
    exec_state = TraceExecState(
        {k: v.clone() for k, v in X.items()},
        time_step_s=DT,
    )
    times_bin = _boundary_time_to_bin_idx(act_times, DT)
    times_bin = torch.where(act_times.isfinite(), times_bin, torch.full_like(times_bin, -1))
    actions = dict(actions)
    for k in (Actions.DELAY, Actions.SEND_UP_AFTER_TIME, Actions.SEND_DOWN_AFTER_TIME):
        if k in actions and actions[k].is_floating_point():
            actions[k] = _duration_to_bin_offsets(actions[k], DT)

    _, T = times_bin.shape
    for t_i in range(T):
        active = times_bin[:, t_i] >= 0
        if not bool(active.any().item()):
            continue
        trace_idx = torch.where(active)[0]
        exec_state.step(
            trace_idx=trace_idx,
            times=times_bin[active, t_i : t_i + 1],
            actions={k: v[active, t_i : t_i + 1] for k, v in actions.items()},
        )
    return exec_state.finalize()


def _time_to_bin_idx(times: torch.Tensor, dt: float) -> torch.Tensor:
    dt_us = max(1, int(round(float(dt) * 1e6)))
    t_us = torch.round(times * 1e6).to(torch.long)
    return torch.div(t_us, dt_us, rounding_mode="floor")


def test_delay_right_edge_clamps_next_window() -> None:
    # dt = 0.02; action time at 0.02 delays [0.02, 0.04) -> 0.04.
    X = {
        Feats.TIMES: torch.tensor([[0.01, 0.03]], dtype=torch.float32),
        Feats.DIRS: torch.tensor([[UPLOAD, DOWNLOAD]], dtype=torch.float32),
        Feats.PADDING: torch.tensor([[0.0, 0.0]], dtype=torch.float32),
    }

    act_times = torch.tensor([[0.02]], dtype=torch.float32)
    actions = {
        Actions.SEND_COUNT_UP: torch.zeros_like(act_times),
        Actions.SEND_COUNT_DOWN: torch.zeros_like(act_times),
        Actions.SEND_UP_AFTER_TIME: torch.zeros_like(act_times),
        Actions.SEND_DOWN_AFTER_TIME: torch.zeros_like(act_times),
        Actions.DELAY: torch.full_like(act_times, 0.02),
    }

    X_obs = _exec(X, act_times, actions)
    exp_bins = torch.tensor([[0, 2]], dtype=torch.long)
    got_bins = _time_to_bin_idx(X_obs[Feats.TIMES], dt=DT)
    _assert_close(got_bins.float(), exp_bins.float(), msg="right-edge delay bin clamp")
    if float(X_obs[Feats.TIMES][0, 1].item()) < 0.04:
        raise AssertionError("Delayed packet must be at or after right edge")


def test_delay_clamps_padding_packets_too() -> None:
    # Step0 schedules one UP padding packet exactly at 0.04.
    # Step1 delays [0.04, 0.06) -> 0.06, so that padding must move to 0.06.
    X = {
        Feats.TIMES: torch.tensor([[0.01]], dtype=torch.float32),
        Feats.DIRS: torch.tensor([[UPLOAD]], dtype=torch.float32),
        Feats.PADDING: torch.tensor([[0.0]], dtype=torch.float32),
    }

    act_times = torch.tensor([[0.02, 0.04]], dtype=torch.float32)
    actions = {
        Actions.SEND_COUNT_UP: torch.tensor([[1.0, 0.0]], dtype=torch.float32),
        Actions.SEND_COUNT_DOWN: torch.tensor([[0.0, 0.0]], dtype=torch.float32),
        Actions.SEND_UP_AFTER_TIME: torch.tensor([[0.02, 0.0]], dtype=torch.float32),
        Actions.SEND_DOWN_AFTER_TIME: torch.zeros_like(act_times),
        Actions.DELAY: torch.tensor([[0.0, 0.02]], dtype=torch.float32),
    }

    X_obs = _exec(X, act_times, actions)

    # Two packets: original at 0.01, padding at 0.06.
    exp_times = torch.tensor([[0.01, 0.06]], dtype=torch.float32)
    exp_pad = torch.tensor([[0.0, 1.0]], dtype=torch.float32)
    exp_bins = torch.tensor([[0, 3]], dtype=torch.long)
    got_bins = _time_to_bin_idx(X_obs[Feats.TIMES], dt=0.02)
    _assert_close(got_bins.float(), exp_bins.float(), msg="delay clamps padding bins")
    if float(X_obs[Feats.TIMES][0, 1].item()) < float(exp_times[0, 1].item()):
        raise AssertionError("Delayed padding packet must be at or after right edge")
    _assert_close(X_obs[Feats.PADDING], exp_pad, msg="padding flag preserved")


def main() -> None:
    torch.set_printoptions(precision=8)
    test_delay_right_edge_clamps_next_window()
    test_delay_clamps_padding_packets_too()
    print("OK: delay execution semantics")


if __name__ == "__main__":
    main()
