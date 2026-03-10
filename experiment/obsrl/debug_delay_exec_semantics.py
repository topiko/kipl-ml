"""Minimal regression checks for DELAY execution semantics.

This script intentionally avoids policy/streaming logic and tests the core
execution behavior implemented in send_exec.
"""

from __future__ import annotations

import torch

from kipl_ml.data.utils import DOWNLOAD, UPLOAD
from kipl_ml.rl.action import send_exec
from kipl_ml.rl.enums import Actions
from kipl_ml.trace.enums import Feats


def _assert_close(a: torch.Tensor, b: torch.Tensor, *, msg: str) -> None:
    if not torch.allclose(a, b, atol=0, rtol=0):
        raise AssertionError(f"{msg}:\n{a}\n!=\n{b}")


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

    X_obs = send_exec(X, act_times, actions)
    exp_times = torch.tensor([[0.01, 0.04]], dtype=torch.float32)
    _assert_close(X_obs[Feats.TIMES], exp_times, msg="right-edge delay clamps next-bin")


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

    X_obs = send_exec(X, act_times, actions)

    # Two packets: original at 0.01, padding at 0.06.
    exp_times = torch.tensor([[0.01, 0.06]], dtype=torch.float32)
    exp_pad = torch.tensor([[0.0, 1.0]], dtype=torch.float32)
    _assert_close(X_obs[Feats.TIMES], exp_times, msg="delay clamps padding times")
    _assert_close(X_obs[Feats.PADDING], exp_pad, msg="padding flag preserved")


def main() -> None:
    torch.set_printoptions(precision=8)
    test_delay_right_edge_clamps_next_window()
    test_delay_clamps_padding_packets_too()
    print("OK: delay execution semantics")


if __name__ == "__main__":
    main()
