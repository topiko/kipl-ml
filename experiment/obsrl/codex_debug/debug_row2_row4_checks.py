"""Comprehensive row2 == row4-padding invariant checks.

row2: obs features (fd)
row4: final defended trace (X_obs)
"""

from __future__ import annotations

import argparse

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
from kipl_ml.rl.enums import Actions
from kipl_ml.rl.simulate import policy_rollout_streaming
from kipl_ml.trace.enums import Feats
from kipl_ml.trace.features import FeatureTrs


class PatternSelector(torch.nn.Module):
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


def _sorted_rows(X: dict[Feats, torch.Tensor], n: int, idx: int) -> np.ndarray:
    t = X[Feats.TIMES][idx, :n].detach().cpu().numpy().astype(np.float64)
    d = X[Feats.DIRS][idx, :n].detach().cpu().numpy().astype(np.float64)
    p = X[Feats.PADDING][idx, :n].detach().cpu().numpy().astype(np.float64)
    t = np.round(t, 6)
    rows = np.stack([t, d, p], axis=1)
    order = np.lexsort((rows[:, 2], rows[:, 1], rows[:, 0]))
    return rows[order]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=str(Datasets.BIGENOUGH))
    ap.add_argument("--trace_len", type=int, default=1000)
    ap.add_argument("--trim_beginning", type=int, default=10)
    ap.add_argument("--dt", type=float, default=0.02)
    ap.add_argument("--max_silence_s", type=float, default=0.1)
    ap.add_argument("--extend_end_s", type=float, default=0.0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--start_idx", type=int, default=0)
    ap.add_argument("--n", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_failures", type=int, default=20)
    ap.add_argument(
        "--selector_pattern",
        choices=["delay_only", "send_and_delay_cycle", "do_nothing", "all"],
        default="all",
    )
    args = ap.parse_args()

    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))

    device = torch.device(args.device)

    _, ds_valid, _ = get_train_valid_test(
        dataset=args.dataset,
        label=assets.PAGE_LABEL,
        n_splits=5,
        test_xv=0,
        feature_trs=FeatureTrs(feature_names=[Feats.DIRS, Feats.TIMES], n_packets=args.trace_len),
        trim_raw=args.trim_beginning,
    )

    n = max(1, int(args.n))
    start = max(0, int(args.start_idx))
    if start >= len(ds_valid):
        raise ValueError(f"start_idx={start} outside dataset of size {len(ds_valid)}")

    idxs = [(start + i) % len(ds_valid) for i in range(n)]
    patterns = (
        ["do_nothing", "send_and_delay_cycle", "delay_only"]
        if args.selector_pattern == "all"
        else [args.selector_pattern]
    )

    failures: list[str] = []
    checked = 0

    for pattern_name in patterns:
        enable_delay = pattern_name in {"delay_only", "send_and_delay_cycle"}
        obs = AGENT1(
            time_step=float(args.dt),
            max_silence_s=float(args.max_silence_s),
            enable_delay=enable_delay,
            send_mode="fixed",
            prob_eps=0.0,
        ).to(device)
        obs.eval()

        selector = PatternSelector(
            pattern=_pattern(pattern_name),
            n_actions=5 if enable_delay else 4,
        )
        obs.actor["action_selection"] = selector.to(device)

        for ds_idx in idxs:
            X, _ = ds_valid[int(ds_idx)]
            Xb = {k: v.unsqueeze(0).to(device) for k, v in X.items()}
            if Feats.PADDING not in Xb:
                Xb[Feats.PADDING] = torch.zeros_like(Xb[Feats.TIMES])

            with torch.no_grad():
                fd, act_times, actions, _, _, _, _, X_obs = policy_rollout_streaming(
                    obs,
                    {k: v.clone() for k, v in Xb.items()},
                    sample=False,
                    extend_end_s=float(args.extend_end_s),
                    max_packets=None,
                )

            row24 = check_row2_equals_row4_minus_padding(
                fd,
                X_obs,
                dt_s=float(args.dt),
                idx=0,
                max_report=20,
            )
            if not row24.ok:
                failures.append(
                    f"pattern={pattern_name} idx={ds_idx}\n{format_row24_report(row24)}"
                )

            fd_ref = check_fd_matches_recomputed_nonpadding(
                fd,
                X_obs,
                dt_s=float(args.dt),
                max_silence_s=float(args.max_silence_s),
                idx=0,
                max_report=20,
            )
            if not fd_ref.ok:
                failures.append(
                    f"pattern={pattern_name} idx={ds_idx}\n{format_recomputed_fd_report(fd_ref)}"
                )

            delay_rep = count_packets_inside_delay_windows(
                act_times,
                actions,
                X_obs,
                dt_s=float(args.dt),
                idx=0,
                max_report=20,
            )
            if delay_rep.bad_all > 0:
                failures.append(
                    f"pattern={pattern_name} idx={ds_idx}\n{format_delay_leak_report(delay_rep)}"
                )

            checked += 1
            if len(failures) >= int(args.max_failures):
                break

        if len(failures) >= int(args.max_failures):
            break

    if failures:
        msg = (
            f"Found {len(failures)} invariant failures across {checked} checks.\n"
            + "\n\n".join(failures[: int(args.max_failures)])
        )
        raise AssertionError(msg)

    print(
        "OK: comprehensive row2==row4-padding checks passed "
        + f"for {checked} rollouts (patterns={','.join(patterns)})"
    )


if __name__ == "__main__":
    main()
