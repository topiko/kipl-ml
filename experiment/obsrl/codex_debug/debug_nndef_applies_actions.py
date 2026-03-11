from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn

from kipl_ml.data.utils import Datasets, get_std_trace_dict, load_dataset_meta_df
from kipl_ml.defences.nndefs import RNNDef
from kipl_ml.models.trgen import AGENT1
from kipl_ml.rl.simulate import (
    policy_obfuscate_trace_single_pass,
    policy_obfuscate_trace_streaming,
)
from kipl_ml.trace.enums import Feats


class _FixedHead(nn.Module):
    def __init__(self, out_dim: int, idx: int, high: float = 80.0):
        super().__init__()
        self.out_dim = int(out_dim)
        self.idx = int(idx)
        self.high = float(high)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, l, _ = x.shape
        y = torch.full((b, l, self.out_dim), -self.high, device=x.device, dtype=x.dtype)
        y[..., self.idx] = self.high
        return y


def _sorted_rows(X: dict[Feats, torch.Tensor]) -> np.ndarray:
    n = int((X[Feats.DIRS] != 0).sum().item())
    t = np.round(X[Feats.TIMES][:n].detach().cpu().numpy().astype(np.float64), 6)
    d = X[Feats.DIRS][:n].detach().cpu().numpy().astype(np.int32)
    p = X[Feats.PADDING][:n].detach().cpu().numpy().astype(np.int32)
    a = np.stack([t, d, p], axis=1)
    order = np.lexsort((a[:, 2], a[:, 1], a[:, 0]))
    return a[order]


def _assert_same_trace(
    got: dict[Feats, torch.Tensor],
    ref: dict[Feats, torch.Tensor],
    *,
    title: str,
) -> None:
    a = _sorted_rows(got)
    b = _sorted_rows(ref)
    if a.shape != b.shape or not np.array_equal(a, b):
        raise AssertionError(f"{title}: nndef output differs from direct policy obfuscation")


def _to_model_input(
    trace: dict[Feats, torch.Tensor],
    *,
    n_packets: int,
    device: torch.device,
) -> dict[Feats, torch.Tensor]:
    return {
        k: trace[k][:n_packets].unsqueeze(0).to(device).float()
        for k in (Feats.TIMES, Feats.DIRS, Feats.PADDING)
    }


def _run_direct(
    obs: nn.Module,
    trace_in: dict[Feats, torch.Tensor],
    *,
    sample: bool,
    extend_end_s: float,
    max_packets: int,
) -> dict[Feats, torch.Tensor]:
    if getattr(obs, "enable_delay", False):
        out = policy_obfuscate_trace_streaming(
            obs,
            trace_in,
            sample=sample,
            extend_end_s=extend_end_s,
            max_packets=max_packets,
        )
    else:
        out = policy_obfuscate_trace_single_pass(
            obs,
            trace_in,
            sample=sample,
            extend_end_s=extend_end_s,
        )
        out = {k: v[:, :max_packets] for k, v in out.items()}

    return {k: v.squeeze(0).detach().cpu() for k, v in out.items()}


def _run_nndef(
    obs: nn.Module,
    trace_path: Path,
    *,
    n_packets: int,
    sample: bool,
    extend_end_s: float,
    max_packets: int,
) -> dict[Feats, torch.Tensor]:
    defence = RNNDef(
        network_delay_millis=(0, 0),
        network_pps=(40_000, 40_000),
        obs_model=obs,
        n_packets=n_packets,
        simul_kwargs={
            "sample": sample,
            "extend_end_s": extend_end_s,
            "max_packets": max_packets,
        },
    )
    out = defence(trace_path)
    return {k: out[k].detach().cpu() for k in (Feats.TIMES, Feats.DIRS, Feats.PADDING)}


def _check_send_actions_applied(
    raw_trim: dict[Feats, torch.Tensor], out: dict[Feats, torch.Tensor]
) -> None:
    p0 = int(raw_trim[Feats.PADDING].sum().item())
    p1 = int(out[Feats.PADDING].sum().item())
    if p1 <= p0:
        raise AssertionError(
            "Forced SEND policy did not increase padding packets in output trace"
        )


def _check_delay_actions_applied(
    raw_trim: dict[Feats, torch.Tensor], out: dict[Feats, torch.Tensor]
) -> None:
    m0 = (raw_trim[Feats.DIRS] != 0) & (raw_trim[Feats.PADDING] == 0)
    m1 = (out[Feats.DIRS] != 0) & (out[Feats.PADDING] == 0)

    t0 = torch.sort(raw_trim[Feats.TIMES][m0].to(torch.float64))[0]
    t1 = torch.sort(out[Feats.TIMES][m1].to(torch.float64))[0]
    if t0.shape != t1.shape:
        raise AssertionError("Forced DELAY policy changed number of non-padding packets")
    if torch.allclose(t0, t1, atol=1e-9, rtol=0.0):
        raise AssertionError("Forced DELAY policy did not modify non-padding timestamps")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=str(Datasets.BIGENOUGH))
    ap.add_argument("--idx", type=int, default=0)
    ap.add_argument("--n_packets", type=int, default=1000)
    ap.add_argument("--dt", type=float, default=0.02)
    ap.add_argument("--max_silence_s", type=float, default=0.1)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_search", type=int, default=64)
    args = ap.parse_args()

    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))

    device = torch.device(args.device)

    meta = load_dataset_meta_df(args.dataset, include_xv_cols=False)
    idx = int(args.idx) % len(meta)
    trace_path = Path(meta.iloc[idx]["trace_path"])

    raw = get_std_trace_dict(trace_path)
    raw_trim = {
        k: raw[k][: int(args.n_packets)].float()
        for k in (Feats.TIMES, Feats.DIRS, Feats.PADDING)
    }
    trace_in = _to_model_input(raw, n_packets=int(args.n_packets), device=device)

    # Case 1: forced SEND_BOTH with fixed send-count/time heads.
    obs_send = AGENT1(
        time_step=float(args.dt),
        max_silence_s=float(args.max_silence_s),
        enable_delay=False,
        send_mode="fixed",
        prob_eps=0.0,
    ).to(device)
    obs_send.actor["action_selection"] = _FixedHead(4, 3).to(device)
    obs_send.actor["send_count_u"] = _FixedHead(len(obs_send.send_count_bins), 1).to(device)
    obs_send.actor["send_count_d"] = _FixedHead(len(obs_send.send_count_bins), 1).to(device)
    obs_send.actor["send_time_u"] = _FixedHead(len(obs_send.send_time_bins), 0).to(device)
    obs_send.actor["send_time_d"] = _FixedHead(len(obs_send.send_time_bins), 0).to(device)
    obs_send.eval()

    send_ref = _run_direct(
        obs_send,
        {k: v.clone() for k, v in trace_in.items()},
        sample=False,
        extend_end_s=0.0,
        max_packets=int(args.n_packets),
    )
    send_out = _run_nndef(
        obs_send,
        trace_path,
        n_packets=int(args.n_packets),
        sample=False,
        extend_end_s=0.0,
        max_packets=int(args.n_packets),
    )
    _assert_same_trace(send_out, send_ref, title="SEND_BOTH")
    _check_send_actions_applied(raw_trim, send_out)

    # Case 2: forced DELAY-only.
    obs_delay = AGENT1(
        time_step=float(args.dt),
        max_silence_s=float(args.max_silence_s),
        enable_delay=True,
        send_mode="fixed",
        prob_eps=0.0,
    ).to(device)
    obs_delay.actor["action_selection"] = _FixedHead(5, 4).to(device)
    obs_delay.actor["send_count_u"] = _FixedHead(len(obs_delay.send_count_bins), 0).to(device)
    obs_delay.actor["send_count_d"] = _FixedHead(len(obs_delay.send_count_bins), 0).to(device)
    obs_delay.actor["send_time_u"] = _FixedHead(len(obs_delay.send_time_bins), 0).to(device)
    obs_delay.actor["send_time_d"] = _FixedHead(len(obs_delay.send_time_bins), 0).to(device)
    obs_delay.eval()

    delay_idx = idx
    delay_path = trace_path
    delay_raw_trim = raw_trim
    delay_ref = _run_direct(
        obs_delay,
        {k: v.clone() for k, v in trace_in.items()},
        sample=False,
        extend_end_s=0.0,
        max_packets=int(args.n_packets),
    )
    for off in range(int(args.max_search)):
        try:
            _check_delay_actions_applied(delay_raw_trim, delay_ref)
            break
        except AssertionError:
            delay_idx = (idx + off + 1) % len(meta)
            delay_path = Path(meta.iloc[delay_idx]["trace_path"])
            raw_d = get_std_trace_dict(delay_path)
            delay_raw_trim = {
                k: raw_d[k][: int(args.n_packets)].float()
                for k in (Feats.TIMES, Feats.DIRS, Feats.PADDING)
            }
            trace_in_d = _to_model_input(raw_d, n_packets=int(args.n_packets), device=device)
            delay_ref = _run_direct(
                obs_delay,
                trace_in_d,
                sample=False,
                extend_end_s=0.0,
                max_packets=int(args.n_packets),
            )
    else:
        raise AssertionError(
            "Could not find trace where forced DELAY modifies non-padding timestamps"
        )

    delay_out = _run_nndef(
        obs_delay,
        delay_path,
        n_packets=int(args.n_packets),
        sample=False,
        extend_end_s=0.0,
        max_packets=int(args.n_packets),
    )
    _assert_same_trace(delay_out, delay_ref, title="DELAY_ONLY")
    _check_delay_actions_applied(delay_raw_trim, delay_out)

    print("OK: RNNDef output matches direct policy obfuscation")
    print("OK: forced SEND actions are reflected in simulated trace")
    print(f"OK: forced DELAY actions are reflected in simulated trace (idx={delay_idx})")


if __name__ == "__main__":
    main()
