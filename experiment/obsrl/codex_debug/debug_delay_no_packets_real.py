"""Verify delay produces empty windows (real data).

For a real trace batch, force the policy to emit DELAY for N steps and verify:
- No packets appear in X_obs during the blocked interval.
- No packets are observed in obs features (UP/DOWN counts) during the same interval.

This checks the interaction between WindowFeatureStreamer delay and send_exec delay.
"""

from __future__ import annotations

import argparse

import torch

from kipl_ml.data.utils import Datasets, assets
from kipl_ml.data.wf_dataset import get_train_valid_test
from kipl_ml.models.trgen import AGENT1
from kipl_ml.rl.enums import Actions
from kipl_ml.rl.simulate import policy_rollout_streaming
from kipl_ml.rl.utils import _time_to_bin_idx
from kipl_ml.trace.enums import Feats


class _AlwaysDelay(torch.nn.Module):
    def __init__(self, n_actions: int, delay_idx: int = 4, high: float = 50.0):
        super().__init__()
        self.n_actions = int(n_actions)
        self.delay_idx = int(delay_idx)
        self.high = float(high)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, H) -> logits: (B, L, n_actions)
        b, l = int(x.shape[0]), int(x.shape[1])
        logits = torch.zeros((b, l, self.n_actions), device=x.device, dtype=x.dtype)
        logits[..., self.delay_idx] = self.high
        return logits


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=str(Datasets.BIGENOUGH))
    ap.add_argument("--trace_len", type=int, default=1000)
    ap.add_argument("--trim_beginning", type=int, default=10)
    ap.add_argument("--dt", type=float, default=0.02)
    ap.add_argument("--max_silence_s", type=float, default=0.1)
    ap.add_argument("--n_delays", type=int, default=50)
    ap.add_argument("--idx", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    device = torch.device(args.device)

    _, ds_valid, _ = get_train_valid_test(
        dataset=args.dataset,
        label=assets.PAGE_LABEL,
        n_splits=5,
        test_xv=0,
        feature_trs=None,
        trim_raw=args.trim_beginning,
    )

    X, y = ds_valid[int(args.idx)]
    X = {k: v[: args.trace_len].unsqueeze(0).to(device) for k, v in X.items()}
    y = y.unsqueeze(0).to(device)
    _ = y

    obs = AGENT1(
        time_step=float(args.dt),
        max_silence_s=float(args.max_silence_s),
        enable_delay=True,
        send_mode="fixed",
        prob_eps=0.0,
    ).to(device)
    obs.eval()

    # Force DELAY selector.
    obs.actor["action_selection"] = _AlwaysDelay(n_actions=5, delay_idx=4, high=80.0).to(
        device
    )

    with torch.no_grad():
        fd, act_times, actions, _, _, _, _, X_obs = policy_rollout_streaming(
            obs,
            X,
            sample=False,
            extend_end_s=2.0,
            max_packets=None,
        )

    dt = float(args.dt)
    n = int(args.n_delays)

    delay = actions[Actions.DELAY][0]
    t_all = act_times[0]
    # act_times are int bins; -1 for inactive
    m_delay = (t_all >= 0) & (delay > 0)
    if int(m_delay.sum().item()) < n:
        raise AssertionError(f"Only produced {int(m_delay.sum().item())} delay steps, need >= {n}")

    t0s = t_all[m_delay][:n]
    ds = delay[m_delay][:n]

    # 1) Verify no packets in executed trace during blocked interval.
    pkt_t = X_obs[Feats.TIMES][0]
    pkt_m = (X_obs[Feats.DIRS][0] != 0) & torch.isfinite(pkt_t)
    pkt_t = pkt_t[pkt_m]
    pkt_bins = _time_to_bin_idx(pkt_t, dt)
    for t0, dd in zip(t0s, ds):
        s = int(t0.item())
        sh = int(dd.item())
        if sh <= 0:
            continue
        e = s + sh
        in_block = (pkt_bins >= s) & (pkt_bins < e)
        if bool(in_block.any().item()):
            bad = pkt_t[in_block][:20].detach().cpu().tolist()
            raise AssertionError(
                f"Found packets during delay block bins [{s}, {e}): {bad}"
            )

    # 2) Verify obs features observe no packets during same interval.
    w_t = fd[Feats.TIMES][0]
    w_m_valid = w_t >= 0
    w_bins = w_t[w_m_valid]
    up_w = fd[Feats.UP_COUNT][0][w_m_valid]
    down_w = fd[Feats.DOWN_COUNT][0][w_m_valid]

    for t0, dd in zip(t0s, ds):
        s = int(t0.item())
        sh = int(dd.item())
        if sh <= 0:
            continue
        e = s + sh
        m_block = (w_bins >= s) & (w_bins < e)
        if bool(m_block.any().item()):
            up = up_w[m_block]
            down = down_w[m_block]
            if not bool((up == 0).all().item()) or not bool((down == 0).all().item()):
                raise AssertionError(
                    "Observed packets during delay block in obs features: "
                    + f"up_max={int(up.max().item())}, down_max={int(down.max().item())}"
                )

    # 2b) Sanity: totals in features match executed trace totals.
    up_total_fd = int(fd[Feats.UP_COUNT][0][fd[Feats.UP_COUNT][0] >= 0].sum().item())
    down_total_fd = int(fd[Feats.DOWN_COUNT][0][fd[Feats.DOWN_COUNT][0] >= 0].sum().item())
    up_total_x = int((X_obs[Feats.DIRS][0] == 1).sum().item())
    down_total_x = int((X_obs[Feats.DIRS][0] == -1).sum().item())
    if up_total_fd != up_total_x or down_total_fd != down_total_x:
        raise AssertionError(
            "Feature totals mismatch executed trace totals: "
            + f"up(fd)={up_total_fd}, up(X_obs)={up_total_x}, "
            + f"down(fd)={down_total_fd}, down(X_obs)={down_total_x}"
        )

    # 3) Verify first N emitted actions were DELAY.
    delay_steps = delay[act_times[0] >= 0]
    if delay_steps.numel() < n:
        raise AssertionError(f"Only produced {int(delay_steps.numel())} steps, need >= {n}")
    if not bool((delay_steps[:n] > 0).all().item()):
        raise AssertionError("Not all first N actions were DELAY")

    print("OK: no packets observed during forced delay block")


if __name__ == "__main__":
    main()
