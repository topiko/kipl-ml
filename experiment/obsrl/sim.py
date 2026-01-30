import time

import torch
from torch import nn

from kipl_ml.logging.logger import get_logger
from kipl_ml.rl.action import send_exec
from kipl_ml.rl.enums import Actions
from kipl_ml.rl.observation import get_window_feature_dict
from kipl_ml.trace.enums import Feats
from kipl_ml.trace.features import FeatureTrs

logger = get_logger(__name__)


def get_rewards(
    action_times: torch.Tensor,
    X: dict[Feats, torch.Tensor],
    y: torch.Tensor,
    disc_logits: torch.Tensor,
    seq_lens: torch.Tensor,
    reward_scales: dict[str, float] = {"clf_scale": 1.0, "padding_scale": 1.0},
) -> dict[str, torch.Tensor]:
    # Shapes:
    # - action_times: (B, T)
    # - disc_logits: (B, N, C)
    # - seq_lens: (B,)
    N = disc_logits.shape[1]
    bs, T = action_times.shape

    rewards: dict[str, torch.Tensor] = {
        k.replace("_scale", ""): torch.zeros_like(action_times) for k in reward_scales
    }

    # (B, N)
    times = X[Feats.TIMES][:, :N]
    padding = X[Feats.PADDING][:, :N].bool()

    # (B, N, C)
    probs = nn.functional.softmax(disc_logits, dim=-1)
    # (B, N)
    target_probs = probs.gather(2, y[:, None, None].expand(-1, N, 1)).squeeze(-1)

    # Assign each packet time to an action interval [t_i, t_{i+1}).
    # We treat NaNs in action_times as +inf, which makes the last finite action
    # cover the rest of the trace.
    boundaries = action_times.nan_to_num(nan=float("inf"))
    # (B, N) in [0..T], then shift to [ -1 .. T-1 ]
    idx = torch.searchsorted(boundaries, times, right=True) - 1
    valid_idx = (idx >= 0) & (idx < T)

    # Ignore padded packets beyond seq_lens.
    pkt_valid = (
        torch.arange(N, device=times.device)[None, :]
        < seq_lens.to(times.device)[:, None]
    )
    valid = valid_idx & pkt_valid

    idx_clamped = idx.clamp(0, T - 1)

    # Padding penalty: count padding packets per action interval.
    pad_w = (padding & valid).to(times.dtype)
    npad = torch.zeros((bs, T), device=times.device, dtype=times.dtype).scatter_add_(
        1, idx_clamped, pad_w
    )
    rewards["padding"] -= npad * reward_scales["padding_scale"]

    # Classifier reward: mean over normal packets per interval.
    p_lvl = 0.1
    normal_w = ((~padding) & valid).to(times.dtype)
    normal_cnt = torch.zeros(
        (bs, T), device=times.device, dtype=times.dtype
    ).scatter_add_(1, idx_clamped, normal_w)
    normal_sum = torch.zeros(
        (bs, T), device=times.device, dtype=times.dtype
    ).scatter_add_(1, idx_clamped, (p_lvl - target_probs) * normal_w)
    mean_p = torch.where(normal_cnt > 0, normal_sum / normal_cnt, 0.0)
    rewards["clf"] += mean_p * reward_scales["clf_scale"]

    return rewards


def rollout(
    obs: nn.Module,
    critic: nn.Module,
    disc: nn.Module,
    X: dict[Feats, torch.Tensor],
    y: torch.Tensor,
    disc_league: list[tuple[int, nn.Module.state_dict]],
    disc_features: FeatureTrs | None = None,
    detach_period: int = 20,
    reward_scales: dict[str, float] | None = None,
    timing: dict[str, float] | None = None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    dict[str, torch.Tensor] | None,
    dict[str, torch.Tensor],
    torch.Tensor,
    dict[Actions, torch.Tensor],
    dict[Feats, torch.Tensor],
    dict[Feats, torch.Tensor],
]:
    hobs = None
    rewards = None
    device = y.device

    # Timing defaults (used only when reward_scales is not None)
    disc_fwd_s = 0.0
    rewards_s = 0.0
    critic_s = 0.0
    league_values = None

    t0 = time.perf_counter()
    if X[Feats.TIMES].isnan().any():
        raise ValueError("NaN in times feature")

    fd = get_window_feature_dict(
        X, obs.time_step, obs.max_silence_s, features=obs.features, extend_end_s=2
    )
    if X[Feats.TIMES].isnan().any():
        raise ValueError("NaN in times feature")

    t1 = time.perf_counter()

    # We need the seq. lens in forward.
    action_seq_lens = fd.pop(Feats.SEQ_LENS)
    L = action_seq_lens.max().item()

    act_times, actions, log_ps, sel_probs, _, entropies, h = obs.act(
        fd, hobs, h_detach_period=detach_period, seq_lens=action_seq_lens
    )

    if h is not None:
        h_norm = h[0].norm(2, dim=-1).max().item()
        c_norm = h[1].norm(2, dim=-1).max().item()
        if h_norm > 100 or c_norm > 400:
            print("Huge hidden/cell:", h_norm, c_norm)

    t2 = time.perf_counter()
    send_timing: dict[str, float] = {}
    Xobs = send_exec(X, act_times, actions, timing=send_timing)

    t3 = time.perf_counter()

    if reward_scales is not None:
        current_disc_state = {
            k: v.detach().clone() for k, v in disc.state_dict().items()
        }
        rewards_l = []
        league_values_l = []
        packet_seq_lens_gpu = (Xobs[Feats.DIRS] != 0).sum(dim=1).long()
        packet_seq_lens = packet_seq_lens_gpu.cpu()

        if disc_features is not None:
            ttf0 = time.perf_counter()
            X_ = disc_features.transform_batch(Xobs)
            if timing is not None:
                timing["rollout_disc_features_ms"] = (
                    time.perf_counter() - ttf0
                ) * 1000
        else:
            X_ = Xobs

        if Feats.LABEL in critic.features:
            fd[Feats.LABEL] = y.unsqueeze(1).repeat(1, L)

        disc_load_s = 0.0
        disc_pack_s = 0.0
        disc_rnn_s = 0.0
        disc_head_s = 0.0
        disc_cat_s = 0.0
        disc_unpack_s = 0.0

        for disc_id, state_d in disc_league:
            tload0 = time.perf_counter()
            disc.load_state_dict({k: v.to(device) for k, v in state_d.items()})
            disc_load_s += time.perf_counter() - tload0

            hdisc = None
            disc.eval()
            with torch.no_grad():
                tdisc0 = time.perf_counter()
                disc_timing: dict[str, float] = {}
                logits, hdisc = disc.pack_and_forward(
                    X_,
                    hdisc,
                    packet_seq_lens,
                    timing=disc_timing,
                    timing_prefix="disc",
                )
                disc_fwd_s += time.perf_counter() - tdisc0
                disc_cat_s += disc_timing.get("disc_cat_mask_ms", 0.0) / 1000
                disc_pack_s += disc_timing.get("disc_pack_ms", 0.0) / 1000
                disc_rnn_s += disc_timing.get("disc_rnn_ms", 0.0) / 1000
                disc_unpack_s += disc_timing.get("disc_unpack_ms", 0.0) / 1000
                disc_head_s += disc_timing.get("disc_head_ms", 0.0) / 1000

            trew0 = time.perf_counter()
            rewards_ = get_rewards(
                act_times,
                Xobs,
                y,
                logits,
                packet_seq_lens_gpu,
                reward_scales=reward_scales,
            )
            rewards_s += time.perf_counter() - trew0

            rewards_l.append(rewards_)

            if Feats.DISC_ID in critic.features:
                fd[Feats.DISC_ID] = torch.full_like(fd[critic.features[0]], disc_id)

            tcrit0 = time.perf_counter()
            league_values_ = critic(
                fd, None, h_detach_period=detach_period, seq_lens=action_seq_lens
            )[0][Feats.STATE_VALUE]
            critic_s += time.perf_counter() - tcrit0

            league_values_l.append(league_values_)

        # Rewards from different disc. checkpoints.
        rewards = {
            k: torch.stack([r[k] for r in rewards_l], dim=0)
            for k in rewards_l[0].keys()
        }

        # Make the values tensor
        league_values = torch.stack(league_values_l, dim=0)

        # Make sure correct state is restored.
        disc.load_state_dict(current_disc_state)

    t4 = time.perf_counter()

    if timing is not None:
        timing["rollout_fd_ms"] = (t1 - t0) * 1000
        timing["rollout_act_ms"] = (t2 - t1) * 1000
        timing["rollout_send_exec_ms"] = (t3 - t2) * 1000
        for k, v in send_timing.items():
            timing[f"{k}"] = float(v)
        timing["rollout_disc_forward_ms"] = disc_fwd_s * 1000
        timing["rollout_disc_load_state_ms"] = disc_load_s * 1000
        timing["rollout_disc_cat_mask_ms"] = disc_cat_s * 1000
        timing["rollout_disc_pack_ms"] = disc_pack_s * 1000
        timing["rollout_disc_rnn_ms"] = disc_rnn_s * 1000
        timing["rollout_disc_unpack_ms"] = disc_unpack_s * 1000
        timing["rollout_disc_head_ms"] = disc_head_s * 1000
        timing["rollout_rewards_ms"] = rewards_s * 1000
        timing["rollout_critic_ms"] = critic_s * 1000
        timing["rollout_disc_rewards_critic_ms"] = (t4 - t3) * 1000
        timing["rollout_total_ms"] = (t4 - t0) * 1000
        timing["rollout_league_n"] = float(len(disc_league))

    return (
        log_ps,
        sel_probs,
        league_values,
        rewards,
        entropies,
        act_times,
        actions,
        Xobs,
        fd,
    )
