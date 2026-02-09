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
    # probs = nn.functional.softmax(disc_logits, dim=-1)
    # (B, N)
    # target_probs = probs.gather(2, y[:, None, None].expand(-1, N, 1)).squeeze(-1)

    # (B, N, 1)
    target_idx = y[:, None, None].expand(-1, N, 1)

    # (B, N)
    target_logits = disc_logits.gather(2, target_idx).squeeze(-1)
    # (B, N, C)
    other_logits = disc_logits.scatter(2, target_idx, float("-inf"))
    # (B, N)
    rest_lse = torch.logsumexp(other_logits, dim=-1)

    # (B, N)
    m = target_logits - rest_lse

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
    # =============================================
    pad_w = (padding & valid).to(times.dtype)
    npad = torch.zeros((bs, T), device=times.device, dtype=times.dtype).scatter_add_(
        1, idx_clamped, pad_w
    )
    rewards["padding"] -= npad * reward_scales["padding_scale"]

    # Classifier reward: mean over normal packets per interval.
    # =============================================
    normal_w = ((~padding) & valid).to(times.dtype)
    normal_cnt = torch.zeros(
        (bs, T), device=times.device, dtype=times.dtype
    ).scatter_add_(1, idx_clamped, normal_w)

    r_pkt = torch.clamp(-m, min=-10, max=10.0)
    normal_sum = torch.zeros(
        (bs, T), device=times.device, dtype=times.dtype
    ).scatter_add_(1, idx_clamped, r_pkt * normal_w)
    mean_p = torch.where(normal_cnt > 0, normal_sum / normal_cnt, 0.0)
    rewards["clf"] += mean_p * reward_scales["clf_scale"]

    # change in prob reward

    mask = mean_p != 0
    rewards["d_clf"] += torch.where(
        mask & mask.roll(1, dims=1),
        (
            mean_p.diff(dim=1, prepend=mean_p[:, :1].clone())
            / action_times.diff(
                dim=1,
                prepend=torch.ones((bs, 1), device=action_times.device) * float("inf"),
            )
        ).clamp(max=0)
        * reward_scales["d_clf_scale"],
        0,
    )

    return rewards


def rollout(
    obs: nn.Module,
    critic: nn.Module | None,
    disc: nn.Module,
    X: dict[Feats, torch.Tensor],
    y: torch.Tensor,
    disc_league: list[tuple[int, nn.Module.state_dict]],
    disc_features: FeatureTrs | None = None,
    detach_period: int = 20,
    reward_scales: dict[str, float] | None = None,
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
    values = None
    device = y.device

    if X[Feats.TIMES].isnan().any():
        raise ValueError("NaN in times feature")

    fd = get_window_feature_dict(
        X, obs.time_step, obs.max_silence_s, features=obs.features, extend_end_s=2
    )
    if X[Feats.TIMES].isnan().any():
        raise ValueError("NaN in times feature")

    # We need the seq. lens in forward.
    action_seq_lens = fd.pop(Feats.SEQ_LENS)
    L = action_seq_lens.max().item()

    act_times, actions, log_ps, sel_probs, values_actor, entropies, h = obs.act(
        fd, hobs, h_detach_period=detach_period, seq_lens=action_seq_lens
    )

    if h is not None:
        h_norm = h[0].norm(2, dim=-1).max().item()
        c_norm = h[1].norm(2, dim=-1).max().item()
        if h_norm > 100 or c_norm > 400:
            print("Huge hidden/cell:", h_norm, c_norm)

    Xobs = send_exec(X, act_times, actions)

    if reward_scales is not None:
        current_disc_state = {
            k: v.detach().clone() for k, v in disc.state_dict().items()
        }
        rewards_l = []
        packet_seq_lens_gpu = (Xobs[Feats.DIRS] != 0).sum(dim=1).long()

        if disc_features is not None:
            X_ = disc_features.transform_batch(Xobs)
        else:
            X_ = Xobs

        disc_seq_lens = disc.seq_len_fun(X_)

        # Critic feat building:
        if critic is not None and Feats.LABEL in critic.features:
            fd[Feats.LABEL] = y.unsqueeze(1).repeat(1, L)

        for _, state_d in disc_league:
            if state_d is None:
                disc.load_state_dict(current_disc_state)
            else:
                disc.load_state_dict({k: v.to(device) for k, v in state_d.items()})

            hdisc = None
            disc.eval()
            with torch.no_grad():
                logits, hdisc = disc.pack_and_forward(X_, hdisc, disc_seq_lens)

            rewards_ = get_rewards(
                act_times,
                Xobs,
                y,
                logits,
                packet_seq_lens_gpu,
                reward_scales=reward_scales,
            )

            rewards_l.append(rewards_)

        # Rewards from different disc. checkpoints.
        rewards = {
            k: torch.stack([r[k] for r in rewards_l], dim=0)
            for k in rewards_l[0].keys()
        }

        # Make the values tensor
        if critic is not None:
            values = critic(
                fd, None, h_detach_period=detach_period, seq_lens=action_seq_lens
            )[0][Feats.STATE_VALUE]
        else:
            values = values_actor

        # Make sure correct state is restored.
        disc.load_state_dict(current_disc_state)

    return (
        log_ps,
        sel_probs,
        values,
        rewards,
        entropies,
        act_times,
        actions,
        Xobs,
        fd,
    )
