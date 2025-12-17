import time

import torch
from torch import nn

from kipl_ml.rl.action import send_exec
from kipl_ml.rl.enums import Actions
from kipl_ml.rl.observation import get_window_feature_dict
from kipl_ml.trace.enums import Feats


def get_rewards(
    action_times: torch.Tensor,
    X: dict[Feats, torch.Tensor],
    y: torch.Tensor,
    disc_logits: torch.Tensor,
    seq_lens: torch.Tensor,
    reward_scales: dict[str, float] = {"clf_scale": 1.0, "padding_scale": 1.0},
) -> dict[str, torch.Tensor]:
    N = disc_logits.shape[1]

    bs, T = action_times.shape

    # (B, T)
    rewards: dict[str, torch.Tensor] = {
        k.replace("_scale", ""): torch.zeros_like(action_times) for k in reward_scales
    }

    # (B, N)
    times = X[Feats.TIMES]
    padding = X[Feats.PADDING].bool()

    # (B, N, C)
    probs = nn.functional.softmax(disc_logits, dim=-1)

    # (B, N)
    target_probs = probs.gather(2, y.unsqueeze(1).expand(-1, N).unsqueeze(-1)).squeeze(
        -1
    )

    for i in range(T):
        # (B, 1)
        t0 = action_times[:, i].unsqueeze(1)

        # t1 is now beyond last step, e.g., seq_len = 10, i = 8, i + 1 = 9,
        # and when i = 9, i + 1 = 10 -> t1 = inf
        len_mask = (seq_lens - 1) < i + 1

        if i < T - 1:
            t1 = action_times[:, i + 1].unsqueeze(1)
        else:
            t1 = torch.ones_like(t0) * torch.inf

        if len_mask.any():
            t1[len_mask, :] = torch.inf

            # t0 is now at last step or beyond
            t0_beyond_last = (seq_lens - 1) < i
            t0[t0_beyond_last, :] = torch.inf

        # From the last action we take rewards all the way to end of times.
        mask = (times >= t0) & (times < t1)

        # (B, )
        npad = (padding & mask).sum(dim=1).float()
        rewards["padding"][:, i] -= npad * reward_scales["padding_scale"]

        # Count the "clf reward" only from the normal packets.
        p_lvl = 0.1
        float_mltp = (mask & ~padding).float()
        nnormal = float_mltp.sum(dim=1)
        mean_p = torch.where(
            nnormal > 0,
            ((p_lvl - target_probs) * (mask & ~padding).float()).sum(dim=1) / nnormal,
            0,
        )

        #
        normal_mask = (nnormal > 0) & ~len_mask
        rewards["clf"][normal_mask, i] += (
            mean_p[normal_mask] * reward_scales["clf_scale"]
        )

    return rewards


def rollout(
    obs: nn.Module,
    disc: nn.Module,
    X: dict[Feats, torch.Tensor],
    y: torch.Tensor,
    detach_period: int = 20,
    reward_scales: dict[str, float] | None = None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    dict[str, torch.Tensor] | None,
    torch.Tensor,
    torch.Tensor,
    dict[Actions, torch.Tensor],
    dict[Feats, torch.Tensor],
]:
    hdisc = None
    hobs = None

    rewards = None

    t0 = time.time()
    fd = get_window_feature_dict(
        X, obs.time_step, obs.max_silence_s, features=obs.features
    )

    t1 = time.time()

    bs, _ = fd[Feats.Dt].shape

    # We need the seq. lens in forward.
    seq_lens = (fd[Feats.Dt] != 0).sum(dim=1) + 1

    act_times, actions, log_ps, values, entropies, h = obs.act(
        fd, hobs, h_detach_period=detach_period, seq_lens=seq_lens
    )

    if h is not None:
        h_norm = h[0].norm(2, dim=-1).max().item()
        c_norm = h[1].norm(2, dim=-1).max().item()
        if h_norm > 100 or c_norm > 10000:
            print("Huge hidden/cell:", h_norm, c_norm)

    t2 = time.time()
    Xobs = send_exec(X, act_times, actions)

    t3 = time.time()
    if reward_scales is not None:
        disc.eval()
        with torch.no_grad():
            logits, hdisc = disc(Xobs, hdisc)

        t4 = time.time()
        rewards = get_rewards(
            act_times, Xobs, y, logits, seq_lens, reward_scales=reward_scales
        )
    t5 = time.time()

    # print()
    # print(
    #     f"Extr: {t1 - t0:.4f}, Act: {t2 - t1:.4f}, Exec: {t3 - t2:.4f}, Disc: {t4 - t3:.4f}, Rew: {t5 - t4:.4f}"
    # )

    return (
        log_ps,
        values,
        rewards,
        entropies,
        act_times,
        actions,
        Xobs,
    )
