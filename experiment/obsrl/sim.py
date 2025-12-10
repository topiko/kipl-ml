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

    for i in range(action_times.shape[1] - 1):
        # (B, 1)
        t0 = action_times[:, i].unsqueeze(1)
        t1 = action_times[:, i + 1].unsqueeze(1)

        len_mask = (seq_lens - 1) <= i
        if len_mask.any():
            t1[seq_lens - 1 <= i] = torch.inf
            t0[seq_lens - 1 < i] = torch.inf

        # From the last action we take rewards all the way to end of times.
        mask = (times >= t0) & (times < t1)

        # (B, )
        npad = (padding & mask).sum(dim=1).float()
        rewards["padding"][:, i] -= npad * reward_scales["padding_scale"]

        # Count the "clf reward" only from the normal packets.
        float_mltp = (mask & ~padding).float()
        nnormal = float_mltp.sum(dim=1)
        normal_mask = nnormal > 0
        mean_p = torch.where(
            nnormal > 0,
            (target_probs * (mask & ~padding).float()).sum(dim=1) / nnormal,
            0,
        )

        rewards["clf"][normal_mask, i] += (0.1 - mean_p[normal_mask]) * reward_scales[
            "clf_scale"
        ]

    return rewards


def rollout(
    obs: nn.Module,
    disc: nn.Module,
    X: dict[Feats, torch.Tensor],
    y: torch.Tensor,
    dt: float = 0.01,
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

    features = [Feats.DOWN_COUNT, Feats.UP_COUNT, Feats.Dt]
    fd = get_window_feature_dict(X, dt, 100.0, features=features)

    t0 = time.time()

    bs, L = fd[Feats.Dt].shape

    # Due to different seq. lens, run each seq. separately.
    act_times = torch.zeros((bs, L), device=fd[Feats.Dt].device)
    log_ps = torch.zeros_like(act_times)
    values = torch.zeros_like(act_times)
    entropies = torch.zeros_like(act_times)
    actions: dict[Actions, torch.Tensor] = {}
    seq_lens = torch.zeros((bs,), dtype=torch.long)
    for i in range(bs):
        padc = (fd[Feats.Dt][i] == 0).sum()
        L_ = L - padc
        fd_ = {k: v[i : i + 1, :L_] for k, v in fd.items()}

        act_times_, actions_, log_ps_, values_, entropies_, _ = obs.act(
            fd_, hobs, h_detach_period=detach_period
        )

        act_times[i, :L_] = act_times_
        for k, v in actions_.items():
            if k not in actions:
                actions[k] = torch.zeros_like(act_times)
            actions[k][i, :L_] = v
        log_ps[i, :L_] = log_ps_
        values[i, :L_] = values_
        entropies[i, :L_] = entropies_
        seq_lens[i] = L_

    t1 = time.time()
    Xobs = send_exec(X, act_times, actions)

    t2 = time.time()
    if reward_scales is not None:
        with torch.no_grad():
            logits, hdisc = disc(Xobs, hdisc)

        t3 = time.time()

        rewards = get_rewards(
            act_times, Xobs, y, logits, seq_lens, reward_scales=reward_scales
        )
    t4 = time.time()

    # print(
    #     f"Obs: {t1 - t0:.4f}, Act: {t2 - t1:.4f}, Disc: {t3 - t2:.4f}, Rew: {t4 - t3:.4f}"
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
