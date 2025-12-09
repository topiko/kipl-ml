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
    reward_scales: dict[str, float] = {"clf_scale": 1.0, "padding_scale": 1.0},
) -> torch.Tensor:
    N = disc_logits.shape[1]

    # (B, T)
    rewards = torch.zeros_like(action_times)

    # (B, N)
    times = X[Feats.TIMES]
    padding = X[Feats.PADDING]

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
        mask = ((times >= t0) & (times < t1)).float()

        # (B, )
        npad = (padding * mask).sum(dim=1)

        mean_p = (target_probs * (mask - padding).clip(0, 1)).mean(dim=1)

        rewards[:, i] -= npad * reward_scales["padding_scale"]
        rewards[:, i] += (1 - mean_p) * reward_scales["clf_scale"]

    return rewards


def rollout(
    obs: nn.Module,
    disc: nn.Module,
    X: dict[Feats, torch.Tensor],
    y: torch.Tensor,
    dt: float = 0.01,
    detach_period: int = 20,
    reward_scales: dict[str, float] | None = {"clf_scale": 1.0, "padding_scale": 1.0},
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor | None,
    torch.Tensor,
    torch.Tensor,
    dict[Actions, torch.Tensor],
    dict[Feats, torch.Tensor],
]:
    # Dummy run to get init hdisc...
    hdisc = None
    hobs = None

    rewards = None

    features = [Feats.DOWN_COUNT, Feats.UP_COUNT, Feats.Dt]
    fd = get_window_feature_dict(X, dt, 100.0, features=features)

    t0 = time.time()

    T = fd[features[0]].shape[1]

    act_times = []
    actions = []
    log_ps = []
    values = []
    entropies = []
    for i in range(T // detach_period + 1):
        fd_chunk = {
            k: v[:, i * detach_period : (i + 1) * detach_period] for k, v in fd.items()
        }
        act_times_, actions_, log_ps_, values_, entropies_, hobs = obs.act(
            fd_chunk, hobs
        )

        hobs = tuple(v.detach() for v in hobs)

        act_times.append(act_times_)
        actions.append(actions_)
        log_ps.append(log_ps_)
        values.append(values_)
        entropies.append(entropies_)

    act_times = torch.cat(act_times, dim=1)
    actions = {k: torch.cat([a[k] for a in actions], dim=1) for k in actions_.keys()}
    log_ps = torch.cat(log_ps, dim=1)
    values = torch.cat(values, dim=1)
    entropies = torch.cat(entropies, dim=1)

    t1 = time.time()
    Xobs = send_exec(X, act_times, actions)

    t2 = time.time()
    if reward_scales is not None:
        with torch.no_grad():
            logits, hdisc = disc(Xobs, hdisc)

        t3 = time.time()

        rewards = get_rewards(act_times, Xobs, y, logits, reward_scales=reward_scales)
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
