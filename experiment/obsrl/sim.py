import torch
from torch import nn

from kipl_ml.rl.action import ActionsExec
from kipl_ml.rl.enums import Actions
from kipl_ml.rl.observation import BaseTraceObservation
from kipl_ml.rl.utils import _flush_left
from kipl_ml.trace.enums import Feats


def _unpack_xobs_l(
    xobs_l: list[dict[Feats, torch.Tensor]],
) -> dict[Feats, torch.Tensor]:
    Xobs: dict[Feats, torch.Tensor] = {}
    for k in xobs_l[0].keys():
        vals_ = torch.cat([v[k] for v in xobs_l], dim=1)
        Xobs[k] = vals_

    mask = Xobs[Feats.DIRS] != 0
    max_len = mask.sum(dim=1).max().item()

    for k, v in Xobs.items():
        Xobs[k] = _flush_left(v, mask)[:, :max_len]

    Xobs[Feats.PADDING] = Xobs[Feats.PADDING].bool()

    return Xobs


@torch.no_grad()
def get_reward(
    disc: nn.Module,
    hdisc: tuple[torch.Tensor, ...],
    y: torch.Tensor,
    actions: torch.Tensor,
    Xobs: dict[Feats, torch.Tensor],
    padding_scale: float = 1,
    clf_scale: float = 10,
) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
    rewards = torch.zeros(len(y), device=y.device)

    packet_counts = (Xobs[Feats.DIRS] != 0).sum(dim=1)
    padding_counts = (Xobs[Feats.PADDING] != 0).sum(dim=1)

    # Padding cost
    if padding_counts.any():
        has_padding = padding_counts != 0
        rewards[has_padding] -= padding_counts[has_padding].float() * padding_scale

    # Classification reward
    if packet_counts.any():
        has_action = packet_counts != 0
        logits, hdisc = disc.pack_and_forward(Xobs, hdisc, packet_counts.cpu())

        probs = nn.functional.softmax(logits, dim=1)

        # (A, )
        cl_probs = probs.gather(1, y[has_action].unsqueeze(1)).squeeze(1)

        # Small correct cl prob --> large reward
        rewards[has_action] += clf_scale * (0.1 - cl_probs)

    return rewards, hdisc


def rollout(
    obs: nn.Module,
    disc: nn.Module,
    X: dict[Feats, torch.Tensor],
    y: torch.Tensor,
    dt: float = 0.01,
    maxT: float = 10,
    detach_every_delta_t: float = 1,
    clf_scale: float = 1.0,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    list[dict[Actions, torch.Tensor]],
    dict[Feats, torch.Tensor],
]:
    device = X[Feats.DIRS].device
    # Dummy run to get init hdisc...
    hdisc = None
    hobs = None

    log_ps_l = []
    values_l = []
    rewards_l = []
    entropy_l = []
    actions_l = []
    times_l = []
    xobs_l = []
    timings = []

    i = 0
    detach_every = detach_every_delta_t // dt

    hidden_penalty = torch.zeros(2, device=device)
    ackt_exec = ActionsExec(dt)
    bto = BaseTraceObservation(X)

    while True:
        # Step BaseTraceObservation
        bto.step(dt)

        # Step obsf.action(base_trace_obs, hobs)
        actions_, log_ps_, values_, entropies_, hobs = obs.act(bto.feature_dict, hobs)

        hidden_penalty[0] = hidden_penalty[0] + hobs[0].pow(2).mean()
        hidden_penalty[1] = hidden_penalty[1] + hobs[1].pow(2).mean()

        # Step Various Action execs
        curXobs = ackt_exec.step(actions_, bto.feature_dict)
        xobs_l.append(curXobs)

        # Get rewards
        rewards_, hdisc = get_reward(
            disc, hdisc, y, actions_, curXobs, padding_scale=1, clf_scale=clf_scale
        )

        actions_l.append(actions_)
        log_ps_l.append(log_ps_.unsqueeze(1))
        values_l.append(values_.unsqueeze(1))
        entropy_l.append(entropies_.unsqueeze(1))
        rewards_l.append(rewards_.unsqueeze(1))
        times_l.append(bto.t)

        if bto.t > maxT:
            break

        if (i != 0) and (i % detach_every == 0):
            hobs = tuple(h_.detach() for h_ in hobs)

        i += 1

    # print(timings / timings.sum())

    hidden_penalty /= i
    log_ps: torch.Tensor = torch.cat(log_ps_l, dim=1)
    values: torch.Tensor = torch.cat(values_l, dim=1)
    entropies: torch.Tensor = torch.cat(entropy_l, dim=1)
    times: torch.Tensor = torch.tensor(times_l)
    rewards: torch.Tensor = torch.cat(rewards_l, dim=1)

    Xobs = _unpack_xobs_l(xobs_l)

    return (
        log_ps,
        values,
        rewards,
        entropies,
        hidden_penalty,
        times,
        actions_l,
        Xobs,
    )
