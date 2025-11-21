import time

import numpy as np
import torch
from torch import nn

from kipl_ml.rl.utils import PacketBuffer, TraceObservation, step_actions
from kipl_ml.trace.enums import Feats


def _hidden_w_mask(
    h: tuple[torch.Tensor, ...],
    mask: torch.Tensor,
    hmasked: tuple[torch.Tensor, ...] | None = None,
) -> tuple[torch.Tensor, ...]:
    if hmasked is None:
        return tuple(h_[:, mask] for h_ in h)

    for i, h_ in enumerate(h):
        h_[:, mask] = hmasked[i]

    return h


def get_reward(
    actions: torch.Tensor,
    curXobs: TraceObservation,
    y: torch.Tensor,
    disc: nn.Module,
    buffer: PacketBuffer,
    hdisc: tuple[torch.Tensor, ...],
    sc: float = 1.0,
) -> torch.Tensor:
    rewards = torch.zeros(actions.shape[0], device=actions.device)
    buffer_counts = buffer.bcounts
    buffer_times = buffer.btimes
    with torch.no_grad():
        while not curXobs.is_waiting.all():
            obs, is_waiting = curXobs.pop_oldest()

            has_action = ~is_waiting  # has_action.sum() = A
            has_buffer = buffer.bcounts != 0
            sends_padding = obs[Feats.PADDING].squeeze(1).bool()

            if has_action.any():
                hdisc_ = _hidden_w_mask(hdisc, has_action)
                curXnotwaiting = {k: v[has_action] for k, v in obs.items()}
                logits, hdisc_ = disc(curXnotwaiting, hdisc_)

                hdisc = _hidden_w_mask(hdisc, has_action, hdisc_)

                # (A, n_actions)
                probs = torch.nn.functional.softmax(logits, dim=-1).squeeze(1)

                # Punish for discriminator high prop:

                # (A, )
                cl_probs = probs.gather(1, y[has_action].unsqueeze(1)).squeeze(1)

                # High correct cl prob --> small reward
                rewards[has_action] += (1 - cl_probs) * sc
            else:
                # When you delay the buffer
                delayed_buffer_mask = has_buffer & is_waiting
                rewards[delayed_buffer_mask] -= (
                    10.0
                    * sc
                    * buffer_times[delayed_buffer_mask]
                    * buffer_counts[delayed_buffer_mask]
                )

            # When you send padding while having buffer
            rewards[sends_padding & has_buffer] -= 10 * sc

            # When you send padding w. empty buffer
            rewards[sends_padding & ~has_buffer] -= 1 * sc

    return rewards, hdisc


def rollout(
    obs: nn.Module,
    disc: nn.Module,
    X: dict[Feats, torch.Tensor],
    y: torch.Tensor,
    dt: float = 0.01,
) -> tuple[dict[Feats, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
    hobs = None

    T = 1
    buffer = PacketBuffer(dt)

    if set(X.keys()) != {Feats.TIMES, Feats.DIRS}:
        raise ValueError("Single set of features accepted!")

    bs = X[Feats.DIRS].shape[0]
    device = X[Feats.DIRS].device

    log_ps = []
    values = []
    rewards = []

    curXobs = TraceObservation(B=bs, L=100)

    # Dummy run to get init hdisc...
    _, hdisc = disc(curXobs.pop_oldest()[0], None)

    idx2ackts = obs.action_map
    packet_counts = torch.zeros(bs, device=device)
    actions = []
    times = []
    timings = []
    Xobs_l = []
    while buffer.t < T:
        t0 = time.time()
        buffer.step(X)
        t1 = time.time()

        actions_, log_ps_, values_, hobs = obs.act(buffer.feature_dict, hobs)
        t2 = time.time()

        log_ps.append(log_ps_.unsqueeze(1))
        values.append(values_.unsqueeze(1))
        actions.append(actions_.unsqueeze(1))
        times.append(buffer.t)

        curXobs, buffer = step_actions(actions_, idx2ackts, curXobs, buffer, buffer.t)
        t3 = time.time()

        Xobs_l.append(curXobs.X.clone())
        packet_counts += ((curXobs.X[..., 0] == 1) & (curXobs.X[..., 2] == 2)).sum(
            dim=1
        )
        t4 = time.time()

        rewards_, hdisc = get_reward(actions_, curXobs, y, disc, buffer, hdisc)

        t5 = time.time()
        rewards.append(rewards_.unsqueeze(1))

        timings.append([[t1 - t0, t2 - t1, t3 - t2, t4 - t3, t5 - t4]])

    timings = np.concat(timings, axis=0).mean(axis=0)

    # print(timings / timings.sum())

    log_ps = torch.cat(log_ps, dim=1)
    values = torch.cat(values, dim=1)
    actions = torch.cat(actions, dim=1)
    times = torch.tensor(times)
    rewards = torch.cat(rewards, dim=1)

    # Append the observations:
    Xobs = torch.cat(Xobs_l, dim=1)

    Lmax = Xobs[..., 0].ne(0).sum(dim=1).max()

    Xobsd = {}
    for f, i in zip((Feats.DIRS, Feats.TIMES, Feats.PADDING), range(3)):
        mask = Xobs[..., i] != 0
        lens = mask.sum(dim=1)

        idxs = torch.arange(Lmax, device=device).unsqueeze(0).expand(bs, -1)

        new_mask = idxs < lens.unsqueeze(1)

        # (B, Lmax, 3)
        Xobs_ = torch.zeros((bs, Lmax), device=device)
        Xobs_[new_mask] = Xobs[mask, i]

        Xobsd[f] = Xobs_

    Xobsd[Feats.PADDING] = Xobsd[Feats.PADDING] == 1

    if not (
        ((Xobsd[Feats.DIRS] == 1) & (~Xobsd[Feats.PADDING])).sum(dim=1) == packet_counts
    ).all():
        print(((Xobsd[Feats.DIRS] == 1) & (~Xobsd[Feats.PADDING])).sum(dim=1))
        print(packet_counts)
        breakpoint()

    nmissing = (X[Feats.DIRS] == 1).sum(dim=1) - packet_counts

    if (nmissing < 0).any():
        print(nmissing)
        breakpoint()

    return Xobsd, log_ps, values, rewards, actions, times
