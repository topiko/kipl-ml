import time

import numpy as np
import torch
from torch import nn

from kipl_ml.rl.utils import PacketBuffer, TraceObservation, step_actions
from kipl_ml.trace.enums import Feats


def get_reward(
    actions: torch.Tensor,
    curXobs: TraceObservation,
    y: torch.Tensor,
    disc: nn.Module,
    buffer: PacketBuffer,
    hdisc: tuple[torch.Tensor, ...],
    sc: float = 1.0,
) -> torch.Tensor:
    buffer_counts = buffer.bcounts
    buffer_times = buffer.btimes
    with torch.no_grad():
        rewards = torch.zeros(actions.shape[0], device=actions.device)
        has_buffer = buffer.bcounts != 0
        # When you delay the buffer
        delayed_buffer_mask = has_buffer & curXobs.is_waiting
        rewards[delayed_buffer_mask] -= (
            10.0
            * sc
            * buffer_times[delayed_buffer_mask]
            * buffer_counts[delayed_buffer_mask]
        )

        # Send counts:
        count_padding = curXobs.count_padding

        # When you send padding while having buffer
        rewards[has_buffer] -= 10 * sc * count_padding[has_buffer]

        # When you send padding w. empty buffer
        rewards[~has_buffer] -= 1 * sc * count_padding[~has_buffer]

        count_packets = curXobs.count_send

        if count_packets.any():
            has_action = count_packets != 0
            logits, hdisc = disc.pack_and_forward(
                curXobs.feature_dict, hdisc, count_packets.cpu()
            )

            probs = nn.functional.softmax(logits, dim=1)
            # (A, )
            cl_probs = probs.gather(1, y[has_action].unsqueeze(1)).squeeze(1)

            # High correct cl prob --> small reward
            rewards[has_action] += (1 - cl_probs) * sc

        curXobs.reset()

    return rewards, hdisc


def rollout(
    obs: nn.Module,
    disc: nn.Module,
    X: dict[Feats, torch.Tensor],
    y: torch.Tensor,
    dt: float = 0.01,
    maxT: float = 10,
) -> tuple[dict[Feats, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
    hobs = None

    buffer = PacketBuffer(dt)

    if set(X.keys()) != {Feats.TIMES, Feats.DIRS}:
        raise ValueError("Single set of features accepted!")

    bs = X[Feats.DIRS].shape[0]
    device = X[Feats.DIRS].device

    log_ps = []
    values = []
    rewards = []

    curXobs = TraceObservation(B=bs, L=1024, device=device)

    # Dummy run to get init hdisc...
    _, hdisc = disc(curXobs.pop_oldest()[0], None)

    idx2ackts = obs.action_map
    packet_counts = torch.zeros(bs, device=device)
    actions = []
    times = []
    timings = []
    Xobs_l = []
    while buffer.t < maxT:
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

    mask = Xobs[..., 0] != 0
    Lmax = mask.sum(dim=1).max()

    Xobsd = {}
    for f, i in zip((Feats.DIRS, Feats.TIMES, Feats.PADDING), range(3)):
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
