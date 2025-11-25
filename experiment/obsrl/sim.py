import time

import numpy as np
import torch
from torch import nn

from kipl_ml.rl.utils import PacketBuffer, TraceObservation, step_actions
from kipl_ml.trace.enums import Feats


def get_reward(
    actions: torch.Tensor,
    Xobs: TraceObservation,
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
        delayed_buffer_mask = has_buffer & Xobs.is_waiting
        rewards[delayed_buffer_mask] -= (
            10.0
            * sc
            * buffer_times[delayed_buffer_mask]
            * buffer_counts[delayed_buffer_mask]
        )

        # Send counts:
        count_padding = Xobs.count_padding

        # When you send padding while having buffer
        rewards[has_buffer] -= 10 * sc * count_padding[has_buffer]

        # When you send padding w. empty buffer
        rewards[~has_buffer] -= 1 * sc * count_padding[~has_buffer]

        count_packets = Xobs.count_send

        if count_packets.any():
            has_action = count_packets != 0
            logits, hdisc = disc.pack_and_forward(
                Xobs.feature_dict, hdisc, count_packets.cpu()
            )

            probs = nn.functional.softmax(logits, dim=1)
            # (A, )
            cl_probs = probs.gather(1, y[has_action].unsqueeze(1)).squeeze(1)

            # High correct cl prob --> small reward
            rewards[has_action] += (1 - cl_probs) * sc

    return rewards, hdisc


def rollout(
    obs: nn.Module,
    disc: nn.Module,
    X: dict[Feats, torch.Tensor],
    y: torch.Tensor,
    dt: float = 0.01,
    maxT: float = 10,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    dict[Feats, torch.Tensor],
    dict[Feats, torch.Tensor],
    torch.Tensor,
    torch.Tensor,
]:
    hobs = None

    buffer = PacketBuffer(dt)

    if set(X.keys()) != {Feats.TIMES, Feats.DIRS}:
        raise ValueError("Single set of features accepted!")

    bs = X[Feats.DIRS].shape[0]
    device = X[Feats.DIRS].device

    Xobs = TraceObservation(B=bs, L=1024, device=device)

    # Dummy run to get init hdisc...
    hdisc = (
        torch.randn((disc.rnn.num_layers, bs, disc.rnn.hidden_size), device=device),
        torch.randn((disc.rnn.num_layers, bs, disc.rnn.hidden_size), device=device),
    )

    idx2ackts = obs.action_map
    packet_counts = torch.zeros(bs, device=device)
    log_ps_l = []
    values_l = []
    rewards_l = []
    actions_l = []
    times_l = []
    timings = []
    while buffer.t < maxT:
        t0 = time.time()
        buffer.step(X)
        t1 = time.time()

        actions_, log_ps_, values_, hobs = obs.act(buffer.feature_dict, hobs)
        t2 = time.time()

        log_ps_l.append(log_ps_.unsqueeze(1))
        values_l.append(values_.unsqueeze(1))
        actions_l.append(actions_.unsqueeze(1))
        times_l.append(buffer.t)

        Xobs, buffer = step_actions(actions_, idx2ackts, Xobs, buffer, buffer.t)
        t3 = time.time()

        Xobs.append()
        buffer.append()

        packet_counts += ((Xobs.X[..., 0] == 1) & (Xobs.X[..., 2] == 2)).sum(dim=1)
        t4 = time.time()

        rewards_, hdisc = get_reward(actions_, Xobs, y, disc, buffer, hdisc)

        Xobs.reset()

        t5 = time.time()
        rewards_l.append(rewards_.unsqueeze(1))

        timings.append([[t1 - t0, t2 - t1, t3 - t2, t4 - t3, t5 - t4]])

    timings = np.concat(timings, axis=0).mean(axis=0)

    # print(timings / timings.sum())

    log_ps: torch.Tensor = torch.cat(log_ps_l, dim=1)
    values: torch.Tensor = torch.cat(values_l, dim=1)
    actions: torch.Tensor = torch.cat(actions_l, dim=1)
    times: torch.Tensor = torch.tensor(times_l)
    rewards: torch.Tensor = torch.cat(rewards_l, dim=1)

    Xobsd = Xobs.history

    breakpoint()

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

    return log_ps, values, rewards, Xobsd, buffer.history, times, actions
