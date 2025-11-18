import time

import numpy as np
import torch
from torch import nn

from kipl_ml.rl.enums import Actions
from kipl_ml.rl.utils import PacketBuffer, step_actions
from kipl_ml.trace.enums import Feats


def get_reward(
    actions: torch.Tensor,
    ackts2idxs: dict[Actions, int],
    curXobs: dict[Feats, torch.Tensor],
    idxs: torch.Tensor,
    y: torch.Tensor,
    disc: nn.Module,
    buffer: PacketBuffer,
    hdisc: torch.Tensor | None,
) -> torch.Tensor:
    rewards = torch.zeros(actions.shape[0], device=actions.device)
    wait_idx = ackts2idxs[Actions.WAIT]
    with torch.no_grad():
        mask = actions != wait_idx

        if mask.any():
            logits, _ = disc(curXobs, hdisc)
            probs = torch.nn.functional.softmax(logits, dim=-1).squeeze(1)

            # When discriminator is able to predict the correct label
            if mask.sum() > 0:
                tp_cl_probs = probs.gather(1, y[mask].unsqueeze(1)).squeeze(1)
                rewards[mask] -= tp_cl_probs

        buffer_counts = buffer.bcounts()

        # When you delay the buffer
        delay_mask = (buffer_counts != 0) & (actions == ackts2idxs[Actions.WAIT])
        rewards[delay_mask] += -0.05 * buffer_counts[delay_mask]

        # When you send padding
        padding_mask = actions == ackts2idxs[Actions.SEND_PADDING_UP]
        rewards[padding_mask] += -0.1

        padding_mask = actions == ackts2idxs[Actions.SEND_PADDING_DOWN]
        rewards[padding_mask] += -0.1

    return rewards


def rollout(
    obs: nn.Module, disc: nn.Module, X: dict[Feats, torch.Tensor], y: torch.Tensor
) -> tuple[dict[Feats, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
    hobs = None
    hdisc = None
    idxs = None

    T = 5
    dt = 0.002
    t = 0.0
    buffer = PacketBuffer(dt)

    if set(X.keys()) != {Feats.TIMES, Feats.DIRS}:
        raise ValueError("Single set of features accepted!")

    bs = X[Feats.DIRS].shape[0]
    device = X[Feats.DIRS].device

    log_ps = []
    values = []
    rewards = []

    curXobs = {
        Feats.DIRS: torch.zeros((bs, 1), device=device).float(),
        Feats.TIMES: torch.zeros((bs, 1), device=device).float(),
        Feats.PADDING: torch.zeros((bs, 1), device=device).bool(),
    }

    idx2ackts = obs.ACTIONS
    ackts2idxs = {a.item(): i for i, a in enumerate(obs.ACTIONS)}
    actions = []
    times = []
    timings = []
    while t < T:
        t0 = time.time()
        buffer.step(X)
        t1 = time.time()

        actions_, log_ps_, values_, hobs = obs.act(buffer.get_feats(), hobs)
        t2 = time.time()

        log_ps.append(log_ps_.unsqueeze(1))
        values.append(values_.unsqueeze(1))
        actions.append(actions_.unsqueeze(1))
        times.append(t)

        curXobs, buffer = step_actions(actions_, idx2ackts, curXobs, buffer, t)
        t3 = time.time()

        rewards_ = get_reward(
            actions_, ackts2idxs, curXobs, idxs, y, disc, buffer, hdisc
        )
        t4 = time.time()
        rewards.append(rewards_.unsqueeze(1))

        t += dt

        timings.append([[t1 - t0, t2 - t1, t3 - t2, t4 - t3]])

    timings = np.concat(timings, axis=0).mean(axis=0)

    print(timings / timings.sum())

    log_ps = torch.cat(log_ps, dim=1)
    values = torch.cat(values, dim=1)
    actions = torch.cat(actions, dim=1)
    times = torch.tensor(times)
    rewards = torch.cat(rewards, dim=1)

    return curXobs, log_ps, values, rewards, actions, times
