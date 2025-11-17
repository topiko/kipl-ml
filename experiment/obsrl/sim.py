import torch
from torch import nn

from kipl_ml.data.utils import DOWNLOAD, UPLOAD
from kipl_ml.rl.enums import Actions
from kipl_ml.trace.enums import Feats


def update_buffer(
    X: dict[Feats, torch.Tensor],
    t: float,
    dt: float,
    buffer: dict[Feats, torch.Tensor] | None = None,
) -> dict[Feats, torch.Tensor]:
    times = X[Feats.TIMES]
    mask = (t <= times) & (times < t + dt)

    up = (mask & (X[Feats.DIRS] == UPLOAD)).sum(dim=1, keepdim=True).float()
    down = (mask & (X[Feats.DIRS] == DOWNLOAD)).sum(dim=1, keepdim=True).float()

    if buffer is None:
        buffer = {
            Feats.UP_BUFFER: up,
            Feats.DOWN_BUFFER: down,
            Feats.TIMES: torch.ones_like(up) * t,
        }
    else:
        buffer[Feats.UP_BUFFER] += up
        buffer[Feats.DOWN_BUFFER] += down
        buffer[Feats.TIMES].fill_(t)

    return buffer


def consume(
    actions: torch.Tensor,
    idx2ackts: dict[int, Actions],
    Xobs: dict[Feats, torch.Tensor],
    buffer: dict[Feats, torch.Tensor],
    t: float,
    idxs: torch.Tensor | None = None,
) -> tuple[dict[Feats, torch.Tensor], dict[Feats, torch.Tensor], torch.Tensor]:
    idxs = idxs if idxs is not None else torch.zeros_like(actions).long()

    def _append_buffer(buffer, direction, action_mask):
        buffer = buffer.squeeze(1)
        mb = buffer.max().floor().int().item()
        for b in range(mb):
            buffer_mask = buffer > b
            mask = buffer_mask & action_mask

            Xobs[Feats.DIRS][mask, idxs[mask]] = direction
            idxs[mask] += 1

        buffer[action_mask] = 0
        Xobs[Feats.TIMES][action_mask, idxs[action_mask] - 1] = t

    for a in actions.unique():
        mask = actions == a
        a = idx2ackts[a.item()]

        match a:
            case Actions.SEND_PADDING_UP:
                Xobs[Feats.DIRS][mask, idxs[mask]] = UPLOAD
                Xobs[Feats.TIMES][mask, idxs[mask]] = t
                idxs[mask] += 1
            case Actions.SEND_PADDING_DOWN:
                Xobs[Feats.DIRS][mask, idxs[mask]] = DOWNLOAD
                Xobs[Feats.TIMES][mask, idxs[mask]] = t
                idxs[mask] += 1
            case Actions.SEND_BUFFER:
                _append_buffer(buffer[Feats.UP_BUFFER], UPLOAD, mask)
                _append_buffer(buffer[Feats.DOWN_BUFFER], DOWNLOAD, mask)
            case Actions.WAIT:
                pass
            case _:
                raise ValueError(f"Unknown action: {a}")

    return Xobs, buffer, idxs


def get_reward(
    actions: torch.Tensor,
    ackts2idxs: dict[Actions, int],
    Xobs: dict[Feats, torch.Tensor],
    idxs: torch.Tensor,
    y: torch.Tensor,
    disc: nn.Module,
    buffer: dict[Feats, torch.Tensor],
    hdisc: torch.Tensor | None,
) -> torch.Tensor:
    rewards = torch.zeros(actions.shape[0], device=actions.device)
    with torch.no_grad():
        mask = actions != ackts2idxs[Actions.WAIT]

        Xobs_ = {f: v.gather(1, idxs[mask].unsqueeze(1) - 1) for f, v in Xobs.items()}
        logits, _ = disc(Xobs_, hdisc)
        probs = torch.nn.functional.softmax(logits, dim=-1).squeeze(1)

        # When discriminator is able to predict the correct label
        if mask.sum() > 0:
            tp_cl_probs = probs.gather(1, y[mask].unsqueeze(1)).squeeze(1)
            rewards[mask] -= tp_cl_probs

        non_zero_buffer_mask = (buffer[Feats.UP_BUFFER].squeeze(1) != 0) | (
            buffer[Feats.DOWN_BUFFER].squeeze(1) != 0
        )

        # When you delay the buffer
        delay_mask = non_zero_buffer_mask & (actions == ackts2idxs[Actions.WAIT])
        rewards[delay_mask] += -0.05

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
    buffer = None
    T = 1
    dt = 0.01
    t = 0.0

    if set(X.keys()) != {Feats.TIMES, Feats.DIRS}:
        raise ValueError("Single set of features accepted!")

    bs = X[Feats.DIRS].shape[0]
    device = X[Feats.DIRS].device

    log_ps = []
    values = []
    rewards = []

    npackets = X[Feats.DIRS].shape[1]
    NT = int(T // dt) + 1 + npackets

    Xobs = {
        Feats.DIRS: torch.zeros((bs, NT), device=device).float(),
        Feats.TIMES: torch.zeros((bs, NT), device=device).float(),
    }

    idx2ackts = obs.ACTIONS
    ackts2idxs = {a: i for i, a in enumerate(obs.ACTIONS)}
    actions = []
    times = []

    while t < T:
        buffer = update_buffer(X, t, dt, buffer)

        action, log_ps_, values_, hobs = obs.act(buffer, hobs)
        log_ps.append(log_ps_.unsqueeze(1))
        values.append(values_.unsqueeze(1))
        actions.append(action.unsqueeze(1))
        times.append(t)

        Xobs, buffer, idxs = consume(action, idx2ackts, Xobs, buffer, t, idxs)

        rewards_ = get_reward(action, ackts2idxs, Xobs, idxs, y, disc, buffer, hdisc)

        rewards.append(rewards_.unsqueeze(1))

        t += dt

    log_ps = torch.cat(log_ps, dim=1)
    values = torch.cat(values, dim=1)
    actions = torch.cat(actions, dim=1)
    times = torch.tensor(times)
    rewards = torch.cat(rewards, dim=1)

    return Xobs, log_ps, values, rewards, actions, times
