import numpy as np
import torch
from torch import nn

from kipl_ml.data.utils import DOWNLOAD, UPLOAD
from kipl_ml.models.trgen import AGENT1
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
    actions: dict[Feats, torch.Tensor],
    Xobs: dict[Feats, torch.Tensor],
    buffer: dict[Feats, torch.Tensor],
    t: float,
    idxs: torch.Tensor | None = None,
) -> tuple[dict[Feats, torch.Tensor], dict[Feats, torch.Tensor], torch.Tensor]:
    idxs = idxs if idxs is not None else torch.zeros(Xobs[Feats.DIRS].shape[0]).long()

    def _append_buffer(buffer, direction, action_mask):
        buffer = buffer.squeeze()
        mb = buffer.max().floor().int().item()
        for b in range(mb):
            buffer_mask = (buffer > b).numpy()
            mask = buffer_mask & action_mask

            Xobs[Feats.DIRS][mask, idxs[mask]] = direction
            idxs[mask] += 1
        buffer[action_mask] = 0
        Xobs[Feats.TIMES][action_mask, idxs[action_mask] - 1] = t

    for a in AGENT1.ACTIONS:
        mask = actions == a

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

    return Xobs, buffer, idxs


def get_reward(
    action: np.ndarray[Actions],
    Xobs: dict[Feats, torch.Tensor],
    idxs: torch.Tensor,
    y: torch.Tensor,
    disc: nn.Module,
    hdisc: torch.Tensor | None,
    buffer: dict[Feats, torch.Tensor] | None = None,
) -> torch.Tensor:
    rewards = torch.zeros(action.shape[0])
    with torch.no_grad():
        mask = action != Actions.WAIT
        Xobs_ = {f: v.gather(1, idxs[mask].unsqueeze(1) - 1) for f, v in Xobs.items()}
        logits, _ = disc(Xobs_, hdisc)
        probs = torch.nn.functional.softmax(logits).squeeze()

        # When discriminator is able to predict the correct label
        rewards[mask] += -probs.gather(1, y[mask].unsqueeze(1)).squeeze()

        non_zero_buffer_mask = (
            (buffer[Feats.UP_BUFFER].squeeze() != 0)
            | (buffer[Feats.DOWN_BUFFER].squeeze() != 0)
        ).numpy()

        # When you delay the buffer
        delay_mask = non_zero_buffer_mask & (action == Actions.WAIT)
        rewards[delay_mask] += -0.1

        # When you send padding
        padding_mask = action == Actions.SEND_PADDING_UP
        rewards[padding_mask] += -0.05

        padding_mask = action == Actions.SEND_PADDING_DOWN
        rewards[padding_mask] += -0.05

    return rewards


def sim(obs: nn.Module, disc: nn.Module, X: dict[Feats, torch.Tensor], y: torch.Tensor):
    hobs = None
    hdisc = None
    idxs = None
    buffer = None
    T = 60
    dt = 0.01
    t = 0.0

    bs = X[Feats.DIRS].shape[0]
    NT = int(T // dt) + 1

    Xobs = {
        Feats.DIRS: torch.zeros((bs, NT)).float(),
        Feats.TIMES: torch.zeros((bs, NT)).float(),
    }
    while t < T:
        buffer = update_buffer(X, t, dt, buffer)

        action, log_ps, values, hobs = obs.act(buffer, hobs)

        Xobs, buffer, idxs = consume(action, Xobs, buffer, t, idxs)

        rewards = get_reward(action, Xobs, idxs, y, disc, hdisc, buffer)
        t += dt

    # Here we return log_ps, values, and rewards and further build returns, advantages, etc.
