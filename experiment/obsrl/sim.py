import torch
from torch import nn

from kipl_ml.data.utils import DOWNLOAD, UPLOAD
from kipl_ml.trace.enums import Feats


def get_slice(
    X: dict[Feats, torch.Tensor], t: float, dt: float
) -> dict[Feats, torch.Tensor]:
    times = X[Feats.TIMES]
    mask = (t <= times) & (times < t + dt)

    up = (mask & (X[Feats.DIRS] == UPLOAD)).sum(dim=1, keepdim=True).float()
    down = (mask & (X[Feats.DIRS] == DOWNLOAD)).sum(dim=1, keepdim=True).float()

    buffer = {Feats.UP_BUFFER: up, Feats.DOWN_BUFFER: down}

    return buffer


def consume(
    send: dict[Feats, torch.Tensor],
    Xobs: dict[Feats, torch.Tensor],
    idxs: torch.Tensor | None = None,
) -> tuple[dict[Feats, torch.Tensor], torch.Tensor]:
    idxs = idxs if idxs is not None else torch.zeros(Xobs[Feats.DIRS].shape[0]).long()

    def _append_buffer(buffer, direction):
        buffer = buffer.squeeze()
        mb = buffer.max().floor().int().item()
        for b in range(mb):
            mask = buffer > b

            Xobs[Feats.DIRS][mask, idxs[mask]] = direction
            idxs[mask] += 1

    _append_buffer(send[Feats.UP_BUFFER], UPLOAD)
    _append_buffer(send[Feats.DOWN_BUFFER], DOWNLOAD)

    return Xobs, idxs


def sim(obs: nn.Module, disc: nn.Module, X: dict[Feats, torch.Tensor]):
    hobs = None
    idxs = None
    T = 60
    dt = 0.01
    t = 0.0

    Xobs = {Feats.DIRS: torch.zeros((X[Feats.DIRS].shape[0], int(T // dt))).float()}
    with torch.no_grad():
        while t < T:
            buffer = get_slice(X, t, dt)

            send, hobs = obs(buffer, hobs)

            Xobs, idxs = consume(send, Xobs, idxs)

            t += dt
            print(t)

    logits, _ = disc(Xobs)
