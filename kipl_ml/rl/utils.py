import torch

from kipl_ml.data.utils import DOWNLOAD, UPLOAD
from kipl_ml.rl.enums import Actions
from kipl_ml.trace.enums import Feats


class PacketBuffer:
    def __init__(self, dt: float):
        self.t = 0.0
        self.dt = dt
        self._buffer: dict[Feats, torch.Tensor] = {}

    @property
    def buffer(self) -> dict[Feats, torch.Tensor]:
        return self._buffer

    def step(self, X: dict[Feats, torch.Tensor]):
        times = X[Feats.TIMES]
        mask = (self.t <= times) & (times < self.t + self.dt)

        up = (mask & (X[Feats.DIRS] == UPLOAD)).sum(dim=1, keepdim=True).float()
        down = (mask & (X[Feats.DIRS] == DOWNLOAD)).sum(dim=1, keepdim=True).float()

        if self._buffer:
            self._buffer[Feats.UP_BUFFER] += up
            self._buffer[Feats.DOWN_BUFFER] += down
            self._buffer[Feats.TIMES].fill_(self.t)
        else:
            self._buffer = {
                Feats.UP_BUFFER: up,
                Feats.DOWN_BUFFER: down,
                Feats.TIMES: torch.ones_like(up) * self.t,
            }

        self.t += self.dt

    def reset(self, mask: torch.Tensor, feat: Feats):
        self._buffer[feat][mask] = 0

    def is_empty(self, keys: Feats | list[Feats]) -> torch.Tensor:
        if isinstance(keys, list):
            mask = None
            for f in keys:
                mask_ = (self.buffer[f] == 0).squeeze(1)
                if mask is None:
                    mask = mask_
                else:
                    mask = mask & mask_

            return mask

        # (B, 1) -> (B,)
        return (self._buffer[keys] != 0).squeeze(1)


def step_actions(
    actions: torch.Tensor,
    idx2ackts: dict[int, Actions],
    Xobs: dict[Feats, torch.Tensor],
    buffer: PacketBuffer,
    t: float,
    idxs: torch.Tensor | None = None,
) -> tuple[dict[Feats, torch.Tensor], PacketBuffer, torch.Tensor]:
    idxs = idxs if idxs is not None else torch.zeros_like(actions).long()

    def _send_buffer(buffer, direction, action_mask):
        buffer = buffer.squeeze(1)
        mb = buffer.max().int().item()
        for b in range(mb):
            buffer_mask = buffer > b
            mask = buffer_mask & action_mask

            Xobs[Feats.DIRS][mask, idxs[mask]] = direction
            Xobs[Feats.TIMES][mask, idxs[mask]] = t
            idxs[mask] += 1

        # buffer[action_mask] = 0

    for a in actions.unique():
        mask = actions == a
        a = idx2ackts[a.item()]

        match a:
            case Actions.SEND_PADDING_DOWN | Actions.SEND_PADDING_UP:
                Xobs[Feats.DIRS][mask, idxs[mask]] = (
                    DOWNLOAD if a == Actions.SEND_PADDING_DOWN else UPLOAD
                )
                Xobs[Feats.PADDING][mask, idxs[mask]] = True
                Xobs[Feats.TIMES][mask, idxs[mask]] = t
                idxs[mask] += 1
            case Actions.SEND_BUFFER:
                _send_buffer(buffer.buffer[Feats.UP_BUFFER], UPLOAD, mask)
                buffer.reset(mask, Feats.UP_BUFFER)
                _send_buffer(buffer.buffer[Feats.DOWN_BUFFER], DOWNLOAD, mask)
                buffer.reset(mask, Feats.DOWN_BUFFER)
            case Actions.WAIT:
                pass
            case _:
                raise ValueError(f"Unknown action: {a}")

    return Xobs, buffer, idxs
