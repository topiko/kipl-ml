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

    def _send_buffer(buffer_counts, direction, action_mask):
        # buffer_counts: (B, 1) or (B,)
        # action_mask: (B,) bool
        # uses outer-scope Xobs, idxs, t

        counts = buffer_counts.squeeze(1)  # (B,)
        active_rows = torch.nonzero(action_mask, as_tuple=False).squeeze(1)  # (R,)
        if active_rows.numel() == 0:
            return

        counts = counts[active_rows].long()  # (R,)
        total_count = counts.sum()
        if total_count == 0:
            return

        max_c = counts.max()  # still on device

        # shape: (max_c,)
        arange = torch.arange(max_c, device=counts.device)
        # (1, max_c) < (R, 1) -> (R, max_c)
        valid = arange.unsqueeze(0) < counts.unsqueeze(1)

        # row indices
        # (R, 1) -> (R, max_c)[valid] -> (total, )
        row_idx = active_rows.unsqueeze(1).expand(-1, max_c)[valid]  # (total,)

        # col indices: start at idxs[row] and go up
        # (R, 1)
        start = idxs[active_rows].unsqueeze(1)
        # (R, 1) + (1, max_c) -> (R, max_c)[valid] -> (total, )
        col_idx = (start + arange.unsqueeze(0))[valid]

        Xobs[Feats.DIRS][row_idx, col_idx] = direction
        Xobs[Feats.TIMES][row_idx, col_idx] = t

        # bump idxs by how many entries we filled per row
        idxs[active_rows] += counts

    for a in actions.unique():
        mask = actions == a
        a = idx2ackts[a]
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
