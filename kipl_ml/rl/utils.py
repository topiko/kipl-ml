import torch

from kipl_ml.data.utils import DOWNLOAD, UPLOAD
from kipl_ml.rl.enums import Actions
from kipl_ml.trace.enums import Feats


def _append_to_buffer(
    buffer: torch.Tensor, counts: torch.Tensor, val: int | float
) -> torch.Tensor:
    """
    buffer: (B, M) int/long tensor with current contents
            (0 = empty slot, or whatever convention you use)
    counts: (B,) or (B, 1) tensor: how many new items to append per batch row
    val:    scalar value to write into the buffer
    """
    device = buffer.device
    B, M = buffer.shape

    # (B,) counts as long
    counts = counts.squeeze(-1).long()

    # Current fill level per row: number of non-zero entries
    start = (buffer != 0).sum(dim=1)  # (B,)
    end = start + counts  # (B,)

    # Overflow check – all writes must stay inside [0, M)
    if (end > M).any():
        raise ValueError("Buffer overflow: not enough space to append counts")

    # Build row/col grids of shape (B, M)
    col_grid = torch.arange(M, device=device).view(1, M).expand(B, M)

    # For each row i, valid columns are in [start[i], end[i])
    start_grid = start.view(B, 1)
    end_grid = end.view(B, 1)

    write_mask = (col_grid >= start_grid) & (col_grid < end_grid)  # (B, M) bool

    buffer[write_mask] = val
    return buffer


class PacketBuffer:
    def __init__(self, dt: float):
        self.dt = dt
        self.t = -dt  # After first update self.t = 0
        self._buffer: torch.Tensor | None = None
        self.blen: int = 1000

    @property
    def buffer(self) -> torch.Tensor:
        return self._buffer

    def step(self, X: dict[Feats, torch.Tensor]):
        times = X[Feats.TIMES]
        self.t += self.dt

        mask = (self.t <= times) & (times < self.t + self.dt)

        up = (mask & (X[Feats.DIRS] == UPLOAD)).sum(dim=1).float()
        down = (mask & (X[Feats.DIRS] == DOWNLOAD)).sum(dim=1).float()

        if self._buffer is None:
            self._buffer = torch.zeros(
                (times.shape[0], self.blen, 2), device=times.device
            )

        self._buffer[..., 0] = _append_to_buffer(self._buffer[..., 0], up, UPLOAD)
        self._buffer[..., 0] = _append_to_buffer(self._buffer[..., 0], down, DOWNLOAD)
        self._buffer[..., 1] = _append_to_buffer(
            self._buffer[..., 1], up + down, self.t
        )

    def reset(self, mask: torch.Tensor, feat: Feats):
        self._buffer[feat][mask] = 0

    def is_empty(self) -> torch.Tensor:
        return self.buffer[..., 0].sum(dim=1) == 0

    def pop_oldest(self, mask: torch.Tensor) -> torch.Tensor:
        dirs = self.buffer[mask, 0, 0].clone()
        self.buffer[mask] = self.buffer[mask].roll(-1, dims=1)
        self.buffer[mask, -1, :] = 0

        return dirs

    @property
    def bcounts(self) -> torch.Tensor:
        return (self.buffer[..., 0] != 0).sum(dim=1)

    @property
    def btimes(self) -> torch.Tensor:
        t = torch.where(self.buffer[..., 1] != 0, -(self.buffer[..., 1] - self.t), 0)
        return t.max(dim=1)[0]

    @property
    def feature_dict(self) -> dict[Feats, torch.Tensor]:
        return {
            Feats.UP_BUFFER: (self.buffer[..., 0] == UPLOAD).sum(dim=1, keepdim=True),
            Feats.DOWN_BUFFER: (self.buffer[..., 0] == DOWNLOAD).sum(
                dim=1, keepdim=True
            ),
            Feats.TIMES: torch.ones(self.buffer.shape[0], 1, device=self.buffer.device)
            * self.t,
        }


def step_actions(
    actions: torch.Tensor,
    idx2ackts: dict[int, Actions],
    curXobs: dict[Feats, torch.Tensor],
    buffer: PacketBuffer,
    t: float,
) -> tuple[dict[Feats, torch.Tensor], PacketBuffer]:
    for action in actions.unique():
        mask = actions == action
        action_ = idx2ackts[action]
        match action_:
            case Actions.SEND_PADDING_DOWN | Actions.SEND_PADDING_UP:
                curXobs[Feats.DIRS][mask] = (
                    DOWNLOAD if action_ == Actions.SEND_PADDING_DOWN else UPLOAD
                )
                curXobs[Feats.PADDING][mask] = True
            case Actions.SEND_BUFFER:
                curXobs[Feats.DIRS][mask] = buffer.pop_oldest(mask).unsqueeze(1)
                curXobs[Feats.PADDING][mask] = False
            case Actions.WAIT:
                curXobs[Feats.DIRS][mask] = 0
                curXobs[Feats.PADDING][mask] = False
            case _:
                raise ValueError(f"Unknown action: {action_}")

    curXobs[Feats.TIMES][...] = t
    return curXobs, buffer


if __name__ == "__main__":
    a = torch.zeros(5, 14, 2)

    for i in range(3):
        counts = torch.randint(3, (5,))

        print(a[..., 0])
        print(counts)
        a[..., 0] = _append_to_buffer(a[..., 0], counts, 1)
        print(a[..., 0])
        print()
