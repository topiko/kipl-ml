import torch

from kipl_ml.data.utils import DOWNLOAD, UPLOAD
from kipl_ml.logging.logger import get_logger
from kipl_ml.rl.enums import Actions
from kipl_ml.trace.enums import Feats

logger = get_logger(__name__)


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


def _append_values(buffer: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    if buffer.shape[0] != values.shape[0]:
        raise ValueError("Batch size of buffer and values must match")

    B = buffer.shape[0]

    # (B, )
    start_idxs = (buffer != 0).sum(dim=1)

    if ((start_idxs + values.shape[1]) > buffer.shape[1]).any():
        raise ValueError("Not enough space in buffer to append values")

    M = values.shape[1]
    # (B, M) column indices
    col_idxs = torch.arange(M, device=buffer.device).unsqueeze(0).expand(
        B, -1
    ) + start_idxs.unsqueeze(1)

    # (B, M) row indices
    row_idxs = torch.arange(B, device=buffer.device).unsqueeze(1).expand(-1, M)

    buffer[row_idxs, col_idxs] = values

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

    def pop_oldest(self, mask: torch.Tensor, count: int = 1) -> torch.Tensor:
        dirs = self.buffer[mask, :count, 0].clone()
        self.buffer[mask] = self.buffer[mask].roll(-count, dims=1)
        self.buffer[mask, -count:, :] = 0

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


class TraceObservation:
    def __init__(self, B: int, L: int, device: torch.DeviceObjType):
        self.L = L
        self.X = torch.zeros((B, L, 3), device=device)  # dirs, times, padding

    @property
    def times(self) -> torch.Tensor:
        return self.X[..., 1]

    @property
    def dirs(self) -> torch.Tensor:
        return self.X[..., 0]

    @property
    def count_send(self) -> torch.Tensor:
        return (self.X[..., 0] != 0).sum(dim=1)

    @property
    def count_padding(self) -> torch.Tensor:
        return (self.X[..., 2] != 1).sum(dim=1)

    @property
    def is_waiting(self) -> torch.Tensor:
        return self.X[:, 0, 0] == 0

    def pop_oldest(self) -> tuple[dict[Feats, torch.Tensor], torch.Tensor]:
        # (B, 1)
        dirs_ = self.X[:, 0, 0].clone()
        times_ = self.X[:, 0, 1].clone()
        padding_ = self.X[:, 0, 2].clone()

        self.X = self.X.roll(-1, dims=1)
        self.X[:, -1, :] = 0

        # (B, 1) -> (B,)
        mask = dirs_ == 0

        # (B, 1)
        obs = {
            Feats.DIRS: dirs_.unsqueeze(1),
            Feats.TIMES: times_.unsqueeze(1),
            Feats.PADDING: padding_.unsqueeze(1),
        }
        return obs, mask

    @property
    def feature_dict(self) -> dict[Feats, torch.Tensor]:
        return {
            Feats.DIRS: self.X[..., 0],
            Feats.TIMES: self.X[..., 1],
            Feats.PADDING: self.X[..., 2],
        }

    def reset(self):
        self.X[...] = 0


def step_actions(
    actions: torch.Tensor,
    idx2ackts: dict[int, tuple[Actions, int]],
    curXobs: TraceObservation,
    buffer: PacketBuffer,
    t: float,
) -> tuple[TraceObservation, PacketBuffer]:
    for action in actions.unique():
        mask = actions == action
        action_, count = idx2ackts[action]
        counts = torch.zeros_like(mask, dtype=torch.int)
        counts[mask] = count
        match action_:
            case Actions.SEND_PADDING_DOWN | Actions.SEND_PADDING_UP:
                val = DOWNLOAD if action_ == Actions.SEND_PADDING_DOWN else UPLOAD

                curXobs.X[mask, :, 0] = _append_to_buffer(
                    curXobs.X[mask, :, 0], counts[mask], val
                )
                curXobs.X[mask, :, 1] = _append_to_buffer(
                    curXobs.X[mask, :, 1], counts[mask], t
                )
                curXobs.X[mask, :, 2] = _append_to_buffer(
                    curXobs.X[mask, :, 2], counts[mask], 1
                )
            case Actions.SEND_BUFFER:
                send = buffer.pop_oldest(mask, count)
                curXobs.X[mask, :, 0] = _append_values(curXobs.X[mask, :, 0], send)

                counts_ = (send != 0).sum(dim=1)
                curXobs.X[mask, :, 1] = _append_to_buffer(
                    curXobs.X[mask, :, 1], counts_, t
                )
                curXobs.X[mask, :, 2] = _append_to_buffer(
                    curXobs.X[mask, :, 2], counts_, 2
                )
            case Actions.WAIT:
                pass
            case _:
                raise ValueError(f"Unknown action: {action_}")

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
