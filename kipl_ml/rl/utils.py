from __future__ import annotations

import torch

from kipl_ml.data.utils import DOWNLOAD, UPLOAD
from kipl_ml.logging.logger import get_logger
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


def _push_left(
    values: torch.Tensor, keep_mask: torch.Tensor, pad_val: float = 0
) -> torch.Tensor:
    if values.shape != keep_mask.shape:
        raise ValueError(
            f"Size of values and mask must match got: {values.shape} and {keep_mask.shape}."
        )

    B = values.shape[0]

    # (B, )
    values_pushed = torch.ones_like(values) * pad_val

    # (B, M) row indices, M = values.shape[1]
    row_idxs = (
        torch.arange(B, device=values.device).unsqueeze(1).expand(-1, values.shape[1])
    )

    col_idxs = keep_mask.cumsum(dim=1) - 1

    # (N, )
    row_valid = row_idxs[keep_mask]
    col_valid = col_idxs[keep_mask]

    values_pushed[row_valid, col_valid] = values[keep_mask]

    return values_pushed


def _append_values(
    base: torch.Tensor, values: torch.Tensor, on_short_base: str = "raise"
) -> torch.Tensor:
    if base.shape[0] != values.shape[0]:
        raise ValueError("Batch size of buffer and values must match")

    B = base.shape[0]

    # (B, )
    start_idxs = (base != 0).sum(dim=1)

    if ((start_idxs + values.shape[1]) > base.shape[1]).any():
        if on_short_base == "raise":
            raise ValueError("Base buffer too short to append values")
        elif on_short_base == "cat":
            # Allow for base expansion
            base = torch.cat(
                (base, torch.zeros((B, values.shape[1]), device=base.device)), dim=1
            )
        else:
            raise ValueError(f"Unknown on_short_base option: {on_short_base}")

    M = values.shape[1]
    # (B, M) column indices
    col_idxs = torch.arange(M, device=base.device).unsqueeze(0).expand(
        B, -1
    ) + start_idxs.unsqueeze(1)

    # (B, M) row indices
    row_idxs = torch.arange(B, device=base.device).unsqueeze(1).expand(-1, M)

    base[row_idxs, col_idxs] = values

    return base


class PacketBuffer:
    def __init__(self, dt: float, blen: int = 5000):
        self.dt = dt
        self.t = -dt  # After first update self.t = 0
        self._buffer: torch.Tensor | None = None
        self.blen: int = blen
        self._buffer_list: list[torch.Tensor] = []

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

    def reset(self):
        self._buffer[...] = 0

    def is_empty(self) -> torch.Tensor:
        return self.buffer[..., 0].sum(dim=1) == 0

    def pop_oldest(self, mask: torch.Tensor, count: int = 1) -> torch.Tensor:
        dirs = self.packets[mask, :count].clone()
        self.buffer[mask] = self.buffer[mask].roll(-count, dims=1)
        self.buffer[mask, -count:, :] = 0

        return dirs

    @property
    def packets(self) -> torch.Tensor:
        return self.buffer[..., 0]

    @property
    def times(self) -> torch.Tensor:
        return self.buffer[..., 1]

    @property
    def bcounts(self) -> torch.Tensor:
        return (self.packets != 0).sum(dim=1)

    @property
    def btimes(self) -> torch.Tensor:
        t = torch.where(self.times != 0, -(self.times - self.t - self.dt), 0)
        return t.max(dim=1)[0]

    @property
    def feature_dict(self) -> dict[Feats, torch.Tensor]:
        return {
            Feats.UP_BUFFER: (self.packets == UPLOAD).sum(dim=1, keepdim=True).float(),
            Feats.DOWN_BUFFER: (self.packets == DOWNLOAD)
            .sum(dim=1, keepdim=True)
            .float(),
            Feats.TIMES: torch.ones(
                self.buffer.shape[0], 1, device=self.buffer.device
            ).float()
            * self.t,
        }

    def append(self):
        self._buffer_list.append(
            {k: v.clone().detach() for k, v in self.feature_dict.items()}
        )

    @property
    def history(self) -> dict[Feats, torch.Tensor]:
        feats = [Feats.UP_BUFFER, Feats.DOWN_BUFFER, Feats.TIMES]
        return {k: torch.cat([b[k] for b in self._buffer_list], dim=1) for k in feats}
