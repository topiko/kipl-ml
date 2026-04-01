from __future__ import annotations

import numpy as np
import torch

from kipl_ml.data.utils import DOWNLOAD, UPLOAD
from kipl_ml.logging.logger import get_logger
from kipl_ml.rl.enums import Actions, StepAction, StepActions
from kipl_ml.rl.utils import _flush_left
from kipl_ml.trace.enums import Feats
from kipl_ml.utils.time import (
    _time_to_bin_idx,
)

logger = get_logger(__name__)


def _add_actions_to_silence_periods(
    feature_dict: dict[Feats, torch.Tensor], time_step: float, max_silence_s: float
) -> dict[Feats, torch.Tensor]:
    """Insert empty action windows during silence periods.

    This operates in *time bin index* space for speed and numerical stability.
    - feature_dict[Feats.TIME_BINS] is integer bin indices (long) with -1 after seq end.
    - We insert at least one window for every gap between consecutive bins.
    - Additional windows are inserted every K bins, where
      K = max(1, floor(max_silence_s / time_step)).

    Inserted windows have UP/DOWN counts set to 0.
    """

    idx = feature_dict[Feats.TIME_BINS]
    if idx.ndim != 2:
        raise ValueError("TIMES must be (B, L)")

    B, L = idx.shape
    device = idx.device

    # Convert max silence to a bin step (floor), min 1.
    K = int(max_silence_s / time_step) if time_step > 0 else 1
    K = max(K, 1)

    prev = idx[:, :-1]
    nxt = idx[:, 1:]
    pair_ok = (prev >= 0) & (nxt >= 0)

    # Gap in bins strictly between prev and next.
    gap_bins = nxt - prev - 1
    has_gap = pair_ok & (gap_bins >= 1)

    if not has_gap.any():
        return feature_dict

    rows, cols = torch.where(has_gap)
    # Integer bin start immediately after prev.
    start = prev[rows, cols] + 1
    gap_i = gap_bins[rows, cols]
    # Insert: start + j*K for j=0..count-1 while < nxt.
    count = ((gap_i - 1) // K) + 1

    if (total := int(count.sum().item())) == 0:
        return feature_dict

    starts_rep = start.repeat_interleave(count)
    rows_rep = rows.repeat_interleave(count)

    # Build 0..count-1 offsets per gap without Python loops.
    seg_start = count.cumsum(0) - count
    seg_start_rep = seg_start.repeat_interleave(count)
    offsets = torch.arange(total, device=device, dtype=torch.long) - seg_start_rep

    new_bins = starts_rep + offsets * K

    # Pack per-row inserted bins into a padded (B, max_add) tensor.
    # Sort by row to get contiguous segments.
    order = torch.argsort(rows_rep)
    rows_s = rows_rep[order]
    new_bins_s = new_bins[order]

    row_counts = torch.bincount(rows_s, minlength=B)
    max_add = int(row_counts.max().item())
    add_times = torch.full((B, max_add), -1, device=device, dtype=torch.long)

    row_offsets = row_counts.cumsum(0) - row_counts
    pos = torch.arange(total, device=device, dtype=torch.long) - row_offsets[rows_s]
    add_times[rows_s, pos] = new_bins_s

    # Inserted windows have zero counts.
    add_up = torch.zeros(
        (B, max_add), device=device, dtype=feature_dict[Feats.UP_COUNT].dtype
    )
    add_down = torch.zeros(
        (B, max_add), device=device, dtype=feature_dict[Feats.DOWN_COUNT].dtype
    )

    # Concatenate and sort. Use large value to push -1 sentinels to the end.
    times_all = torch.cat([feature_dict[Feats.TIME_BINS], add_times], dim=1)
    up_all = torch.cat([feature_dict[Feats.UP_COUNT], add_up], dim=1)
    down_all = torch.cat([feature_dict[Feats.DOWN_COUNT], add_down], dim=1)

    sort_key = torch.where(
        times_all >= 0, times_all, torch.full_like(times_all, int(1e18))
    )
    sort_idx = torch.argsort(sort_key, dim=1)

    feature_dict[Feats.TIME_BINS] = times_all.gather(1, sort_idx)
    feature_dict[Feats.UP_COUNT] = up_all.gather(1, sort_idx)
    feature_dict[Feats.DOWN_COUNT] = down_all.gather(1, sort_idx)

    return feature_dict


def get_window_feature_dict(
    X: dict[Feats, torch.Tensor],
    dt: float,
    max_silence_s: float,
    features: list[Feats],
) -> dict[Feats, torch.Tensor]:
    if dt <= 0:
        raise ValueError(f"dt must be > 0, got {dt}")

    # The silence insertion logic assumes max_silence_s is aligned to the bin grid.
    ratio = max_silence_s / dt
    if abs(ratio - round(ratio)) > 1e-8:
        raise ValueError(
            f"max_silence_s must be divisible by dt (max_silence_s={max_silence_s}, dt={dt})."
        )

    if set(X.keys()) > {Feats.PADDING, Feats.DIRS, Feats.TIMES}:
        raise ValueError("Invalid set of feats")

    # (B, L)
    times = X[Feats.TIMES]
    bin_idx = _time_to_bin_idx(times, dt)

    if bin_idx.min() < 0:
        raise ValueError("Negative bin indices found!")

    feature_dict: dict[Feats, torch.Tensor] = {}
    device = times.device
    bs = int(times.shape[0])
    n_bins = int(bin_idx.max().item()) + 1
    shape = (bs, n_bins)

    up_counts = torch.zeros(shape, device=device, dtype=torch.long).scatter_add_(
        1, bin_idx, (X[Feats.DIRS] == UPLOAD).long()
    )
    down_counts = torch.zeros(shape, device=device, dtype=torch.long).scatter_add_(
        1, bin_idx, (X[Feats.DIRS] == DOWNLOAD).long()
    )
    # Store bin indices as long; use scatter to pick the bin index for each occupied bin.
    times_idx = torch.zeros(shape, device=device, dtype=torch.long).scatter_(
        1, bin_idx, bin_idx
    )

    mask = (up_counts != 0) | (down_counts != 0)
    max_l = mask.sum(dim=1).max()
    feature_dict[Feats.UP_COUNT] = _flush_left(up_counts, mask)[:, :max_l]
    feature_dict[Feats.DOWN_COUNT] = _flush_left(down_counts, mask)[:, :max_l]

    times_idx = _flush_left(times_idx.float(), mask).to(torch.long)[:, :max_l]
    mask_fl = _flush_left(mask.float(), mask, pad_val=0).bool()[:, :max_l]

    # Use -1 sentinel for positions after seq end.
    times_bins = torch.where(mask_fl, times_idx, torch.full_like(times_idx, -1))
    feature_dict[Feats.TIME_BINS] = times_bins

    # Insert extra windows into silent gaps (operates in int-bin space).
    feature_dict = _add_actions_to_silence_periods(feature_dict, dt, max_silence_s)

    # Valid mask: bins >= 0.
    mask = feature_dict[Feats.TIME_BINS] >= 0

    seq_lens = mask.sum(dim=1)
    # Compute Dt as bin differences (int bins).
    time_bins = feature_dict[Feats.TIME_BINS]
    dt_bins = torch.zeros_like(time_bins)
    dt_bins[:, :-1] = torch.where(
        mask[:, :-1] & mask[:, 1:],
        time_bins[:, 1:] - time_bins[:, :-1],
        torch.ones_like(time_bins[:, :-1]),  # default 1 bin
    )
    # Last valid window gets 1 bin.
    dt_bins[torch.arange(bs), seq_lens - 1] = 1

    feature_dict[Feats.Dt_BINS] = dt_bins
    # Also provide Dt (duration in seconds) for convenience.
    feature_dict[Feats.Dt] = dt_bins.to(torch.float64) * dt
    max_l = mask.sum(dim=1).max()
    # Flush left with -1 sentinel for int-bin features, 0 for counts.
    out: dict[Feats, torch.Tensor] = {}
    for k, v in feature_dict.items():
        if k in (Feats.TIME_BINS, Feats.Dt_BINS):
            out[k] = _flush_left(v.float(), mask, pad_val=-1).to(torch.long)[:, :max_l]
        elif k == Feats.Dt:
            out[k] = _flush_left(v, mask, pad_val=-1.0)[:, :max_l]
        else:
            out[k] = _flush_left(v, mask, pad_val=0)[:, :max_l]
    feature_dict = out

    feature_dict[Feats.SEQ_LENS] = mask.sum(dim=1)

    if Feats.SILENCE_FLAG in features:
        feature_dict[Feats.SILENCE_FLAG] = (
            (feature_dict[Feats.UP_COUNT] == 0) & (feature_dict[Feats.DOWN_COUNT] == 0)
        ).float()

    K = max(1, int(max_silence_s / dt))
    dt_valid = feature_dict[Feats.Dt_BINS][feature_dict[Feats.Dt_BINS] >= 0]
    if dt_valid.numel() > 0 and int(dt_valid.max().item()) > K:
        logger.warning(
            f"Found max Dt bin {int(dt_valid.max().item())}, "
            f"whereas K={K} (max_silence_s={max_silence_s}, dt={dt})."
        )

    if not all(f in feature_dict for f in features):
        raise ValueError("Some requested features are missing!")

    for f, k in zip((Feats.UP_COUNT, Feats.DOWN_COUNT), (UPLOAD, DOWNLOAD)):
        got = feature_dict[f].sum(dim=1)
        exp = (X[Feats.DIRS] == k).sum(dim=1)
        if (got != exp).any():
            raise ValueError(
                f"Missing packets for {f}: got={got.detach().cpu().tolist()} expected={exp.detach().cpu().tolist()}"
            )

    return feature_dict


class Buffer:
    def __init__(self):
        self.times = torch.Tensor([])
        self.dirs = torch.Tensor([])
        self.pad = torch.Tensor([])

    def add(
        self, times: torch.Tensor, dirs: torch.Tensor, pad: torch.Tensor | None = None
    ) -> None:
        self.times = torch.cat([self.times, times], dim=0)
        self.dirs = torch.cat([self.dirs, dirs], dim=0)
        pad = torch.zeros_like(times) if pad is None else pad
        self.pad = torch.cat([self.pad, pad], dim=0)

    def _delay(self, direction: int, duration_s: float) -> None:
        """Delay packets in the given direction by duration_s seconds."""
        mask = self.dirs == direction
        self.times[mask] += duration_s

    def delay_up(self, duration_s: float) -> None:
        self._delay(UPLOAD, duration_s)

    def delay_down(self, duration_s: float) -> None:
        self._delay(DOWNLOAD, duration_s)

    def flush(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Flush the buffer and return (times, dirs, pad)."""
        times = self.times
        dirs = self.dirs
        pad = self.pad
        self.times = torch.Tensor([])
        self.dirs = torch.Tensor([])
        self.pad = torch.Tensor([])
        return times, dirs, pad


def _get_from_interval(
    times: torch.Tensor, dirs: torch.Tensor, start_s: float, end_s: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Get packets in the interval [start_s, end_s)."""
    mask = (times >= start_s) & (times < end_s)
    return times[mask], dirs[mask]


class TraceStateCursor:
    """Per-trace cursor producing the same bin sequence as get_window_feature_dict.

    It iterates over packet bins and inserts silence bins in gaps using the same
    K-step rule as _add_actions_to_silence_periods().
    """

    def __init__(
        self,
        times: torch.Tensor,
        dirs: torch.Tensor,
        dt: float,
        terminate_after_s: float | None = None,
        max_silence_bins: int | None = None,
    ):
        self.dt = dt
        self.times = times
        self.dirs = dirs
        self.buffer: Buffer = Buffer()
        self.cursor_time_bin: int = 0
        self.prev_time_bin: int = 0
        self.max_silence_bins = max_silence_bins
        self.terminate_after_bin = (
            int(round(float(terminate_after_s) / float(dt)))
            if terminate_after_s is not None
            else int(_time_to_bin_idx(times.max().unsqueeze(0), dt).item()) + 1
        )

    def step(
        self, actions: StepAction
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
        if self.prev_time_bin >= self.terminate_after_bin:
            raise StopIteration

        # Action logic here.
        if Actions.DO_NOTHING in actions:
            times, dirs = _get_from_interval(
                self.times,
                self.dirs,
                float(self.cursor_time_bin) * self.dt,
                float(self.cursor_time_bin + 1) * self.dt,
            )
            self.buffer.add(times, dirs)
            dt_bins = self.cursor_time_bin - self.prev_time_bin
        else:
            raise NotImplementedError(f"Unsupported action set: {actions.keys()}")

        self.prev_time_bin = self.cursor_time_bin
        self.cursor_time_bin += 1

        times, dirs, pad = self.buffer.flush()

        return times, dirs, pad, self.cursor_time_bin, dt_bins


class WindowFeatureStreamer:
    """Stream action-window features (B, 1) step-by-step.

    This matches get_window_feature_dict() for the feature set used by AGENT1.
    """

    def __init__(
        self,
        X: dict[Feats, torch.Tensor],
        dt: float,
        max_silence_s: float,
        features: list[Feats],
        cut_off_time_s: float | torch.Tensor | None = None,
    ):
        if dt <= 0:
            raise ValueError(f"dt must be > 0, got {dt}")

        ratio = max_silence_s / dt
        if abs(ratio - round(ratio)) > 1e-8:
            raise ValueError(
                f"max_silence_s must be divisible by dt "
                f"(max_silence_s={max_silence_s}, dt={dt})."
            )

        if set(X.keys()) > {Feats.PADDING, Feats.DIRS, Feats.TIMES}:
            raise ValueError("Invalid set of feats")

        self.features = features
        self.device = X[Feats.TIMES].device
        self.bs = X[Feats.TIMES].shape[0]
        self.dt = float(dt)
        self.X = {k: v.clone() for k, v in X.items()}

        # K in bin index space.
        K = int(max_silence_s / dt) if dt > 0 else 1
        self.K = max(K, 1)

        self._cursors: list[TraceStateCursor] = []
        for i in range(self.bs):
            self._cursors.append(
                TraceStateCursor(
                    times=self.X[Feats.TIMES][i],
                    dirs=self.X[Feats.DIRS][i],
                    dt=dt,
                    max_silence_bins=self.K,
                )
            )

        # Per-trace cutoff in bin space. None means "last packet bin + 1".
        if cut_off_time_s is None:
            self.cut_off_bins = torch.tensor(
                [
                    int(
                        _time_to_bin_idx(
                            self.X[Feats.TIMES][i][self.X[Feats.DIRS][i] != 0], dt
                        )
                        .max()
                        .item()
                    )
                    + 1
                    for i in range(self.bs)
                ],
                device=self.device,
                dtype=torch.long,
            )
        else:
            # cut_off = torch.as_tensor(cut_off_time_s, device=self.device)
            raise NotImplementedError("cut_off_time_s is not implemented yet")

        self.done = np.zeros(self.bs, dtype=bool)

    def active_mask(self) -> torch.Tensor:
        return ~self.done

    def step(
        self, actions: StepActions
    ) -> tuple[
        dict[Feats, torch.Tensor], dict[Feats, list[torch.Tensor]], torch.Tensor
    ]:
        bs = self.bs
        device = self.device

        # Emit int bins (not float times)
        time_bins = torch.full(
            (bs, 1), -1, device=device, dtype=torch.long
        )  # -1 = invalid
        up = torch.zeros((bs, 1), device=device, dtype=torch.long)
        down = torch.zeros((bs, 1), device=device, dtype=torch.long)
        dt_bins = torch.full((bs, 1), -1, device=device, dtype=torch.long)  # bin count

        times_l: list[torch.Tensor] = []
        dirs_l: list[torch.Tensor] = []
        padding_l: list[torch.Tensor] = []

        active_idxs = np.arange(bs)[self.active_mask()]
        for i, aidx in enumerate(active_idxs):
            try:
                w_times, w_dirs, w_padding, time_bin_, dt_bins_ = self._cursors[
                    aidx
                ].step(actions[i])
            except StopIteration:
                self.done[i] = True
                continue

            times_l.append(w_times)
            dirs_l.append(w_dirs)
            padding_l.append(w_padding)

            up[i, 0] = (w_dirs == UPLOAD).sum()
            down[i, 0] = (w_dirs == DOWNLOAD).sum()
            dt_bins[i, 0] = dt_bins_
            time_bins[i, 0] = time_bin_

        fd: dict[Feats, torch.Tensor] = {
            Feats.UP_COUNT: up,
            Feats.DOWN_COUNT: down,
            Feats.Dt_BINS: dt_bins,
            Feats.Dt: dt_bins.to(torch.float64) * self.dt,
            Feats.TIME_BINS: time_bins,
        }

        if Feats.SILENCE_FLAG in self.features:
            fd[Feats.SILENCE_FLAG] = ((up == 0) & (down == 0)).float()

        if not all(f in fd for f in self.features):
            raise ValueError("Some requested features are missing!")

        # If time_bins < 0, this row is done and the values are invalid.
        # Mask out just in case.
        terminated = (time_bins < 0).squeeze(1)
        return (
            {f: fd[f] for f in self.features},
            {Feats.TIMES: times_l, Feats.DIRS: dirs_l, Feats.PADDING: padding_l},
            ~terminated,
        )
