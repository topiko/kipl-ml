from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import torch

from kipl_ml.data.utils import DOWNLOAD, UPLOAD
from kipl_ml.logging.logger import get_logger
from kipl_ml.rl.enums import ActDelay, Actions, NoAction, StepAction, StepActions
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
    warnings.warn(
        "get_window_feature_dict is deprecated; use WindowFeatureStreamer instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    if dt <= 0:
        raise ValueError(f"dt must be > 0, got {dt}")

    # The silence insertion logic assumes max_silence_s is aligned to the bin grid.
    ratio = max_silence_s / dt
    if abs(ratio - round(ratio)) > 1e-8:
        raise ValueError(
            f"max_silence_s must be divisible by dt (max_silence_s={max_silence_s}, dt={dt})."
        )

    if set(X.keys()) > {Feats.DECOY, Feats.DIRS, Feats.TIMES}:
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


class SendBuffer:
    def __init__(
        self,
        flush_bin: int,
        dt: float,
        times: torch.Tensor,
        dirs: torch.Tensor,
        decoy: bool = False,
        replace: bool = False,
        bypass: bool = False,
    ):
        self.times = times
        self.dirs = dirs
        self._flush_bin = flush_bin
        self.dt = dt
        self.replace = replace
        self.bypass = bypass
        self.decoy = decoy
        self.flushed = False

    @property
    def flush_bin(self) -> int:
        return self._flush_bin

    def _delay(self, direction: int, time_bin: int) -> None:
        """Delay packets in the given direction by duration_s seconds."""
        mask = self.dirs == direction
        self.times[mask] += self.dt

        self._flush_bin = max(self._flush_bin, time_bin + 1)

    def delay_up(self, time_bin: int) -> None:
        self._delay(UPLOAD, time_bin)

    def delay_down(self, time_bin: int) -> None:
        self._delay(DOWNLOAD, time_bin)

    def subtract(self, dirs: torch.Tensor) -> None:
        if not self.replace:
            return
        if not self.decoy:
            return

        mydir = self.dirs.unique()
        if len(mydir) != 1:
            raise ValueError(
                "Expected all packets in the buffer to have the same direction"
            )

        # Decoy buffer with replace enabled -> subtract matching queued normal packets.
        npackets = len(self.dirs)
        if (to_sub := min(npackets, (dirs == mydir).sum())) <= 0:
            return

        self.dirs = self.dirs[:-to_sub]
        self.times = self.times[:-to_sub]

    def flush(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Flush the buffer and return (times, dirs)."""
        if self.flushed:
            raise RuntimeError("Buffer already flushed")
        self.flushed = True
        decoy = torch.full_like(self.times, self.decoy)
        return self.times, self.dirs, decoy


@dataclass
class DelayState:
    """Active delay window for one direction."""

    steps_left: int = 0
    bypass: bool = False
    replace: bool = False

    @property
    def active(self) -> bool:
        return self.steps_left > 0

    def update(self, action: ActDelay) -> None:
        """Start or extend the active delay window.

        `replace=True` overwrites the current state. If a delay is already
        active and `replace=False`, the active delay must remain unchanged.
        """

        if self.active and not action.replace:
            return

        self.steps_left = max(0, int(action.steps))
        self.bypass = action.bypass
        self.replace = action.replace

    def step(self, buffers: list[SendBuffer], time_bin: int, direction: int) -> None:
        """Apply one delay step to eligible send buffers."""

        if not self.active:
            return

        for buf in buffers:
            # If both the active delay and the buffer are bypass-enabled, leave it
            # untouched. If self.bypass is False, the delay applies to everything.
            if buf.bypass and self.bypass:
                continue
            if direction == UPLOAD:
                buf.delay_up(time_bin)
            else:
                buf.delay_down(time_bin)

        self.steps_left -= 1


def _get_from_interval(
    times: torch.Tensor,
    dirs: torch.Tensor,
    start_s: float,
    end_s: float,
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
        decoy: torch.Tensor,
        dt: float,
        terminate_after_s: float | None = None,
    ):
        self.dt = dt
        self.times = times
        self.dirs = dirs
        self.decoy = decoy
        # `send_buffers` collect packets scheduled during the current cursor walk.
        # They hold both original packets from the active window and decoy packets
        # injected by the current action, then flush once their target bin is reached.
        self.send_buffers: list[SendBuffer] = []
        self.delay_up = DelayState()
        self.delay_down = DelayState()
        self.cursor_time_bin: int = 0
        self.prev_time_bin: int = 0
        self.terminate_after_s = terminate_after_s or times.max().item() + dt
        self.device = times.device

    def step(
        self, actions: StepAction
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
        if self.prev_time_bin * self.dt >= self.terminate_after_s:
            raise StopIteration

        assert actions.time == self.cursor_time_bin, (
            f"Expected action time {self.cursor_time_bin}, got {actions.time}"
        )

        times, dirs = _get_from_interval(
            self.times,
            self.dirs,
            float(self.cursor_time_bin) * self.dt,
            float(self.cursor_time_bin + 1) * self.dt,
        )
        self.send_buffers.append(
            SendBuffer(
                flush_bin=self.cursor_time_bin,
                dt=self.dt,
                times=times,
                dirs=dirs,
                decoy=False,
            )
        )
        # Action logic here.
        # DelayTraffic semantics to mirror later implementation:
        # - delay is framework-scoped and affects all outgoing traffic
        # - bypass=True means only bypass-enabled decoy traffic may skip the delay
        # - replace=True replaces the active delay; otherwise keep the strongest
        #   active delay (max duration and max packet count independently)
        # - when a delay is started or adjusted, its bypassable status is replaced too
        # DECOY follows the packet through the buffers so the final reconstructed
        # trace can distinguish original packets from injected decoys.
        act_time_bin = actions.time
        dt_bins = self.cursor_time_bin - self.prev_time_bin

        # Send down actions:
        if Actions.SEND_DOWN in actions:
            act_s_d = actions[Actions.SEND_DOWN]
            times = torch.full(
                (act_s_d.count,),
                (act_time_bin + act_s_d.after_steps + 0.5) * self.dt,
                device=self.device,
            )
            dirs = torch.ones_like(times) * DOWNLOAD
            sdb = SendBuffer(
                act_time_bin + act_s_d.after_steps,
                self.dt,
                times,
                dirs,
                True,
                replace=act_s_d.replace,
                bypass=act_s_d.bypass,
            )

            self.send_buffers.append(sdb)
            if act_s_d.replace:
                raise NotImplementedError("replace=True is not implemented yet")

        # Send up actions:
        if Actions.SEND_UP in actions:
            act_s_u = actions[Actions.SEND_UP]
            times = torch.full(
                (act_s_u.count,),
                (act_time_bin + act_s_u.after_steps + 0.5) * self.dt,
                device=self.device,
            )
            dirs = torch.ones_like(times) * UPLOAD
            sdb = SendBuffer(
                act_time_bin + act_s_u.after_steps,
                self.dt,
                times,
                dirs,
                True,
                replace=act_s_u.replace,
                bypass=act_s_u.bypass,
            )

            self.send_buffers.append(sdb)
            if act_s_u.replace:
                raise NotImplementedError("replace=True is not implemented yet")

        if Actions.DELAY_DOWN in actions:
            self.delay_down.update(actions[Actions.DELAY_DOWN])

        if Actions.DELAY_UP in actions:
            self.delay_up.update(actions[Actions.DELAY_UP])

        # Apply delays:
        self.delay_down.step(self.send_buffers, act_time_bin, DOWNLOAD)
        self.delay_up.step(self.send_buffers, act_time_bin, UPLOAD)

        # Then flush buffers and collect.
        times_l: list[torch.Tensor] = []
        dirs_l: list[torch.Tensor] = []
        decoy_l: list[torch.Tensor] = []

        # First apply the normal packets
        for buf in self.send_buffers:
            if buf.decoy:
                continue

            if buf.flush_bin < self.cursor_time_bin:
                raise RuntimeError(
                    f"Buffer flush_bin {buf.flush_bin} is in the past (cursor_time_bin={self.cursor_time_bin})"
                )
            elif buf.flush_bin == self.cursor_time_bin:
                times, dirs, decoy = buf.flush()
                times_l.append(times)
                dirs_l.append(dirs)
                decoy_l.append(decoy)

                # For all scheduled future decoy buffers with replace=True,
                # subtract the flushed packets if possible.
                for future_buf in self.send_buffers:
                    if future_buf.decoy and future_buf.replace:
                        future_buf.subtract(dirs)

        # Then the decoy ones:
        for buf in self.send_buffers:
            if not buf.decoy:
                continue

            if buf.flush_bin < self.cursor_time_bin:
                raise RuntimeError(
                    f"Buffer flush_bin {buf.flush_bin} is in the past (cursor_time_bin={self.cursor_time_bin})"
                )
            if buf.flush_bin == self.cursor_time_bin:
                times, dirs, decoy = buf.flush()
                times_l.append(times)
                dirs_l.append(dirs)
                decoy_l.append(decoy)

        self.send_buffers = [buf for buf in self.send_buffers if not buf.flushed]

        self.prev_time_bin = self.cursor_time_bin
        self.cursor_time_bin += 1

        times = torch.cat(times_l) if times_l else torch.Tensor([])
        dirs = torch.cat(dirs_l) if dirs_l else torch.Tensor([])
        decoy = torch.cat(decoy_l) if decoy_l else torch.Tensor([])

        return times, dirs, decoy, self.cursor_time_bin, dt_bins


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
        # K in bin index space.
        self.max_silence_bins = int(ratio)
        if abs(ratio - round(ratio)) > 1e-8:
            raise ValueError(
                f"max_silence_s must be divisible by dt "
                f"(max_silence_s={max_silence_s}, dt={dt})."
            )

        if set(X.keys()) > {Feats.DECOY, Feats.DIRS, Feats.TIMES}:
            raise ValueError("Invalid set of feats")

        self.features = features
        self.device = X[Feats.TIMES].device
        self.bs = X[Feats.TIMES].shape[0]
        self.dt = float(dt)
        self.X = {k: v.clone() for k, v in X.items()}

        self._cursors: list[TraceStateCursor] = []
        for i in range(self.bs):
            self._cursors.append(
                TraceStateCursor(
                    times=self.X[Feats.TIMES][i],
                    dirs=self.X[Feats.DIRS][i],
                    decoy=self.X[Feats.DECOY][i],
                    dt=dt,
                    terminate_after_s=cut_off_time_s[i].item()
                    if cut_off_time_s is not None
                    else None,
                )
            )

        self.done = np.zeros(self.bs, dtype=bool)

    def active_mask(self) -> torch.Tensor:
        return ~self.done

    def step(
        self,
        actions: StepActions,
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
        decoy_l: list[torch.Tensor] = []

        active_idxs = np.arange(bs)[self.active_mask()]
        for i, aidx in enumerate(active_idxs):
            cursor = self._cursors[aidx]
            cur_action = actions[i]
            slept_bins = 0
            while True:
                try:
                    w_times, w_dirs, w_decoy, time_bin_, dt_bins_ = cursor.step(
                        cur_action
                    )
                except StopIteration:
                    self.done[aidx] = True
                    break

                if w_times.numel() > 0 or slept_bins >= self.max_silence_bins:
                    break

                slept_bins += 1
                cur_action = NoAction(time=cursor.cursor_time_bin)

            if self.done[aidx]:
                continue

            times_l.append(w_times)
            dirs_l.append(w_dirs)
            decoy_l.append(w_decoy)

            up[aidx, 0] = (w_dirs == UPLOAD).sum()
            down[aidx, 0] = (w_dirs == DOWNLOAD).sum()
            dt_bins[aidx, 0] = dt_bins_
            time_bins[aidx, 0] = time_bin_

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
            {Feats.TIMES: times_l, Feats.DIRS: dirs_l, Feats.DECOY: decoy_l},
            ~terminated,
        )
