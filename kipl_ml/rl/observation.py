from __future__ import annotations

import torch

from kipl_ml.data.utils import DOWNLOAD, UPLOAD
from kipl_ml.logging.logger import get_logger
from kipl_ml.rl.utils import (
    _boundary_time_to_bin_idx,
    _duration_to_bin_offsets,
    _flush_left,
    _time_to_bin_idx,
    fill_after_seq_end,
)
from kipl_ml.trace.enums import Feats

logger = get_logger(__name__)


def _as_batch_vec(x: torch.Tensor, name: str) -> torch.Tensor:
    """Normalize (B,) or (B,1) tensors to (B,)."""
    if x.ndim == 2:
        if x.shape[1] != 1:
            raise ValueError(f"{name} must be (B,) or (B,1)")
        x = x.squeeze(1)
    if x.ndim != 1:
        raise ValueError(f"{name} must be (B,) or (B,1)")
    return x


def _add_actions_to_silence_periods(
    feature_dict: dict[Feats, torch.Tensor], time_step: float, max_silence_s: float
) -> dict[Feats, torch.Tensor]:
    """Insert empty action windows during silence periods.

    This operates in *time bin index* space for speed and numerical stability.
    - feature_dict[Feats.TIMES] is interpreted as bin indices (float) with NaNs after seq end.
    - We insert at least one window for every gap between consecutive bins.
    - Additional windows are inserted every K bins, where
      K = max(1, floor(max_silence_s / time_step)).

    Inserted windows have UP/DOWN counts set to 0.
    """

    idx = feature_dict[Feats.TIMES]
    if idx.ndim != 2:
        raise ValueError("TIMES must be (B, L)")

    B, L = idx.shape
    device = idx.device

    # Convert max silence to a bin step (floor), min 1.
    K = int(max_silence_s / time_step) if time_step > 0 else 1
    K = max(K, 1)

    prev = idx[:, :-1]
    nxt = idx[:, 1:]
    pair_ok = prev.isfinite() & nxt.isfinite()

    # Gap in bins strictly between prev and next.
    gap_bins = nxt - prev - 1
    has_gap = pair_ok & (gap_bins >= 1)

    if not has_gap.any():
        return feature_dict

    rows, cols = torch.where(has_gap)
    # Integer bin start immediately after prev.
    start = prev[rows, cols].to(torch.long) + 1
    gap_i = gap_bins[rows, cols].to(torch.long)
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
    add_times = torch.full((B, max_add), torch.nan, device=device, dtype=idx.dtype)

    row_offsets = row_counts.cumsum(0) - row_counts
    pos = torch.arange(total, device=device, dtype=torch.long) - row_offsets[rows_s]
    add_times[rows_s, pos] = new_bins_s.to(dtype=idx.dtype)

    # Inserted windows have zero counts.
    add_up = torch.zeros_like(add_times)
    add_down = torch.zeros_like(add_times)

    # Concatenate and sort. Use a stable key to push NaNs to the end.
    times_all = torch.cat([feature_dict[Feats.TIMES], add_times], dim=1)
    up_all = torch.cat([feature_dict[Feats.UP_COUNT], add_up], dim=1)
    down_all = torch.cat([feature_dict[Feats.DOWN_COUNT], add_down], dim=1)

    sort_key = times_all.nan_to_num(nan=float("inf"))
    sort_idx = torch.argsort(sort_key, dim=1)

    feature_dict[Feats.TIMES] = times_all.gather(1, sort_idx)
    feature_dict[Feats.UP_COUNT] = up_all.gather(1, sort_idx)
    feature_dict[Feats.DOWN_COUNT] = down_all.gather(1, sort_idx)

    return feature_dict


def get_window_feature_dict(
    X: dict[Feats, torch.Tensor],
    dt: float,
    max_silence_s: float,
    features: list[Feats],
    extend_end_s: float = 0,
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

    if extend_end_s > 0:
        bs, L = X[Feats.DIRS].shape
        mask = X[Feats.DIRS] == 0
        seq_lens = (~mask).sum(dim=1)
        col_idx = seq_lens[seq_lens != L]
        row_idx = torch.arange(bs, device=seq_lens.device)[seq_lens != L]
        # Here we add artificial packet to end.
        X[Feats.DIRS][row_idx, col_idx] = UPLOAD

        # Here we add the time extension.
        X[Feats.TIMES][mask] += extend_end_s

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

    up_counts = torch.zeros(shape, device=device).scatter_add_(
        1, bin_idx, (X[Feats.DIRS] == UPLOAD).float()
    )
    down_counts = torch.zeros(shape, device=device).scatter_add_(
        1, bin_idx, (X[Feats.DIRS] == DOWNLOAD).float()
    )
    times_idx = torch.zeros(shape, device=device).scatter_(1, bin_idx, bin_idx.float())

    mask = (up_counts != 0) | (down_counts != 0)
    max_l = mask.sum(dim=1).max()
    feature_dict[Feats.UP_COUNT] = _flush_left(up_counts, mask)[:, :max_l]
    feature_dict[Feats.DOWN_COUNT] = _flush_left(down_counts, mask)[:, :max_l]

    times_idx = _flush_left(times_idx, mask)[:, :max_l]
    mask_fl = _flush_left(mask.float(), mask, pad_val=0).bool()[:, :max_l]

    # Keep TIMES as bin indices (float) with NaNs after seq end.
    # TIMES can legitimately contain zeros, so do not use a sentinel pad value.
    times_bins = fill_after_seq_end(times_idx, mask_fl, fill_val="nan")
    feature_dict[Feats.TIMES] = times_bins

    # Insert extra windows into silent gaps using bin indices, then convert to seconds.
    feature_dict = _add_actions_to_silence_periods(feature_dict, dt, max_silence_s)
    feature_dict[Feats.TIMES] = feature_dict[Feats.TIMES] * dt

    mask = feature_dict[Feats.TIMES].isfinite()

    seq_lens = mask.sum(dim=1)
    dts = feature_dict[Feats.TIMES].diff(
        dim=1, append=torch.zeros((bs, 1), device=times.device)
    )
    # The last window is considered to be dt wide.
    dts[torch.arange(bs), seq_lens - 1] = dt

    feature_dict[Feats.Dt] = dts
    max_l = mask.sum(dim=1).max()
    # dict[Feats, Tensor (B, max_l)]
    feature_dict = {
        k: _flush_left(v, mask, pad_val=torch.nan)[:, :max_l]
        for k, v in feature_dict.items()
    }

    feature_dict[Feats.SEQ_LENS] = mask.sum(dim=1)

    if Feats.SILENCE_FLAG in features:
        feature_dict[Feats.SILENCE_FLAG] = (
            (feature_dict[Feats.UP_COUNT] == 0) & (feature_dict[Feats.DOWN_COUNT] == 0)
        ).float()

    if (max_s := feature_dict[Feats.Dt].diff(dim=1).max()) > max_silence_s:
        logger.warning(
            f"Found max silence {max_s:.4f}, whereas you wish max silence = {max_silence_s:.4f}."
        )

    if not all(f in feature_dict for f in features):
        raise ValueError("Some requested features are missing!")

    for f, k in zip((Feats.UP_COUNT, Feats.DOWN_COUNT), (UPLOAD, DOWNLOAD)):
        if (
            torch.where(feature_dict[f].isfinite(), feature_dict[f], 0).sum(dim=1)
            != (X[Feats.DIRS] == k).sum(dim=1)
        ).any():
            got = torch.where(feature_dict[f].isfinite(), feature_dict[f], 0).sum(dim=1)
            exp = (X[Feats.DIRS] == k).sum(dim=1)
            raise ValueError(
                f"Missing packets for {f}: got={got.detach().cpu().tolist()} expected={exp.detach().cpu().tolist()}"
            )

    return feature_dict


class _TraceWindowCursor:
    """Per-trace cursor producing the same bin sequence as get_window_feature_dict.

    It iterates over packet bins and inserts silence bins in gaps using the same
    K-step rule as _add_actions_to_silence_periods().
    """

    def __init__(
        self,
        times: torch.Tensor,
        dirs: torch.Tensor,
        seq_len: int,
        dt: float,
        K: int,
    ):
        self.seq_len = int(seq_len)
        if self.seq_len <= 0:
            raise ValueError(f"seq_len must be > 0, got {self.seq_len}")
        self.dt = float(dt)
        self.K = int(K)

        bins = _time_to_bin_idx(times[: self.seq_len], self.dt)
        dirs_ = dirs[: self.seq_len].to(dtype=torch.long)

        # Unique consecutive bins and inverse indices for scatter_add.
        uniq, inv = torch.unique_consecutive(bins, return_inverse=True)
        nseg = int(uniq.numel())
        up_counts = torch.zeros((nseg,), device=times.device, dtype=torch.float)
        down_counts = torch.zeros((nseg,), device=times.device, dtype=torch.float)
        up_counts.scatter_add_(0, inv, (dirs_ == UPLOAD).float())
        down_counts.scatter_add_(0, inv, (dirs_ == DOWNLOAD).float())

        self._pkt_bins = uniq
        self._up_counts = up_counts
        self._down_counts = down_counts

        self.p = 0
        self.last_bin: int | None = None
        self.last_was_silence = False
        # Maps start_bin -> end_bin (end_bin > start_bin). Used to clamp packets
        # that would occur during a delay window to the window's right edge.
        self._block_map: dict[int, int] = {}
        self.next_pkt_bin: int | None = self._peek_pkt_bin()
        self.silence_next: int | None = None

    def apply_delay_bins(self, start_bin: int, shift_bins: int) -> None:
        """Block packets in [start_bin, start_bin+shift_bins) and clamp to end."""
        if shift_bins <= 0:
            return

        start_bin = int(start_bin)
        end_bin = start_bin + int(shift_bins)

        # Populate mapping for all bins in the interval.
        for b in range(start_bin, end_bin):
            if b in self._block_map:
                end_bin = max(end_bin, self._block_map[b])
            self._block_map[b] = end_bin

        # Refresh cached next bin and silence schedule.
        self.next_pkt_bin = self._peek_pkt_bin()
        if self.last_bin is None or self.next_pkt_bin is None:
            self.silence_next = None
            return

        # Preserve silence cadence after delay updates.
        cand = self.last_bin + (self.K if self.last_was_silence else 1)
        self.silence_next = cand if cand < self.next_pkt_bin else None

    def _map_bin(self, b: int) -> int:
        # Follow block map chains (e.g. consecutive delays).
        b2 = int(b)
        while b2 in self._block_map:
            b2 = self._block_map[b2]
        return b2

    def _peek_pkt_bin(self) -> int | None:
        if self.p >= int(self._pkt_bins.numel()):
            return None
        return self._map_bin(int(self._pkt_bins[self.p].item()))

    def _consume_pkt_bin(self, b: int) -> tuple[float, float]:
        # Consume one or more original bins that map to the same output bin b.
        if self.p >= int(self._pkt_bins.numel()):
            raise StopIteration

        up = 0.0
        down = 0.0
        while self.p < int(self._pkt_bins.numel()):
            bb = self._map_bin(int(self._pkt_bins[self.p].item()))
            if bb != b:
                break
            up += float(self._up_counts[self.p].item())
            down += float(self._down_counts[self.p].item())
            self.p += 1

        self.next_pkt_bin = self._peek_pkt_bin()

        # If there is a gap to the next packet bin, schedule silence windows.
        if self.next_pkt_bin is None:
            self.silence_next = None
        else:
            gap = self.next_pkt_bin - b - 1
            if gap >= 1:
                self.silence_next = b + 1
            else:
                self.silence_next = None

        return up, down

    def _peek_next_bin(self) -> int | None:
        if self.next_pkt_bin is None:
            return None
        if self.silence_next is not None and self.silence_next < self.next_pkt_bin:
            return self.silence_next
        return self.next_pkt_bin

    def step(self) -> tuple[int, float, float, int | None]:
        """Return (bin, up_count, down_count, next_bin_or_none) and advance."""
        if (b := self._peek_next_bin()) is None:
            raise StopIteration

        # Silence window.
        if self.silence_next is not None and self.next_pkt_bin is not None:
            if b == self.silence_next and b < self.next_pkt_bin:
                # Schedule next silence step at +K, but stop before next packet bin.
                self.silence_next = b + self.K
                if self.silence_next >= self.next_pkt_bin:
                    self.silence_next = None
                self.last_bin = b
                self.last_was_silence = True
                return b, 0.0, 0.0, self._peek_next_bin()

        # Packet bin.
        up, down = self._consume_pkt_bin(b)
        self.last_bin = b
        self.last_was_silence = False
        return b, up, down, self._peek_next_bin()


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
        extend_end_s: float = 0,
    ):
        if dt <= 0:
            raise ValueError(f"dt must be > 0, got {dt}")

        ratio = max_silence_s / dt
        if abs(ratio - round(ratio)) > 1e-8:
            raise ValueError(
                f"max_silence_s must be divisible by dt (max_silence_s={max_silence_s}, dt={dt})."
            )

        if set(X.keys()) > {Feats.PADDING, Feats.DIRS, Feats.TIMES}:
            raise ValueError("Invalid set of feats")

        self.dt = float(dt)
        self.max_silence_s = float(max_silence_s)
        self.features = features

        self.X = {k: v.clone() for k, v in X.items()}
        device = self.X[Feats.TIMES].device

        # Convert float times to int bins (bin space throughout)
        self.X[Feats.TIMES] = _time_to_bin_idx(self.X[Feats.TIMES], self.dt)

        if extend_end_s > 0:
            bs, L = self.X[Feats.DIRS].shape
            mask = self.X[Feats.DIRS] == 0
            seq_lens = (~mask).sum(dim=1)
            col_idx = seq_lens[seq_lens != L]
            row_idx = torch.arange(bs, device=device)[seq_lens != L]
            self.X[Feats.DIRS][row_idx, col_idx] = UPLOAD
            # Extend end in bin space
            extend_end_bins = int(round(extend_end_s / self.dt))
            self.X[Feats.TIMES][mask] += extend_end_bins

        # K in bin index space.
        K = int(self.max_silence_s / self.dt) if self.dt > 0 else 1
        self.K = max(K, 1)

        dirs = self.X[Feats.DIRS]
        self.pkt_seq_lens = (dirs != 0).sum(dim=1).long()
        if (self.pkt_seq_lens <= 0).any():
            bad = torch.where(self.pkt_seq_lens <= 0)[0]
            raise ValueError(
                f"Found empty traces (seq_len=0) in WindowFeatureStreamer: n={int(bad.numel())}."
            )
        self.bs = int(dirs.shape[0])
        # Store as int bins (not float)
        self.dtype = torch.long

        self._cursors: list[_TraceWindowCursor] = []
        for i in range(self.bs):
            self._cursors.append(
                _TraceWindowCursor(
                    times=self.X[Feats.TIMES][i],
                    dirs=self.X[Feats.DIRS][i],
                    seq_len=int(self.pkt_seq_lens[i].item()),
                    dt=self.dt,
                    K=self.K,
                )
            )

        self.done = torch.zeros((self.bs,), device=device, dtype=torch.bool)

    def active_mask(self) -> torch.Tensor:
        return ~self.done

    def step(self) -> dict[Feats, torch.Tensor]:
        device = self.done.device
        bs = self.bs

        emit_bins = Feats.WINDOW_BINS in self.features
        bins = None
        if emit_bins:
            bins = torch.zeros((bs, 1), device=device, dtype=torch.long)
        # Emit int bins (not float times)
        times = torch.full((bs, 1), -1, device=device, dtype=torch.long)  # -1 = invalid
        up = torch.full((bs, 1), -1, device=device, dtype=torch.long)
        down = torch.full((bs, 1), -1, device=device, dtype=torch.long)
        dts = torch.full((bs, 1), -1, device=device, dtype=torch.long)  # bin count

        for i in range(bs):
            if self.done[i]:
                continue

            try:
                b, u, d, b_next = self._cursors[i].step()
            except StopIteration:
                self.done[i] = True
                continue

            if emit_bins:
                assert bins is not None
                bins[i, 0] = int(b)
            times[i, 0] = b  # int bin index
            up[i, 0] = u
            down[i, 0] = d

            if b_next is None:
                dts[i, 0] = 1  # 1 bin default
                self.done[i] = True
            else:
                dts[i, 0] = b_next - b  # bin difference

        fd: dict[Feats, torch.Tensor] = {
            Feats.UP_COUNT: up,
            Feats.DOWN_COUNT: down,
            Feats.Dt: dts,
            Feats.TIMES: times,
        }

        if emit_bins:
            assert bins is not None
            fd[Feats.WINDOW_BINS] = bins

        if Feats.SILENCE_FLAG in self.features:
            # Match get_window_feature_dict(): SILENCE_FLAG is computed after the
            # final flush-left, so NaN UP/DOWN in padded positions become 0.0.
            fd[Feats.SILENCE_FLAG] = ((up == 0) & (down == 0)).float()

        if not all(f in fd for f in self.features):
            raise ValueError("Some requested features are missing!")

        return {f: fd[f] for f in self.features}

    def apply_delay(self, start_s: torch.Tensor, delay_s: torch.Tensor) -> None:
        """Apply a delay window [start_s, start_s+delay_s) per trace.

        Packets that would occur within the delay window are clamped to the right
        edge (start_s+delay_s). Values must be multiples of dt.
        """
        start_s = _as_batch_vec(start_s, "start_s")
        delay_s = _as_batch_vec(delay_s, "delay_s")

        if start_s.shape[0] != self.bs or delay_s.shape[0] != self.bs:
            raise ValueError("start_s/delay_s batch mismatch")

        if ((delay_s > 0) & ~start_s.isfinite()).any():
            raise ValueError("start_s must be finite when delay_s > 0")

        start_bins = _boundary_time_to_bin_idx(start_s, self.dt)
        shift_bins = _duration_to_bin_offsets(delay_s, self.dt)

        self.apply_delay_bins(start_bins, shift_bins)

    def apply_delay_bins(
        self, start_bins: torch.Tensor, shift_bins: torch.Tensor
    ) -> None:
        """Apply a delay window [start_bin, start_bin+shift_bins) per trace."""
        start_bins = _as_batch_vec(start_bins, "start_bins")
        shift_bins = _as_batch_vec(shift_bins, "shift_bins")

        if start_bins.shape[0] != self.bs or shift_bins.shape[0] != self.bs:
            raise ValueError("start_bins/shift_bins batch mismatch")

        start_bins = start_bins.to(torch.long)
        shift_bins = shift_bins.to(torch.long)

        for i in range(self.bs):
            if self.done[i]:
                continue
            sb = int(start_bins[i].item())
            sh = int(shift_bins[i].item())
            if sh <= 0:
                continue
            self._cursors[i].apply_delay_bins(sb, sh)
