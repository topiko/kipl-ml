from __future__ import annotations

import torch

from kipl_ml.data.utils import DOWNLOAD, UPLOAD
from kipl_ml.logging.logger import get_logger
from kipl_ml.rl.utils import _fill_after_seq_end, _flush_left
from kipl_ml.trace.enums import Feats

logger = get_logger(__name__)


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

    total = int(count.sum().item())
    if total == 0:
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
    bin_idx = (times // dt).long()

    if bin_idx.min() < 0:
        raise ValueError("Negative bin indices found!")

    feature_dict: dict[Feats, torch.Tensor] = {}
    device = times.device
    shape = (times.shape[0], bin_idx.max() + 1)
    bs = shape[0]

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

    # Keep TIMES as bin indices (float) with NaNs after seq end.
    times_bins = _fill_after_seq_end(times_idx, pad_val=0)
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
            print(f, k)
            print(feature_dict[f].sum(dim=1), (X[Feats.DIRS] == k).sum(dim=1))
            breakpoint()
            raise ValueError("Missing packets")

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
        self.times = times
        self.dirs = dirs
        self.seq_len = int(seq_len)
        self.dt = float(dt)
        self.K = int(K)

        self.p = 0
        self.last_bin: int | None = None
        self.offset_bins: int = 0
        self.next_pkt_bin: int | None = self._peek_pkt_bin()
        self.silence_next: int | None = None

    def apply_delay_bins(self, shift_bins: int) -> None:
        if shift_bins <= 0:
            return

        # Delay shifts all *future* bins; modelled as a bin offset.
        self.offset_bins += int(shift_bins)
        self.next_pkt_bin = self._peek_pkt_bin()

        # Recompute silence scheduling relative to the last produced bin.
        if self.last_bin is None or self.next_pkt_bin is None:
            self.silence_next = None
            return

        gap = self.next_pkt_bin - self.last_bin - 1
        if gap >= 1:
            self.silence_next = self.last_bin + 1
        else:
            self.silence_next = None

    def _peek_pkt_bin(self) -> int | None:
        if self.p >= self.seq_len:
            return None
        return int((self.times[self.p] // self.dt).item()) + self.offset_bins

    def _consume_pkt_bin(self, b: int) -> tuple[float, float]:
        up = 0.0
        down = 0.0
        while self.p < self.seq_len:
            bb = int((self.times[self.p] // self.dt).item()) + self.offset_bins
            if bb != b:
                break

            d = int(self.dirs[self.p].item())
            if d == UPLOAD:
                up += 1.0
            elif d == DOWNLOAD:
                down += 1.0
            else:
                raise ValueError("Nonzero dirs expected within seq_len")
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
        b = self._peek_next_bin()
        if b is None:
            raise StopIteration

        # Silence window.
        if self.silence_next is not None and self.next_pkt_bin is not None:
            if b == self.silence_next and b < self.next_pkt_bin:
                # Schedule next silence step at +K, but stop before next packet bin.
                self.silence_next = b + self.K
                if self.silence_next >= self.next_pkt_bin:
                    self.silence_next = None
                self.last_bin = b
                return b, 0.0, 0.0, self._peek_next_bin()

        # Packet bin.
        up, down = self._consume_pkt_bin(b)
        self.last_bin = b
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

        if extend_end_s > 0:
            bs, L = self.X[Feats.DIRS].shape
            mask = self.X[Feats.DIRS] == 0
            seq_lens = (~mask).sum(dim=1)
            col_idx = seq_lens[seq_lens != L]
            row_idx = torch.arange(bs, device=device)[seq_lens != L]
            self.X[Feats.DIRS][row_idx, col_idx] = UPLOAD
            self.X[Feats.TIMES][mask] += extend_end_s

        # K in bin index space.
        K = int(self.max_silence_s / self.dt) if self.dt > 0 else 1
        self.K = max(K, 1)

        dirs = self.X[Feats.DIRS]
        self.pkt_seq_lens = (dirs != 0).sum(dim=1).long()
        self.bs = int(dirs.shape[0])
        self.dtype = self.X[Feats.TIMES].dtype

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

        times = torch.full((bs, 1), torch.nan, device=device, dtype=self.dtype)
        up = torch.full((bs, 1), torch.nan, device=device, dtype=self.dtype)
        down = torch.full((bs, 1), torch.nan, device=device, dtype=self.dtype)
        dts = torch.full((bs, 1), torch.nan, device=device, dtype=self.dtype)

        for i in range(bs):
            if self.done[i]:
                continue

            try:
                b, u, d, b_next = self._cursors[i].step()
            except StopIteration:
                self.done[i] = True
                continue

            times[i, 0] = float(b) * self.dt
            up[i, 0] = u
            down[i, 0] = d

            if b_next is None:
                dts[i, 0] = self.dt
                self.done[i] = True
            else:
                dts[i, 0] = float(b_next - b) * self.dt

        fd: dict[Feats, torch.Tensor] = {
            Feats.UP_COUNT: up,
            Feats.DOWN_COUNT: down,
            Feats.Dt: dts,
            Feats.TIMES: times,
        }

        if Feats.SILENCE_FLAG in self.features:
            # Match get_window_feature_dict(): SILENCE_FLAG is computed after the
            # final flush-left, so NaN UP/DOWN in padded positions become 0.0.
            fd[Feats.SILENCE_FLAG] = ((up == 0) & (down == 0)).float()

        if not all(f in fd for f in self.features):
            raise ValueError("Some requested features are missing!")

        return {f: fd[f] for f in self.features}

    def apply_delay(self, delay_s: torch.Tensor) -> None:
        """Apply a delay (seconds) to future windows for each trace.

        delay_s is (B,) or (B, 1). Values must be non-negative and multiples of dt.
        """
        if delay_s.ndim == 2:
            if delay_s.shape[1] != 1:
                raise ValueError("delay_s must be (B,) or (B,1)")
            delay_s = delay_s.squeeze(1)

        if delay_s.shape[0] != self.bs:
            raise ValueError("delay_s batch mismatch")

        ratio = delay_s / self.dt
        ratio_r = ratio.round()
        if (delay_s > 0).any() and ((ratio - ratio_r).abs().max().item() > 1e-6):
            raise ValueError("delay_s must be multiple of dt")

        shift_bins = ratio_r.to(torch.long)
        for i in range(self.bs):
            if self.done[i]:
                continue
            sb = int(shift_bins[i].item())
            if sb <= 0:
                continue
            self._cursors[i].apply_delay_bins(sb)
