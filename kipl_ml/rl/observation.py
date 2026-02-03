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
