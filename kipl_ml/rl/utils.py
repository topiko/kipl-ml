from __future__ import annotations

import warnings

import torch

from kipl_ml.utils.time import (
    _boundary_time_to_bin_idx as _new_boundary_time_to_bin_idx,
    _duration_to_bin_offsets as _new_duration_to_bin_offsets,
    _time_to_bin_idx as _new_time_to_bin_idx,
    bins_to_seconds as _new_bins_to_seconds,
)


def fill_after_seq_end(
    values: torch.Tensor,
    keep_mask: torch.Tensor,
    *,
    fill_val: str = "nan",
) -> torch.Tensor:
    """Fill values after seq end using an explicit mask.

    This avoids relying on a sentinel pad value (e.g. 0), which is ambiguous for
    legitimate features like TIMES where 0 can be a real value.
    """

    if values.shape != keep_mask.shape:
        raise ValueError(
            f"values and keep_mask must match, got {values.shape} and {keep_mask.shape}"
        )

    out = values.clone()
    B, L = out.shape
    seq_lens = keep_mask.sum(dim=1).long()

    if fill_val == "nan":
        fill = torch.full((B,), torch.nan, device=out.device, dtype=out.dtype)
    elif fill_val == "last":
        idx = (seq_lens - 1).clamp(min=0)
        fill = out.gather(1, idx.view(B, 1)).squeeze(1)
    elif fill_val == "max":
        masked = torch.where(
            keep_mask,
            out,
            torch.full((), -torch.inf, device=out.device, dtype=out.dtype),
        )
        fill = masked.max(dim=1).values
        fill = torch.where(seq_lens > 0, fill, torch.zeros_like(fill))
    else:
        raise ValueError(f"Unknown fill_val option: {fill_val}")

    col = torch.arange(L, device=out.device).unsqueeze(0).expand(B, -1)
    tail = col >= seq_lens.unsqueeze(1)
    out[tail] = fill.unsqueeze(1).expand_as(out)[tail]
    return out


def _flush_left(
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


def _time_to_bin_idx(times, dt: float):
    warnings.warn(
        "kipl_ml.rl.utils._time_to_bin_idx is deprecated; use kipl_ml.utils.time._time_to_bin_idx",
        DeprecationWarning,
        stacklevel=2,
    )
    return _new_time_to_bin_idx(times, dt)


def _boundary_time_to_bin_idx(times, dt: float):
    warnings.warn(
        "kipl_ml.rl.utils._boundary_time_to_bin_idx is deprecated; use kipl_ml.utils.time._boundary_time_to_bin_idx",
        DeprecationWarning,
        stacklevel=2,
    )
    return _new_boundary_time_to_bin_idx(times, dt)


def _duration_to_bin_offsets(durations, dt: float):
    warnings.warn(
        "kipl_ml.rl.utils._duration_to_bin_offsets is deprecated; use kipl_ml.utils.time._duration_to_bin_offsets",
        DeprecationWarning,
        stacklevel=2,
    )
    return _new_duration_to_bin_offsets(durations, dt)


def bins_to_seconds(bins, dt: float):
    warnings.warn(
        "kipl_ml.rl.utils.bins_to_seconds is deprecated; use kipl_ml.utils.time.bins_to_seconds",
        DeprecationWarning,
        stacklevel=2,
    )
    return _new_bins_to_seconds(bins, dt)
