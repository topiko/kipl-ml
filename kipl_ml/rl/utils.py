from __future__ import annotations

import torch


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


def _time_to_bin_idx(times: torch.Tensor, dt: float) -> torch.Tensor:
    """Map seconds to integer bins with microsecond quantization."""
    if dt <= 0:
        raise ValueError(f"dt must be > 0, got {dt}")

    dt_us = max(1, int(round(float(dt) * 1e6)))
    t_us = torch.round(times * 1e6).to(torch.long)
    return torch.div(t_us, dt_us, rounding_mode="floor")


def _boundary_time_to_bin_idx(times: torch.Tensor, dt: float) -> torch.Tensor:
    """Map boundary times (e.g., action times) to integer bins."""
    if dt <= 0:
        raise ValueError(f"dt must be > 0, got {dt}")
    return torch.round(times.to(torch.float64) / float(dt)).to(torch.long)


def _duration_to_bin_offsets(durations: torch.Tensor, dt: float) -> torch.Tensor:
    """Map durations to bin offsets with microsecond quantization and rounding."""
    if dt <= 0:
        raise ValueError(f"dt must be > 0, got {dt}")
    dt_us = max(1, int(round(float(dt) * 1e6)))
    d_us = torch.round(durations * 1e6).to(torch.long)
    return torch.div(d_us + (dt_us // 2), dt_us, rounding_mode="floor")


def bins_to_seconds(bins: torch.Tensor, dt: float) -> torch.Tensor:
    """Convert integer bin indices to seconds.
    
    Args:
        bins: Integer bin indices (can be negative for sentinel values)
        dt: Time step in seconds
        
    Returns:
        Float tensor with times in seconds (sentinel values preserved)
    """
    if dt <= 0:
        raise ValueError(f"dt must be > 0, got {dt}")
    return bins.to(torch.float64) * float(dt)
