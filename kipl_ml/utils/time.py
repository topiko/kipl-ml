from __future__ import annotations

import torch


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
