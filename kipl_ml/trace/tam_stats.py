"""Optional within-window timing statistics on the existing TAM grid."""

from typing import Any

import torch

from kipl_ml.trace.enums import Feats
from kipl_ml.trace.features import _TAM
from kipl_ml.trace.params import DOWNLOAD, UPLOAD

TAM_TIME_STD_FEATURES = (Feats.TAM_UP_TIME_STD, Feats.TAM_DOWN_TIME_STD)


def _population_std_by_bin(
    times: torch.Tensor, bins: torch.Tensor, n_bins: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Two-pass, float64 variance; empty and singleton bins have zero std."""
    counts = torch.bincount(bins, minlength=n_bins)
    values = times.double()
    if values.numel():
        values = values - values[0]
    sums = values.new_zeros(n_bins).scatter_add_(0, bins, values)
    means = sums / counts.clamp_min(1)
    squared_deviations = (values - means[bins]).square()
    variance = values.new_zeros(n_bins).scatter_add_(0, bins, squared_deviations)
    variance = variance / counts.clamp_min(1)
    return counts, variance.sqrt()


class TAM_TIME_STD(_TAM):
    """Population timestamp std / window width, for all visible packets on a side.

    Uses the count transform's packet masks, microsecond binning, and right-edge
    handling. This describes temporal spread, not inter-arrival-time variation.
    No decoy labels are needed: decoys contribute just like normal packets.
    """

    TIMES = False
    DECOY = False

    def __init__(self, feature: Feats, **kwargs: Any):
        if feature not in TAM_TIME_STD_FEATURES:
            raise ValueError(f"Unsupported TAM timing feature: {feature}")
        self.feature = feature
        self.DIR = "upload" if feature == Feats.TAM_UP_TIME_STD else "download"
        super().__init__(**kwargs)

    @property
    def name(self) -> Feats:
        return self.feature

    def __call__(self, trace: dict[Feats, torch.Tensor]) -> dict[Feats, torch.Tensor]:
        direction = UPLOAD if self.DIR == "upload" else DOWNLOAD
        times, bins = self._packets_by_bin_idx(
            trace[Feats.TIMES], trace[Feats.DIRS] == direction
        )
        counts, std = _population_std_by_bin(times, bins, len(self.bins) - 1)
        normalized = (std / self.window_width_s).float()
        return {self.name: normalized[self._get_keep_mask(counts)]}
