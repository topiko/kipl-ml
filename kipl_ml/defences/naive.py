"""
Naive dummy defences for testing.
"""

import os

import kipl_ml.data.assets as assets
import torch
from kipl_ml.defences.base import _Def
from torch.distributions.chi2 import Chi2


class RandomPadding(_Def):
    def __init__(self, fraction: float):
        self.fraction = fraction

    def report(self, to_log: bool = True) -> str:
        return self._report(to_log, fraction=self.fraction)

    def _simulate(self, trace_path: os.PathLike) -> dict[Feats, torch.Tensor]:

        trace = self.load_data(trace_path)

        times = trace[assets.TIMES]
        dirs = trace[assets.DIRS]
        sizes = trace[assets.SIZES]

        device = times.device

        n_pad_packets = int(len(dirs) * self.fraction)

        t0 = times.min()
        t1 = times.max()

        pad_packet_times = torch.rand(n_pad_packets, device=device) * (t1 - t0) + t0
        pad_packet_dirs = torch.randint(0, 2, size=(n_pad_packets,), device=device) - 1
        pad_packet_sizes = torch.ones_like(pad_packet_times, device=device)

        times = torch.cat([times, pad_packet_times])
        dirs = torch.cat([dirs, pad_packet_dirs])
        sizes = torch.cat([sizes, pad_packet_sizes])
        orig_packets = torch.cat(
            [
                torch.ones(len(dirs), dtype=torch.bool, device=device),
                torch.zeros(n_pad_packets, dtype=torch.bool, device=device),
            ]
        )

        times, sort_index = torch.sort(times)
        dirs = dirs[sort_index]
        sizes = sizes[sort_index]
        orig_packets = orig_packets[sort_index]

        return {
            assets.TIMES: times,
            assets.DIRS: dirs,
            assets.SIZES: sizes,
            assets.DECOY: ~orig_packets,
        }


class Chi2Delays(_Def):
    def __init__(self, k: int = 1, scale_frac: float = 0.1):
        if k <= 0:
            self.identity = True
        else:
            self.identity = False
            self.chi2 = Chi2(k)
        self.scale_frac = scale_frac
        self.k = k

    def report(self, to_log: bool = True) -> str:
        return self._report(to_log, k=self.k, scale_frac=self.scale_frac)

    def _simulate(self, trace_path: os.PathLike) -> dict[Feats, torch.Tensor]:

        trace = self.load_data(trace_path)

        if self.identity:
            return trace

        times = trace[assets.TIMES]

        scale_ns = torch.diff(times).mean().item()
        device = times.device

        n_packets = len(times)
        delays = (
            self.chi2.sample((n_packets,)).to(device=device)
            * scale_ns
            * self.scale_frac
        )

        times += torch.cumsum(delays, dim=0)

        trace[assets.TIMES] = times

        return trace
