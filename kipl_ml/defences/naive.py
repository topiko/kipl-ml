"""
Naive dummy defences for testing.
"""

import kipl_ml.assets as assets
import torch
from kipl_ml.defences.base import _Def


class RandomPadding(_Def):
    def __init__(self, fraction: float):
        self.fraction = fraction

    def report(self, to_log: bool = True) -> str:
        return self._report(to_log, fraction=self.fraction)

    def __call__(self, trace: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        times = trace[assets.TIMES]
        packets = trace[assets.PACKETS]
        sizes = trace[assets.SIZES]

        device = times.device

        n_pad_packets = int(packets * self.fraction)

        t0 = times.min()
        t1 = times.max()

        pad_packet_times = torch.randn(n_pad_packets, device=device) * (t1 - t0) + t0
        pad_packet_dirs = torch.randint(0, 2, size=(n_pad_packets,), device=device) - 1
        pad_packet_sizes = torch.ones_like(pad_packet_times, device=device)

        times = torch.cat([times, pad_packet_times])
        packets = torch.cat([packets, pad_packet_dirs])
        sizes = torch.cat([sizes, pad_packet_sizes])
        orig_packets = torch.cat(
            [
                torch.ones(len(packets), dtype=torch.bool, device=device),
                torch.zeros(n_pad_packets, dtype=torch.bool, device=device),
            ]
        )

        times, sort_index = torch.sort(times)
        packets = packets[sort_index]
        sizes = sizes[sort_index]
        orig_packets = orig_packets[sort_index]

        return {
            assets.TIMES: times,
            assets.PACKETS: packets,
            assets.SIZES: sizes,
            assets.ORIG_PACKETS: orig_packets,
        }
