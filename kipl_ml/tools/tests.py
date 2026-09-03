from __future__ import annotations

import unittest

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

from kipl_ml.tools.plottr import plot_tam  # noqa: E402
from kipl_ml.trace.enums import Feats  # noqa: E402


class TestPlotTam(unittest.TestCase):
    def test_silent_tam_trace_plots(self) -> None:
        fig, ax = plt.subplots()
        try:
            plot_tam(
                {
                    Feats.TAM_UP_COUNTS: torch.zeros((1, 3)),
                    Feats.TAM_DOWN_COUNTS: torch.zeros((1, 3)),
                    Feats.TAM_TIMES: torch.tensor([[0.0, 0.5, 1.0]]),
                    Feats.TAM_UP_DECOY: torch.zeros((1, 3)),
                    Feats.TAM_DOWN_DECOY: torch.zeros((1, 3)),
                },
                window_width=0.5,
                idx=0,
                ax=ax,
            )
        finally:
            plt.close(fig)


if __name__ == "__main__":
    unittest.main()
