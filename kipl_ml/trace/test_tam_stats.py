import unittest

import torch

from kipl_ml.trace.enums import Feats
from kipl_ml.trace.features import FeatureTrs, get_feature_tr


def extract(trace, width=1.0, horizon=2.0, **kwargs):
    return FeatureTrs(
        feature_trs=[
            get_feature_tr(
                feature,
                None,
                tam_kwargs={
                    "window_width_s": width,
                    "max_load_time_s": horizon,
                    **kwargs,
                },
            )
            for feature in (
                Feats.TAM_UP_COUNTS,
                Feats.TAM_DOWN_COUNTS,
                Feats.TAM_UP_TIME_STD,
                Feats.TAM_DOWN_TIME_STD,
            )
        ]
    )(trace)


class TestTamTimingStats(unittest.TestCase):
    def test_direction_masks_padding_decoys_and_right_edge_match_counts(self):
        trace = {
            Feats.TIMES: torch.tensor(
                [-0.1, 0.25, 0.75, 0.1, 0.9, 1.25, 2.5, 3.0, 3.001, -1.0],
                dtype=torch.float64,
            ),
            Feats.DIRS: torch.tensor([1, 1, 1, -1, -1, 1, -1, -1, 1, 0]),
            Feats.DECOY: torch.tensor([0, 0, 1, 0, 0, 0, 0, 0, 0, 0]),
        }
        result = extract(trace)
        # Legacy TAM grid extends to 3 s for horizon=2, width=1; its right edge
        # is included in the final bin. The decoy at 0.75 contributes to std.
        expected = {
            Feats.TAM_UP_COUNTS: [2, 1, 0],
            Feats.TAM_DOWN_COUNTS: [2, 0, 2],
            Feats.TAM_UP_TIME_STD: [0.25, 0, 0],
            Feats.TAM_DOWN_TIME_STD: [0.4, 0, 0.25],
        }
        for feature, values in expected.items():
            torch.testing.assert_close(result[feature], torch.tensor(values).float())
        del trace[Feats.DECOY]
        for feature, values in extract(trace).items():
            torch.testing.assert_close(values, result[feature])

    def test_normalization_is_invariant_to_time_scale(self):
        for width in (0.02, 0.1, 1.0):
            with self.subTest(width=width):
                result = extract(
                    {
                        Feats.TIMES: torch.tensor(
                            [0.1, 0.9, 1.25, 1.75], dtype=torch.float64
                        )
                        * width,
                        Feats.DIRS: torch.ones(4),
                    },
                    width=width,
                    horizon=2 * width,
                )
                torch.testing.assert_close(
                    result[Feats.TAM_UP_TIME_STD][:2], torch.tensor([0.4, 0.25])
                )

    def test_empty_singleton_and_identical_timestamps_are_finite_zero(self):
        for times in ([], [0.1], [0.1, 0.1]):
            with self.subTest(times=times):
                result = extract(
                    {
                        Feats.TIMES: torch.tensor(times),
                        Feats.DIRS: torch.ones(len(times)),
                    }
                )
                for feature in (Feats.TAM_UP_TIME_STD, Feats.TAM_DOWN_TIME_STD):
                    self.assertTrue(torch.equal(result[feature], torch.zeros(3)))

    def test_timing_distinguishes_traces_with_identical_counts_without_future_leakage(
        self,
    ):
        dirs = torch.ones(2)
        spread = extract({Feats.TIMES: torch.tensor([0.1, 0.9]), Feats.DIRS: dirs})
        burst = extract({Feats.TIMES: torch.tensor([0.1, 0.2]), Feats.DIRS: dirs})
        torch.testing.assert_close(
            spread[Feats.TAM_UP_COUNTS], burst[Feats.TAM_UP_COUNTS]
        )
        self.assertGreater(
            spread[Feats.TAM_UP_TIME_STD][0], burst[Feats.TAM_UP_TIME_STD][0]
        )
        future = extract(
            {
                Feats.TIMES: torch.tensor([0.1, 0.9, 1.1, 1.9]),
                Feats.DIRS: torch.ones(4),
            }
        )
        self.assertEqual(
            future[Feats.TAM_UP_TIME_STD][0], spread[Feats.TAM_UP_TIME_STD][0]
        )

    def test_pruning_retains_singleton_bins_and_small_spread_is_stable(self):
        result = extract(
            {
                Feats.TIMES: torch.tensor(
                    [60.0, 60.000001, 60.15], dtype=torch.float64
                ),
                Feats.DIRS: torch.ones(3),
            },
            width=0.1,
            horizon=61,
            prune_empty_bins=True,
        )
        torch.testing.assert_close(
            result[Feats.TAM_UP_TIME_STD],
            torch.tensor([5e-6, 0]),
            atol=1e-10,
            rtol=1e-5,
        )
        torch.testing.assert_close(
            result[Feats.TAM_UP_COUNTS], torch.tensor([2.0, 1.0])
        )

    def test_microsecond_boundary_quantization_agrees_with_count_bins(self):
        result = extract(
            {
                Feats.TIMES: torch.tensor([0.9999996, 1.5], dtype=torch.float64),
                Feats.DIRS: torch.ones(2),
            }
        )
        torch.testing.assert_close(
            result[Feats.TAM_UP_COUNTS], torch.tensor([0.0, 2.0, 0.0])
        )
        torch.testing.assert_close(
            result[Feats.TAM_UP_TIME_STD], torch.tensor([0.0, 0.2500002, 0.0])
        )
