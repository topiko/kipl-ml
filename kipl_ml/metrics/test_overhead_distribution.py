import math
import unittest

from kipl_ml.metrics.defence_netwk_metrics import _format_overheads


class TestOverheadSummaryMapping(unittest.TestCase):
    def legacy_summary(self):
        return {
            **{
                f"overhead_{kind}_{stat}_multiple": 0.5
                for kind in ("data", "duration", "data_full", "duration_full")
                for stat in ("mean", "median", "std_dev")
            },
            "base_mean_packets": 10.0,
            "missing_mean_packets": 1.0,
        }

    def test_older_bindings_do_not_invent_unavailable_statistics(self):
        result = _format_overheads(self.legacy_summary())
        self.assertEqual(result["def.delay_mean"], 0.5)
        self.assertEqual(result["sim.missing"], 0.1)
        self.assertNotIn("def.delay_min", result)
        self.assertNotIn("def.bandwidth_p95", result)
        self.assertNotIn("def.delay_count", result)

    def test_new_fields_preserve_units_and_empty_is_not_zero_overhead(self):
        raw = self.legacy_summary()
        raw.update(
            overhead_data_p95_multiple=1.75,
            overhead_data_count=20,
            overhead_duration_count=0,
            overhead_duration_min_multiple=float("nan"),
            trace_pairs_count=25,
            trace_pairs_no_normal_count=5,
        )
        result = _format_overheads(raw)
        self.assertEqual(result["def.bandwidth_p95"], 1.75)
        self.assertEqual(result["def.bandwidth_count"], 20)
        self.assertEqual(result["sim.overhead_no_normal"], 5)
        for stat in ("mean", "median", "std", "min"):
            self.assertTrue(math.isnan(result[f"def.delay_{stat}"]))


if __name__ == "__main__":
    unittest.main()
