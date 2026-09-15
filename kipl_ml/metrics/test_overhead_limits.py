import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import cast
from unittest.mock import patch

import pandas as pd

from kipl_ml.data.assets import DATASET, TRACE_F_PATH
from kipl_ml.data.wf_dataset import WFDataset
from kipl_ml.metrics import defence_netwk_metrics as metrics


class TestOverheadLimits(unittest.TestCase):
    def test_evaluation_can_preserve_policy_limits_and_trim(self):
        observed = []

        class Defence:
            _n_packets = 123
            _max_dur_s = 7.0

            def __call__(self, _path, _machine, *, trim_raw, network_context):
                observed.append((self._n_packets, self._max_dur_s, trim_raw))
                return {}

        class Dataset:
            defence = Defence()
            trim_raw = 3
            meta_df = pd.DataFrame(
                {DATASET: ["fixture"], TRACE_F_PATH: ["/raw/fixture/trace.log"]}
            )

            def clone(self, **kwargs):
                result = Dataset()
                result.__dict__.update(kwargs)
                return result

        class Info:
            def __init__(self, dataset):
                self.base = dataset

            def get_meta(self, index):
                return self.base.meta_df.iloc[index]

        with (
            patch.object(metrics, "RNNDef", Defence),
            patch.object(metrics, "InformativeDataset", Info),
            patch.object(
                metrics, "DataLoader", return_value=[[("original", 0, {})]]
            ) as loader,
            patch.object(metrics, "tensor_dict_to_str", return_value="defended"),
        ):
            for preserve in (False, True):
                with TemporaryDirectory() as original, TemporaryDirectory() as defended:
                    metrics._make_tmp(
                        cast(WFDataset, Dataset()),
                        Path(original),
                        Path(defended),
                        1.0,
                        preserve_defence_limits=preserve,
                        num_workers=0,
                    )
            self.assertEqual(loader.call_args.kwargs["num_workers"], 0)
        self.assertEqual(observed, [(100_000_000, 100_000, 0), (123, 7.0, 3)])
        self.assertEqual(Dataset.defence._n_packets, 123)
