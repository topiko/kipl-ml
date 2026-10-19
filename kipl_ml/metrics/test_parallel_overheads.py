import json
import multiprocessing
import os
import random
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from kipl_ml.data.assets import DATASET, PAGE_LABEL, TRACE_F_PATH
from kipl_ml.data.loading import make_dataloader
from kipl_ml.data.wf_dataset import WFDataset
from kipl_ml.defences.base import NoDefence
from kipl_ml.metrics import defence_netwk_metrics as metrics
from kipl_ml.network.network import NetworkContext
from kipl_ml.trace.enums import Feats


class ProbeDefence(NoDefence):
    def __init__(self, fail=False):
        super().__init__()
        self.fail = fail
        self.rng = np.random.default_rng(42)

    def _simulate(self, trace_path, machine_idx=None, trim_raw=0, network_context=None):
        if self.fail:
            raise RuntimeError("simulation failed")
        trace = super()._simulate(trace_path, machine_idx, trim_raw, network_context)
        draws = [random.random(), np.random.random(), torch.rand(()).item(),
                 self.rng.random()]
        trace[Feats.TIMES] *= 1 + sum(draws)
        Path(trace_path).with_suffix(".defended.json").write_text(json.dumps({
            "pid": os.getpid(), "context": network_context,
            "trim": trim_raw, "draws": draws,
        }))
        return trace


class ProbeDataset(WFDataset):
    def _get_trace_context_and_metadata(self, idx):
        trace, context, metadata = super()._get_trace_context_and_metadata(idx)
        path = Path(self.get_meta(idx)[TRACE_F_PATH])
        path.with_suffix(".original.json").write_text(json.dumps({
            "pid": os.getpid(), "context": context, "trim": self.trim_raw,
        }))
        return trace, context, metadata


def fixture(root, count=12, fail=False):
    paths = []
    for index in range(count):
        path = root / "fixture" / "0" / f"{index}.log"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("0,sn,512\n10000000,rn,512\n20000000,sn,512\n")
        paths.append(str(path))
    return ProbeDataset(
        pd.DataFrame({DATASET: ["fixture"] * count, TRACE_F_PATH: paths,
                      PAGE_LABEL: [0] * count}),
        feature_trs=None, defence=ProbeDefence(fail), trim_raw=1,
        network_context=NetworkContext("vpn_custom", (5, 100), (10, 100), 42, 0),
    )


class TestParallelOverheads(unittest.TestCase):
    def test_pairs_and_metrics_match_across_worker_counts_and_spawn(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = fixture(root)
            torch.manual_seed(123)
            reference = None
            reference_metrics = None
            cases = [(0, None), (1, None), (4, None)]
            if "spawn" in multiprocessing.get_all_start_methods():
                cases.append((2, "spawn"))
            for workers, context in cases:
                with self.subTest(workers=workers, context=context):
                    original = root / f"original-{workers}-{context}"
                    defended = root / f"defended-{workers}-{context}"

                    def loader(*args, **kwargs):
                        if context:
                            kwargs["multiprocessing_context"] = context
                        return make_dataloader(*args, **kwargs)

                    with patch.object(metrics, "make_dataloader", side_effect=loader):
                        metrics._make_tmp(dataset, original, defended, 0.75,
                                          preserve_defence_limits=True,
                                          num_workers=workers)
                    files = [{str(p.relative_to(directory)): p.read_bytes()
                              for p in directory.rglob("*.log")}
                             for directory in (original, defended)]
                    self.assertEqual(len(files[0]), 9)
                    self.assertEqual(files[0].keys(), files[1].keys())
                    pids = set()
                    draws = set()
                    for relative in files[0]:
                        path = root / "fixture" / relative
                        a = json.loads(path.with_suffix(".original.json").read_text())
                        b = json.loads(path.with_suffix(".defended.json").read_text())
                        self.assertEqual(a["context"], b["context"])
                        self.assertEqual((a["trim"], b["trim"]), (1, 1))
                        self.assertEqual(a["pid"], b["pid"])
                        pids.add(b["pid"])
                        draws.add(tuple(b["draws"]))
                    self.assertEqual(len(draws), 9)
                    self.assertEqual(len(pids), workers or 1)
                    self.assertEqual(os.getpid() in pids, workers == 0)
                    raw = metrics.compute_overheads(
                        str(original), str(defended), 60000, False,
                    )
                    summary = metrics._format_overheads(raw)
                    if reference is None:
                        reference, reference_metrics = files, summary
                    else:
                        assert reference_metrics is not None
                        self.assertEqual(files, reference)
                        for key, value in summary.items():
                            self.assertAlmostEqual(value, reference_metrics[key])
            # A different caller seed produces a different defended realization.
            torch.manual_seed(124)
            metrics._make_tmp(dataset, root / "other-original", root / "other", 0.75,
                              num_workers=0, preserve_defence_limits=True)
            changed = {str(p.relative_to(root / "other")): p.read_bytes()
                       for p in (root / "other").rglob("*.log")}
            assert reference is not None
            self.assertNotEqual(changed, reference[1])

    def test_serial_execution_does_not_advance_caller_rngs_or_mutate_dataset(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = fixture(root, count=2)
            python_state, numpy_state = random.getstate(), np.random.get_state()
            torch_state = torch.get_rng_state()
            defence_state = dataset.defence.simul_rng.rng.bit_generator.state
            metrics._make_tmp(dataset, root / "original", root / "defended", 1.0,
                              num_workers=0)
            self.assertEqual(random.getstate(), python_state)
            current = np.random.get_state()
            np.testing.assert_equal(current, numpy_state)
            self.assertTrue(torch.equal(torch.get_rng_state(), torch_state))
            self.assertEqual(dataset.defence.simul_rng.rng.bit_generator.state,
                             defence_state)
            self.assertEqual(dataset.trim_raw, 1)
            self.assertEqual(len(dataset), 2)

    def test_worker_failure_prevents_native_aggregation(self):
        with TemporaryDirectory() as tmp:
            dataset = fixture(Path(tmp), fail=True)
            with (
                patch.object(metrics, "compute_overheads") as aggregate,
                self.assertRaisesRegex(RuntimeError, "simulation failed"),
            ):
                metrics.get_overheads(dataset, 1.0, num_workers=2)
            aggregate.assert_not_called()


if __name__ == "__main__":
    unittest.main()
