import multiprocessing
import os
import random
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch.utils.data import ConcatDataset, Dataset, Subset, get_worker_info

from kipl_ml.data.loading import default_num_workers, make_dataloader
from kipl_ml.defences.base import NoDefence
from kipl_ml.defences.pool import DefencePool
from kipl_ml.network.network import NetworkContext
from kipl_ml.tools.rng_samplers import MachineRng


class ProbeDefence(NoDefence):
    def __init__(self):
        super().__init__(seed=42)
        self.machine_rng = MachineRng(1000000, seed=42)
        self.rng = np.random.default_rng(42)  # Neural-defence league selection.


class RngProbeDataset(Dataset):
    """Exercise the actual pool/simulation/deck/network RNG owners without Rust."""

    def __init__(self):
        children = [ProbeDefence() for _ in range(2)]
        nested = DefencePool(children, seed=42)
        self.defence = DefencePool([nested, NoDefence()], seed=42)
        self.network_context = NetworkContext(
            "tor_custom", (5, 100), (10, 1000), seed=42, tor_profile=1
        )
        self.marker = 0

    def __len__(self):
        return 32

    def __getitem__(self, index):
        nested = self.defence.defences[0]
        assert isinstance(nested, DefencePool)
        values = [self.defence.defence_rng(), nested.defence_rng()]
        for child in nested.defences:
            assert isinstance(child, ProbeDefence)
            values.extend([child.simul_rng(), child.machine_rng(), child.rng.random()])
        network = self.network_context.sample_params()
        values.extend(network[key] for key in (
            "network_rtt_millis", "network_mbps", "tor_e2e_rtt_us", "tor_e2e_tput_bps"
        ))
        values.extend([random.random(), np.random.random(), torch.rand(()).item()])
        worker = get_worker_info()
        return {
            "draws": torch.tensor(values, dtype=torch.float64),
            "worker": worker.id if worker is not None else -1,
            "threads": torch.get_num_threads(),
            "marker": self.marker,
        }


def collect(loader):
    batches = list(loader)
    return {key: torch.cat([batch[key] for batch in batches]) for key in batches[0]}


class TestWorkerStreams(unittest.TestCase):
    def test_workers_and_epochs_are_independent_and_runs_reproducible(self):
        for context in ("fork", "spawn"):
            if context not in multiprocessing.get_all_start_methods():
                continue
            with self.subTest(context=context):
                repetitions = []
                for _ in range(2):
                    dataset = RngProbeDataset()
                    loader = make_dataloader(
                        Subset(dataset, range(len(dataset))), batch_size=2,
                        num_workers=2, multiprocessing_context=context,
                        generator=torch.Generator().manual_seed(1234),
                    )
                    first = collect(loader)
                    dataset.marker = 7
                    second = collect(loader)
                    self.assertTrue(torch.all(first["threads"] == 1))
                    self.assertTrue(torch.all(first["marker"] == 0))
                    self.assertTrue(torch.all(second["marker"] == 7))
                    self.assertFalse(torch.equal(first["draws"], second["draws"]))
                    a = first["draws"][first["worker"] == 0]
                    b = first["draws"][first["worker"] == 1]
                    for column in range(a.shape[1]):
                        self.assertFalse(torch.equal(a[:, column], b[:, column]))
                    # Two identically initialized child defences get distinct streams.
                    self.assertFalse(torch.equal(a[:, 2], a[:, 5]))
                    repetitions.append((first["draws"], second["draws"]))
                for first, repeated in zip(*repetitions):
                    torch.testing.assert_close(first, repeated, rtol=0, atol=0)

    def test_concat_children_and_persistent_workers_keep_advancing(self):
        dataset = ConcatDataset([RngProbeDataset(), RngProbeDataset()])
        loader = make_dataloader(
            dataset, batch_size=2, num_workers=2, persistent_workers=True,
            generator=torch.Generator().manual_seed(12),
        )
        first, second = collect(loader), collect(loader)
        self.assertFalse(torch.equal(first["draws"], second["draws"]))
        self.assertFalse(torch.equal(first["draws"][:32], first["draws"][32:]))

    def test_zero_workers_does_not_run_initializer_or_replace_private_rng(self):
        dataset = RngProbeDataset()
        original = dataset.defence.defence_rng.rng
        with patch("kipl_ml.data.loading.seed_worker") as initialize:
            loader = make_dataloader(dataset, batch_size=2, num_workers=0)
            rows = collect(loader)
        initialize.assert_not_called()
        self.assertIs(dataset.defence.defence_rng.rng, original)
        self.assertTrue(torch.all(rows["worker"] == -1))

    @unittest.skipUnless(hasattr(os, "sched_getaffinity"), "Linux affinity")
    def test_automatic_workers_respect_affinity_and_cap(self):
        with patch("os.sched_getaffinity", return_value={2, 3}):
            self.assertEqual(default_num_workers(), 2)
        with patch("os.sched_getaffinity", return_value=set(range(128))):
            self.assertEqual(default_num_workers(), 24)


if __name__ == "__main__":
    unittest.main()
