import json
import unittest
from pathlib import Path

import torch
from torch.utils.data import Dataset, get_worker_info

from kipl_ml.data.loading import make_dataloader
from kipl_ml.data.test_defence_metadata import MarkerDefence, make_dataset
from kipl_ml.data.wf_dataset import InformativeDataset
from kipl_ml.defences.pool import DefencePool, family_probabilities
from kipl_ml.tools.rng_samplers import MachineRng


def pool(probabilities=None, seed=42):
    return DefencePool(
        [MarkerDefence(-1), MarkerDefence(1), MarkerDefence(2)], seed=seed,
        member_ids=["A", "B", "C"], probabilities=probabilities,
    )


class PoolDrawDataset(Dataset):
    def __init__(self):
        self.defence = pool({"A": 0.3, "B": 0.7})

    def __len__(self):
        return 512

    def __getitem__(self, index):
        worker = get_worker_info()
        return self.defence._select_defence_idx(), worker.id if worker else -1


class TestWeightedPool(unittest.TestCase):
    def test_uniform_preserves_legacy_draws_and_explicit_uniform(self):
        rng = MachineRng(3, seed=42)
        expected = [int(rng()) for _ in range(200)]
        for probabilities in (None, "uniform", dict.fromkeys("ABC", 1 / 3)):
            with self.subTest(probabilities=probabilities):
                sampler = pool(probabilities)
                self.assertEqual([sampler._select_defence_idx() for _ in expected],
                                 expected)

    def test_weighted_draws_are_reproducible_id_keyed_and_zero_safe(self):
        first = pool({"C": 0.8, "A": 0.2})
        repeated = pool({"A": 0.2, "C": 0.8})
        draws = [first._select_defence_idx() for _ in range(10000)]
        self.assertEqual(draws, [repeated._select_defence_idx() for _ in draws])
        self.assertNotIn(1, draws)
        self.assertAlmostEqual(draws.count(0) / len(draws), 0.2, delta=0.015)
        self.assertEqual(first.probabilities, (0.2, 0., 0.8))
        self.assertEqual(json.loads(first.mlflow_log_params()[
            "defence.family_probabilities"
        ]), {"A": 0.2, "B": 0., "C": 0.8})

    def test_invalid_distributions_fail_without_silent_normalization(self):
        for probabilities in (
            {}, {"A": 2}, {"A": 0.5}, {"A": -0.1, "B": 1.1},
            {"A": float("nan")}, {"A": float("inf")}, {"unknown": 1},
            {"A": True}, {"A": "1"}, [0.2, 0.8], "other",
        ):
            with (self.subTest(probabilities=probabilities),
                  self.assertRaises(ValueError)):
                pool(probabilities)
        self.assertEqual(family_probabilities(["only"], {"only": 1}), (1.,))

    def test_metadata_cache_and_vocabulary_follow_selected_family(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            sampler = pool({"B": 1})
            base = make_dataset(Path(tmp), aug=1, pool=sampler)
            info = InformativeDataset(
                base, include_metadata=True, allow_cached_defence_augmentation=True,
            )
            first, repeated = info[0], info[0]
            self.assertEqual(first[-1]["defence_id"], "B")
            self.assertEqual(repeated[-1], first[-1])
            self.assertEqual(sampler.member_ids, ("A", "B", "C"))
            self.assertEqual([child.calls for child in sampler.defences], [0, 1, 0])
            base.wipe_cache()

    def test_worker_streams_reseed_the_weighted_selector(self):
        runs = []
        for _ in range(2):
            loader = make_dataloader(
                PoolDrawDataset(), batch_size=16, num_workers=2,
                generator=torch.Generator().manual_seed(456),
            )
            batches = list(loader)
            draws, workers = (torch.cat([batch[i] for batch in batches])
                              for i in range(2))
            self.assertFalse(torch.any(draws == 2))
            self.assertFalse(torch.equal(draws[workers == 0], draws[workers == 1]))
            runs.append(draws)
        torch.testing.assert_close(*runs, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
