import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import torch

from kipl_ml.data import assets
from kipl_ml.data.wf_dataset import InformativeDataset, WFDataset
from kipl_ml.defences.base import _Def
from kipl_ml.defences.pool import DefencePool
from kipl_ml.network.network import NetworkContext
from kipl_ml.trace.enums import Feats


class MarkerDefence(_Def):
    def __init__(self, marker):
        super().__init__()
        self.marker = marker
        self.calls = 0

    def _simulate(self, trace_path, machine_idx=None, trim_raw=0, network_context=None):
        self.calls += 1
        marker = self.marker + (10 if trace_path.parent.name == "other" else 0)
        return {
            Feats.DIRS: torch.tensor([marker, marker], dtype=torch.float32),
            Feats.TIMES: torch.tensor([0.0, 0.1]),
            Feats.SIZES: torch.tensor([512.0, 512.0]),
            Feats.DECOY: torch.zeros(2, dtype=torch.bool),
        }

    def report(self, to_log=True):
        return "marker"

    def _mlflow_log_params(self):
        return {"defence-type": "marker"}


class CyclingPool(DefencePool):
    def __init__(self, member_ids=("A", "B")):
        super().__init__([MarkerDefence(-1), MarkerDefence(1)], member_ids=member_ids)
        self.selections = 0

    def _select_defence_idx(self):
        index = self.selections % len(self.defences)
        self.selections += 1
        return index


def make_dataset(root: Path, *, aug: int, pool=None, same_basenames=False):
    paths = []
    for directory in ("first", "other") if same_basenames else ("first",):
        path = root / directory / "trace.log"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture\n")
        paths.append(str(path))
    return WFDataset(
        meta_df=pd.DataFrame(
            {
                assets.TRACE_F_PATH: paths,
                assets.TRACE_ID: list(range(len(paths))),
                assets.DATASET: ["fixture"] * len(paths),
                assets.PAGE_LABEL: [13] * len(paths),
            }
        ),
        feature_trs=None,
        network_context=NetworkContext(
            "vpn_custom", (5, 10), (100, 100), seed=42, tor_profile=0
        ),
        defence=pool or CyclingPool(),
        defence_aug=aug,
        dataset_key="metadata-test",
    )


class TestDefenceMetadata(unittest.TestCase):
    def test_concurrent_cache_reads_share_one_simulation_and_label(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = make_dataset(Path(tmp), aug=1)
            info = InformativeDataset(
                base, include_metadata=True, allow_cached_defence_augmentation=True
            )
            with ThreadPoolExecutor(max_workers=4) as executor:
                samples = list(executor.map(lambda _: info[0], range(8)))
            self.assertEqual(base.defence.selections, 1)
            self.assertTrue(all(sample[-1]["defence_id"] == "A" for sample in samples))
            self.assertTrue(
                all(sample[0][Feats.DIRS][0].item() == -1 for sample in samples)
            )
            base.wipe_cache()

    def test_cached_access_requires_explicit_opt_in_even_with_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = make_dataset(Path(tmp), aug=1)
            for kwargs in ({}, {"include_metadata": True}):
                with (
                    self.subTest(kwargs=kwargs),
                    self.assertRaisesRegex(
                        ValueError, "allow_cached_defence_augmentation"
                    ),
                ):
                    InformativeDataset(base, **kwargs)[0]
            self.assertEqual(base.defence.selections, 0)
            allowed = InformativeDataset(base, allow_cached_defence_augmentation=True)
            self.assertEqual(len(allowed[0]), 4)  # Old return contract is unchanged.
            default = InformativeDataset(base)
            base.wipe_cache()
            self.assertEqual(len(default[0]), 4)
            base.defence_aug = 1
            with self.assertRaises(ValueError):
                default[0]  # Guard checks live augmentation, not only construction.
            base.wipe_cache()

    def test_cached_and_uncached_metadata_match_the_exact_trace(self):
        for aug in (0, 2):
            with self.subTest(aug=aug), tempfile.TemporaryDirectory() as tmp:
                base = make_dataset(Path(tmp), aug=aug)
                info = InformativeDataset(
                    base, include_metadata=True, allow_cached_defence_augmentation=True
                )
                observed = []
                for idx in (0, 1, 0, 1) if aug else (0, 0, 0, 0):
                    trace, original_y, returned_idx, _network, metadata = info[idx]
                    member = metadata["defence_id"]
                    self.assertEqual(
                        trace[Feats.DIRS][0].item(), {"A": -1, "B": 1}[member]
                    )
                    self.assertEqual(original_y.item(), 13)
                    self.assertEqual(returned_idx, idx)
                    self.assertNotIn("defence_id", trace)
                    observed.append(member)
                self.assertEqual(observed, ["A", "B", "A", "B"])
                self.assertEqual(base.defence.selections, 2 if aug else 4)
                base.wipe_cache()

    def test_same_basename_does_not_alias_cache_and_legacy_payload_has_no_guessed_id(
        self,
    ):
        with tempfile.TemporaryDirectory() as tmp:
            pool = DefencePool([MarkerDefence(1)], member_ids=["only"])
            base = make_dataset(Path(tmp), aug=1, pool=pool, same_basenames=True)
            info = InformativeDataset(
                base, include_metadata=True, allow_cached_defence_augmentation=True
            )
            self.assertEqual(info[0][0][Feats.DIRS][0].item(), 1)
            self.assertEqual(info[1][0][Feats.DIRS][0].item(), 11)
            files = list(Path(base.tmp_dir.name).glob("*.pt"))
            self.assertEqual(len(files), 2)
            for path in files:
                with torch.serialization.safe_globals([Feats]):
                    payload = torch.load(path, weights_only=True)
                payload.pop("metadata")
                payload["trace"] = {
                    Feats(key): value for key, value in payload["trace"].items()
                }
                torch.save(payload, path)
            self.assertEqual(info[0][-1], {})
            self.assertEqual(info[1][-1], {})
            base.wipe_cache()

    def test_pool_member_ids_validate_and_plain_defences_have_empty_metadata(self):
        for ids in (("same", "same"), ("one",), ("", "two")):
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                CyclingPool(member_ids=ids)
        defence = MarkerDefence(1)
        _, metadata = defence.simulate_with_metadata(Path("trace.log"))
        self.assertEqual(metadata, {})
        self.assertEqual(defence.calls, 1)


if __name__ == "__main__":
    unittest.main()
