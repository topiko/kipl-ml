import hashlib
import math
import random
from contextlib import ExitStack, contextmanager
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch
from mbnt import compute_overheads
from torch.utils.data import Dataset
from tqdm import tqdm

from kipl_ml.data.assets import DATASET, TRACE_F_PATH
from kipl_ml.data.loading import _reseed_rngs, make_dataloader
from kipl_ml.data.utils import tensor_dict_to_str
from kipl_ml.data.wf_dataset import InformativeDataset, WFDataset
from kipl_ml.defences.base import NoDefence
from kipl_ml.defences.nndefs import BrickSelectionDef, RNNDef
from kipl_ml.logging.logger import TQDM_W, get_logger

logger = get_logger(__name__)


def _unwrap_dataset(dataset: WFDataset | InformativeDataset) -> WFDataset:
    if isinstance(dataset, InformativeDataset):
        return dataset.base
    return dataset


@contextmanager
def _trace_randomness(owner, seed: int):
    """Seed private and global CPU RNGs; leave the caller's streams untouched."""
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    try:
        random.seed(seed)
        np.random.seed(seed % (2**32))
        torch.random.default_generator.manual_seed(seed)
        _reseed_rngs(owner, np.random.SeedSequence(seed), set())
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)


class _OverheadDataset(Dataset):
    """Write one paired simulation per worker task; return only its index."""

    def __init__(self, dataset, defence, dir_orig, dir_defended, dataset_name):
        self.base = InformativeDataset(dataset)
        self.defence = defence
        self.dir_orig = dir_orig
        self.dir_defended = dir_defended
        self.dataset_name = dataset_name
        self.seed = torch.initial_seed()

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        trace_path = str(self.base.get_meta(index)[TRACE_F_PATH])
        relative = trace_path.split(self.dataset_name)[-1][1:]
        # Stable across process counts, scheduling, and source-directory moves.
        digest = hashlib.sha256(f"{self.seed}:{relative}".encode()).digest()
        seed = int.from_bytes(digest[:8], "big")
        with _trace_randomness(self, seed):
            original, _, _, network_context = self.base[index]
            defended = self.defence(
                Path(trace_path), None, trim_raw=self.base.trim_raw,
                network_context=network_context,
            )
        for directory, trace in ((self.dir_orig, original),
                                 (self.dir_defended, defended)):
            path = directory / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(tensor_dict_to_str(trace), encoding="utf-8")
        return index


def _make_tmp(
    dataset: WFDataset | InformativeDataset,
    dir_orig: Path,
    dir_defended: Path,
    overhead_frac: float,
    preserve_defence_limits: bool = False,
    num_workers: int | None = None,
):

    dataset = _unwrap_dataset(dataset)
    defence = deepcopy(dataset.defence)

    if isinstance(defence, (RNNDef, BrickSelectionDef)) and not preserve_defence_limits:
        # Measure complete normal traffic, independently of attack/training crops.
        # Native iteration budgets remain enabled and exhaustion is an error.
        defence._n_packets = None
        defence._max_dur_s = math.inf
        if isinstance(defence, RNNDef):
            defence.stop_after_all_normal_packets = True

    # We create a new dataset w.o., defence.
    dataset = dataset.clone(
        feature_trs=None,
        dataset_key="overheads",
        defence=NoDefence(simul_kwargs={"max_trace_length": 0}),
        defence_aug=0,
        trim_raw=dataset.trim_raw if preserve_defence_limits else 0,
    )

    dataset.meta_df = dataset.meta_df.sample(frac=overhead_frac, random_state=0)

    if dataset.meta_df.loc[:, DATASET].nunique() != 1:
        raise ValueError("Multiple datasets detected")

    dataset_name = dataset.meta_df.loc[:, DATASET].unique()[0]

    pairs = _OverheadDataset(dataset, defence, dir_orig, dir_defended, dataset_name)
    dl = make_dataloader(
        pairs,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        generator=torch.Generator().manual_seed(pairs.seed),
    )

    with tqdm(dl, desc="tmp files", ncols=TQDM_W, total=len(dl)) as pbar:
        for _ in pbar:
            pass  # Workers finish both files before returning a completion.


def get_overheads(
    dataset: WFDataset | InformativeDataset,
    overhead_frac: float,
    max_len: int = 0,
    real_world: bool = False,
    full_output: bool = False,
    preserve_defence_limits: bool = False,
    num_workers: int | None = None,
) -> dict[str, float]:
    """Measure overheads with paired simulations in DataLoader worker processes.

    By default neural policies run to normal-traffic completion, without packet
    or time cutoffs. Simulator/rollout safety budgets remain enabled. max_len=0
    aggregates entire paired traces, including traces with >60,000 events.

    Zero workers runs in-process. Trace-local seeds derive from the caller's
    torch initial seed and dataset-relative path, independently of worker count.
    This replaces the historical sequential RNG streams, so new realizations
    need not equal measurements made before trace-local seeding was introduced.
    """
    dataset = _unwrap_dataset(dataset).clone(dataset_key="overheads")

    logger.info("Compute overheads for: %s", dataset.defence.name)
    with ExitStack() as stack:
        dirs = [
            stack.enter_context(TemporaryDirectory(suffix=".traces", prefix=prefix))
            for prefix in ("orig", f"defended_{dataset.defence.name}")
        ]

        _make_tmp(
            dataset,
            Path(dirs[0]),
            Path(dirs[1]),
            overhead_frac,
            preserve_defence_limits=preserve_defence_limits,
            num_workers=num_workers,
        )

        overheads = compute_overheads(dirs[0], dirs[1], max_len, real_world)

    overheads_fin = _format_overheads(overheads)

    if full_output:
        for k, v in overheads.items():
            if k == "missing":
                continue
            overheads_fin[k] = v
    return overheads_fin


def _format_overheads(overheads: dict[str, float]) -> dict[str, float]:
    """Expose native summaries, tolerating older bindings without tail statistics."""
    overheads_fin: dict[str, float] = {}
    overheads_fin["def.bandwidth_median"] = overheads["overhead_data_median_multiple"]
    overheads_fin["def.delay_median"] = overheads["overhead_duration_median_multiple"]
    overheads_fin["def.bandwidth_mean"] = overheads["overhead_data_mean_multiple"]
    overheads_fin["def.bandwidth_std"] = overheads["overhead_data_std_dev_multiple"]
    overheads_fin["def.delay_mean"] = overheads["overhead_duration_mean_multiple"]
    overheads_fin["def.delay_std"] = overheads["overhead_duration_std_dev_multiple"]
    overheads_fin["def.bandwidth_full_mean"] = overheads[
        "overhead_data_full_mean_multiple"
    ]
    overheads_fin["def.delay_full_mean"] = overheads[
        "overhead_duration_full_mean_multiple"
    ]
    overheads_fin["sim.missing"] = (
        overheads["missing_mean_packets"] / overheads["base_mean_packets"]
        if overheads["base_mean_packets"]
        else 0.0
    )

    for name, native in (
        ("bandwidth", "overhead_data"),
        ("delay", "overhead_duration"),
        ("bandwidth_full", "overhead_data_full"),
        ("delay_full", "overhead_duration_full"),
        ("cumdelay", "overhead_cumdelay"),
    ):
        count = overheads.get(f"{native}_count")
        if count is not None:
            overheads_fin[f"def.{name}_count"] = count
        for stat, native_stat in (
            ("min", "min"),
            ("median", "median"),
            ("mean", "mean"),
            ("std", "std_dev"),
            ("p90", "p90"),
            ("p95", "p95"),
            ("p99", "p99"),
            ("max", "max"),
        ):
            key = f"{native}_{native_stat}_multiple"
            if key in overheads:
                overheads_fin[f"def.{name}_{stat}"] = (
                    float("nan") if count == 0 else overheads[key]
                )
    if overheads.get("base_count") == 0:
        overheads_fin["sim.missing"] = float("nan")
    overheads_fin.update(_coverage_metrics(overheads))
    return overheads_fin


def _coverage_metrics(overheads: dict[str, float]) -> dict[str, float]:
    result = {}
    for name, key in (
        ("overhead_pairs", "trace_pairs_count"),
        ("overhead_no_normal", "trace_pairs_no_normal_count"),
        ("incomplete_traces", "trace_pairs_incomplete_count"),
    ):
        if key in overheads:
            result[f"sim.{name}"] = overheads[key]
    pairs = overheads.get("trace_pairs_count")
    if pairs is None:
        return result
    for name, native in (("delay", "duration"), ("cumdelay", "cumdelay")):
        key = f"overhead_{native}_count"
        if key in overheads:
            result[f"sim.{name}_coverage"] = (
                overheads[key] / pairs
                if pairs else float("nan")
            )
    if "trace_pairs_incomplete_count" in overheads:
        incomplete = overheads["trace_pairs_incomplete_count"]
        fraction = incomplete / pairs if pairs else float("nan")
        result["sim.incomplete_fraction"] = fraction
        if fraction > 0.05:
            logger.warning(
                "%d/%d overhead trace pairs (%.1f%%) lack complete normal "
                "traffic; full-trace delay/cumdelay summaries exclude these pairs",
                incomplete, pairs, 100 * fraction,
            )
    return result
