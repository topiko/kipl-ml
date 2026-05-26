import os
from contextlib import ExitStack
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory

from mbnt import compute_overheads
from torch.utils.data import DataLoader
from tqdm import tqdm

from kipl_ml.data.assets import DATASET, TRACE_F_PATH
from kipl_ml.data.utils import tensor_dict_to_str
from kipl_ml.data.wf_dataset import InformativeDataset, WFDataset
from kipl_ml.defences.nndefs import RNNDef
from kipl_ml.logging.logger import TQDM_W, get_logger
from kipl_ml.network.network import NetworkContextIntDict
from kipl_ml.trace.params import MAX_TRACE_LENGTH

logger = get_logger(__name__)


def _unwrap_dataset(dataset: WFDataset | InformativeDataset) -> WFDataset:
    if isinstance(dataset, InformativeDataset):
        return dataset.base
    return dataset


def _collate_overheads(
    batch: list[tuple[dict, object, int, NetworkContextIntDict]],
) -> list[tuple[str, int, NetworkContextIntDict]]:
    out: list[tuple[str, int, NetworkContextIntDict]] = []
    for trace_d, _, idx, _network_context in batch:
        out.append((tensor_dict_to_str(trace_d), int(idx), _network_context))
    return out


def _make_tmp(
    dataset: WFDataset | InformativeDataset,
    dir_orig: Path,
    dir_defended: Path,
    overhead_frac: float,
):

    defence = deepcopy(dataset.defence)

    if isinstance(defence, RNNDef):
        # We want to go over the full trace.
        # Set any simul restricting params to high values.
        defence._n_packets = 100_000_000
        defence._max_dur_s = 100_000

    # We create a new dataset w.o., defence.
    dataset = _unwrap_dataset(dataset).clone(
        feature_trs=None,
        dataset_key="overheads",
        defence=None,
        defence_aug=0,
        trim_raw=0,
    )

    dataset.meta_df = dataset.meta_df.sample(frac=overhead_frac, random_state=0)

    if dataset.meta_df.loc[:, DATASET].nunique() != 1:
        raise ValueError("Multiple datasets detected")

    dataset_name = dataset.meta_df.loc[:, DATASET].unique()[0]

    info_ds = InformativeDataset(dataset)
    dl_kwargs = {
        "batch_size": 128,
        "shuffle": False,
        "num_workers": min(24, os.cpu_count() or 1),
        "collate_fn": _collate_overheads,
    }
    dl = DataLoader(info_ds, **dl_kwargs)

    with tqdm(dl, desc="tmp files", ncols=TQDM_W, total=len(dl)) as pbar:
        for batch in pbar:
            for undefended_str_trace, idx, network_context in batch:
                trace_path = info_ds.get_meta(int(idx))[TRACE_F_PATH]
                sub_folder = trace_path.split(dataset_name)[-1][1:]

                orig_path = dir_orig.joinpath(sub_folder)
                def_path = dir_defended.joinpath(sub_folder)

                defended_trace = defence(
                    Path(trace_path),
                    None,
                    trim_raw=dataset.trim_raw,
                    network_context=network_context,
                )
                defended_str_trace = tensor_dict_to_str(defended_trace)
                for p_, trace_str in (
                    (orig_path, undefended_str_trace),
                    (def_path, defended_str_trace),
                ):
                    if not p_.parent.exists():
                        p_.parent.mkdir()

                    with open(p_, "w", encoding="utf-8") as f:
                        f.write(trace_str)


def get_overheads(
    dataset: WFDataset | InformativeDataset,
    overhead_frac: float,
    max_len: int = MAX_TRACE_LENGTH,
    real_world: bool = False,
    full_output: bool = False,
) -> dict[str, float]:
    dataset = _unwrap_dataset(dataset).clone(dataset_key="overheads")

    logger.info("Compute overheads for: %s", dataset.defence.name)
    with ExitStack() as stack:
        dirs = [
            stack.enter_context(TemporaryDirectory(suffix=".traces", prefix=prefix))
            for prefix in ("orig", f"defended_{dataset.defence.name}")
        ]

        _make_tmp(dataset, Path(dirs[0]), Path(dirs[1]), overhead_frac)

        overheads = compute_overheads(dirs[0], dirs[1], max_len, real_world)

    overheads_fin: dict[str, float] = {}
    overheads_fin["def.bandwidth"] = overheads["load"]
    overheads_fin["def.delay"] = overheads["delay"]
    overheads_fin["sim.missing"] = overheads["missing"]

    if full_output:
        for k, v in overheads.items():
            if k in {"delay", "missing"}:
                continue
            overheads_fin[k] = v
    return overheads_fin
