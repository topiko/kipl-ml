from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from mbnt import load_trace_to_numpy

from kipl_ml.config import PROJECT_ROOT
from kipl_ml.data import assets
from kipl_ml.logging.logger import get_logger
from kipl_ml.trace.params import DOWNLOAD, EVENTS_MULTIPLIER, MAX_TRACE_LENGTH, UPLOAD

logger = get_logger(__name__)

METADF_FNAME = "metadf.h5"


@dataclass
class Datasets:
    BIGENOUGH: str = "bigenough"
    GONG_SURAKAV: str = "gong-surakav"


def _xv_splits_fname(dataset: str, n_splits: int, label_asset: str) -> Path:

    path = get_dataset_root(dataset).joinpath(f"xv_splits-{n_splits}-{label_asset}.csv")

    return path


def get_dataset_root(dataset: str) -> Path:

    return Path(os.path.join(PROJECT_ROOT, ".data", dataset))


def load_dataset_meta_df(dataset: str, include_xv_cols: bool = True) -> pd.DataFrame:
    """
    Load metadata for dataset
    """
    meta_path = os.path.join(get_dataset_root(dataset), "metadf.h5")
    logger.info("Loading metadata for %s from %s...", dataset, meta_path)
    try:
        meta_df = pd.read_hdf(meta_path)
    except FileNotFoundError as e:
        err = f"Could not find metadata file? Have you ran: 'python kipl_ml/data/conversion.py --dataset {dataset}'."
        logger.error(err)
        raise FileNotFoundError(err) from e

    if include_xv_cols:
        for nxv in range(12):
            for label in (assets.PAGE_LABEL, assets.SUB_PAGE_LABEL):
                try:
                    df_ = pd.read_csv(
                        _xv_splits_fname(dataset, nxv, label_asset=label),
                        index_col=False,
                    )
                    meta_df = meta_df.merge(df_, on=assets.TRACE_ID)

                except FileNotFoundError:
                    pass

    return meta_df


def get_std_trace_array(
    path: os.PathLike,
    network_delay_millis: int = 0,
    network_packets_per_second: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load a standard trace array from a file
    The standard is given by:

    return:
        - times [ns]: np.ndarray[float64]
        - dirs: np.ndarray[int8]
        - paddings: np.ndarray[bool]
    """

    return load_trace_to_numpy(
        str(path),
        network_delay_millis=network_delay_millis,
        network_packets_per_second=network_packets_per_second,
        max_trace_length=MAX_TRACE_LENGTH,
        events_multiplier=EVENTS_MULTIPLIER,
    )


def parse_trace_to_tensor_dict(
    times: np.ndarray,
    dirs: np.ndarray,
    paddings: np.ndarray,
    sizes: np.ndarray | None = None,
    time_unit: str = "s",
) -> dict[str, torch.tensor]:

    if time_unit != "s":
        logger.warning("Some feature rely on time unit being 's' beware!")

    match time_unit:
        case "s":
            times = times / 1e9
        case "ms":
            times = times / 1e6
        case "mus":
            times = times / 1e3
        case "ns":
            pass
        case _:
            raise KeyError("Invalid time unit: {time_unit}")

    sizes = sizes or np.ones_like(times)

    # We make the cast to 32bit later in dataset.
    trace_dict = {
        assets.TIMES: torch.tensor(times, dtype=torch.float64),
        assets.DIRS: torch.tensor(dirs, dtype=torch.int8),
        assets.SIZES: torch.tensor(sizes, dtype=torch.int16),
        assets.PADDING: torch.tensor(paddings, dtype=torch.bool),
    }

    return trace_dict


def get_std_trace_dict(
    path: os.PathLike,
    network_delay_millis: int = 0,
    network_packets_per_second: int = 0,
) -> dict[str, torch.tensor]:

    times, dirs, paddings = get_std_trace_array(
        path,
        network_delay_millis=network_delay_millis,
        network_packets_per_second=network_packets_per_second,
    )

    return parse_trace_to_tensor_dict(times, dirs, paddings, None)


def tensor_dict_to_str(trace_d: dict[str, torch.tensor]) -> str:

    times = (trace_d[assets.TIMES] * 1e9).detach().numpy().astype(int)
    dirs = trace_d[assets.DIRS].detach().numpy().astype(int)
    sizes = (trace_d[assets.SIZES].detach().numpy().astype(int) * 512).astype(str)

    dirs_ = np.empty_like(dirs, dtype="<U2")

    paddings = trace_d[assets.PADDING].detach().numpy().astype(bool)

    dirs_[(dirs == UPLOAD) & ~paddings] = "sn"  # send normal
    dirs_[(dirs == UPLOAD) & paddings] = "sp"  # send padding
    dirs_[(dirs == DOWNLOAD) & ~paddings] = "rn"  # receive normal
    dirs_[(dirs == DOWNLOAD) & paddings] = "rp"  # receive

    if any(dirs_ == ""):
        raise ValueError("Invalid dir detected")
    arr = np.vstack((times, dirs_, sizes)).T

    str_ = "\n".join([f"{row[0]},{row[1]},{row[2]}" for row in arr])

    return str_


def generate_xv_splits(
    dataset: str,
    n_splits: int,
    label_asset: str,
    random_state: int = 42,
    overlap_policy: str = "warn",
):

    logger.info(
        "Generating %d splits for dataset %s on label %s...",
        n_splits,
        dataset,
        label_asset,
    )
    meta_df = load_dataset_meta_df(dataset, include_xv_cols=False)
    meta_df = meta_df.sort_values(assets.TRACE_ID).reset_index(drop=True)
    n_labels = meta_df.loc[:, label_asset].value_counts()

    if n_labels.nunique() != 1:
        logger.warning("Different number of items per class --> checks omitted!")
    else:
        n_items_per_class = n_labels.unique()[0]

        if n_items_per_class % n_splits != 0:
            err = "Number of items per class is not divisible by n_splits --> expect overlapping xv splits."
            if overlap_policy == "raise":
                raise ValueError(err)
            if overlap_policy == "warn":
                logger.warning(err)
            else:
                raise ValueError(f"Invalid overlap policy: {overlap_policy}")

    if label_asset != assets.PAGE_LABEL:
        raise NotImplementedError()

    xv_col = assets.XV_SPLIT(n_splits, label_asset)

    if dataset == Datasets.BIGENOUGH:
        if n_splits == 10:
            meta_df.loc[:, xv_col] = meta_df.loc[:, assets.SAMPLE_ID] // 2
        elif n_splits == 5:
            meta_df.loc[:, xv_col] = meta_df.loc[:, assets.SAMPLE_ID] // 4
        else:
            raise NotImplementedError()
    elif dataset == Datasets.GONG_SURAKAV:
        for lbl in meta_df.loc[:, label_asset].unique():
            mask = meta_df.loc[:, label_asset] == lbl

            if mask.sum() % n_splits != 0:
                raise ValueError("Class cannot be split by nsplits")

            n_ = mask.sum() // n_splits
            split_idx = np.repeat(np.arange(n_splits), n_)
            np.random.shuffle(split_idx)

            meta_df.loc[mask, xv_col] = split_idx

        meta_df.loc[:, xv_col] = meta_df.loc[:, xv_col].astype(int)

    fname = _xv_splits_fname(dataset, n_splits, label_asset)

    if not fname.parent.exists():
        os.makedirs(fname.parent, exist_ok=False)

    if os.path.isfile(fname):
        logger.warning(
            "%s xv split file: %s exists - exiting w.o. replace", dataset, fname
        )
        return

    meta_df.loc[:, [assets.TRACE_ID, xv_col]].sort_values(assets.TRACE_ID).to_csv(
        fname, index=False
    )
    logger.info("Saved xv splits (%s) to %s", n_splits, fname)


def preserve_class_frac_sample(
    meta_df: pd.DataFrame,
    n_samples: int,
    random_state: int | None = None,
    missing_classes: str = "raise",
) -> pd.DataFrame:
    """
    Sample n_samples preserving the class fraction
    """

    frac = n_samples / len(meta_df)
    sampled_meta_df = (
        meta_df.groupby(assets.PAGE_LABEL)
        .apply(lambda x: x.sample(frac=frac, random_state=random_state))
        .reset_index(drop=True)
    )

    if set(meta_df[assets.PAGE_LABEL]) != set(sampled_meta_df[assets.PAGE_LABEL]):
        msg = "Sampled meta_df does not contain all classes."
        if missing_classes == "raise":
            raise ValueError(msg)
        elif missing_classes == "warn":
            logger.warning(msg)
        elif missing_classes == "ignore":
            pass
        else:
            raise KeyError("Invalid missing classes key: '{missing_classes}'")

    if len(sampled_meta_df) == 0:
        raise ValueError("Empty df")

    return sampled_meta_df


if __name__ == "__main__":
    _xv_splits_fname(Datasets.BIGENOUGH, 5)
