from __future__ import annotations

import os
from itertools import product
from pathlib import Path

import kipl_ml.data.assets as assets
import numpy as np
import pandas as pd
import torch
from kipl_ml.config import PROJECT_ROOT
from kipl_ml.logging.logger import get_logger
from kipl_ml.trace.params import MAX_TRACE_LENGTH
from omegaconf import DictConfig
from mbnt import load_trace_to_numpy
from sklearn.model_selection import StratifiedKFold

logger = get_logger(__name__)

METADF_FNAME = "metadf.h5"


def _xv_splits_fname(dataset: str, n_splits: int) -> Path:

    path = get_dataset_root(dataset).joinpath(f"xv_splits-{n_splits}.csv")

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
        for nxv in range(16):
            try:
                df_ = pd.read_csv(_xv_splits_fname(dataset, nxv), index_col=False)
                meta_df = meta_df.merge(df_, on=assets.TRACE_ID)

            except FileNotFoundError:
                pass

    return meta_df


def get_std_trace_array(
    path: os.PathLike, network_delay_millis: int = 0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load a standard trace array from a file
    The standard is given by:

    return:
        - times: np.ndarray[float32]
        - dirs: np.ndarray[int8]
        - paddings: np.ndarray[bool]
    """

    return load_trace_to_numpy(
        str(path),
        network_delay_millis=network_delay_millis,
        max_trace_length=MAX_TRACE_LENGTH,
    )


def parse_trace_to_tensor_dict(
    times: np.ndarray,
    dirs: np.ndarray,
    paddings: np.ndarray,
    sizes: np.ndarray | None = None,
) -> dict[str, torch.Tensor]:

    sizes = sizes or np.ones_like(times)
    np_trace = np.vstack(
        [
            times.astype(np.float32),
            dirs.astype(np.float32),
            sizes.astype(np.float32),
        ]
    ).T

    trace_tensor = torch.as_tensor(np_trace)

    trace_dict = {
        assets.TIMES: trace_tensor[:, 0],
        assets.DIRS: trace_tensor[:, 1],
        assets.SIZES: trace_tensor[:, 2],
        assets.PADDING: torch.tensor(paddings, dtype=torch.bool),
    }

    return trace_dict


def get_std_trace_dict(
    path: os.PathLike, network_delay_millis: int = 0
) -> dict[str, torch.Tensor]:

    times, dirs, paddings = get_std_trace_array(
        path, network_delay_millis=network_delay_millis
    )
    return parse_trace_to_tensor_dict(times, dirs, paddings, None)


def generate_xv_splits(
    dataset: str,
    n_splits: int,
    random_state: int = 42,
    overlap_policy: str = "warn",
):

    logger.info("Generating %d splits for dataset %s...", n_splits, dataset)
    meta_df = load_dataset_meta_df(dataset, include_xv_cols=False)
    meta_df = meta_df.sort_values(assets.TRACE_ID).reset_index(drop=True)
    n_labels = meta_df.loc[:, assets.LABEL].value_counts()

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

    xv_splits = []
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    for i, (idxs_train, idxs_test) in enumerate(
        skf.split(
            meta_df.loc[:, [assets.TRACE_ID, assets.LABEL]],
            meta_df.loc[:, assets.LABEL],
        )
    ):
        test_trace_ids = meta_df.loc[idxs_test, assets.TRACE_ID]
        xv_splits.append(test_trace_ids.values)

    # Verify the difference between all splits
    for xv1, xv2 in product(xv_splits, xv_splits):
        if xv1 is not xv2:
            if len(set(xv1).intersection(set(xv2))) != 0:
                raise ValueError("Overlapping xv splits!?")

    fname = _xv_splits_fname(dataset, n_splits)

    if not fname.parent.exists():
        os.makedirs(fname.parent, exist_ok=False)

    if os.path.isfile(fname):
        logger.warning(
            "%s xv split file: %s exists - exiting w.o. replace", dataset, fname
        )
        return

    trace_ids = np.concatenate(xv_splits)
    xv_split = np.concatenate(
        [np.ones(len(split), dtype=int) * i for i, split in enumerate(xv_splits)]
    )

    pd.DataFrame(
        data=np.vstack((trace_ids, xv_split)).T,
        columns=[assets.TRACE_ID, assets.XV_SPLIT(n_splits)],
        dtype=str,
    ).sort_values(assets.TRACE_ID).to_csv(fname, index=False)
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
        meta_df.groupby(assets.LABEL)
        .apply(lambda x: x.sample(frac=frac, random_state=random_state))
        .reset_index(drop=True)
    )

    if set(meta_df[assets.LABEL]) != set(sampled_meta_df[assets.LABEL]):
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
    _xv_splits_fname("bigenough", 5)
