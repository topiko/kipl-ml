import os

import kipl_ml.data.assets as assets
import numpy as np
import pandas as pd
import torch
from dotenv import load_dotenv
from kipl_ml.logging.logger import get_logger
from kipl_ml.trace.params import MAX_TRACE_LENGTH
from omegaconf import DictConfig
from rustbindings import load_trace_to_numpy

logger = get_logger(__name__)
load_dotenv()

STD_FLOWS_DATA_DIR = os.getenv("STD_FLOWS_DATA_DIR")
METADF_FNAME = "metadf.h5"


def get_dataset_root(dataset: str) -> str:
    return os.path.join(STD_FLOWS_DATA_DIR, dataset)


def load_dataset_meta_df(dataset: str) -> pd.DataFrame:
    """
    Load metadata for dataset
    """
    meta_path = os.path.join(get_dataset_root(dataset), "metadf.h5")
    logger.info(f"Loading metadata for {dataset} from {meta_path}...")
    return pd.read_hdf(meta_path)


def get_std_trace_array(
    path: os.PathLike, network_delay_millis: int = 0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load a standard trace array from a file
    The standard is given by:
        - shape: (n_timesteps, n_features), where n_features = 3
        - (n_timesteps, 0) = timestamp in [mus]  float32
        - (n_timesteps, 1) = direction in {-1, 1} int8
        - (n_timesteps, 2) = padding {True, False}] bool

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

    trace_tensor = torch.Tensor(np_trace)

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
