import os

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from kipl_ml.data.assets import assets
from kipl_ml.logging.logger import get_logger
from omegaconf import DictConfig

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


def get_std_trace_array(path: os.PathLike) -> np.ndarray:
    """
    Load a standard trace array from a file
    The standard is given by:
        - shape: (n_timesteps, n_features), where n_features = 3
        - (n_timesteps, 0) = timestamp in [ns]
        - (n_timesteps, 1) = direction in {-1, 1}
        - (n_timesteps, 2) = packet size in [bytes]

    """
    return np.load(path)


def preserve_class_frac_sample(
    meta_df: pd.DataFrame, n_samples: int, missing_classes: str = "raise"
) -> pd.DataFrame:
    """
    Sample n_samples preserving the class fraction
    """

    frac = n_samples / len(meta_df)
    sampled_meta_df = (
        meta_df.groupby(assets.LABEL)
        .apply(lambda x: x.sample(frac=frac))
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
