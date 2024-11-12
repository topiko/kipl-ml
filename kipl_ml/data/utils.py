import os

import numpy as np
import pandas as pd
from dotenv import load_dotenv
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


def get_std_flow_array(path: os.PathLike) -> np.ndarray:
    """
    Load a standard flow array from a file
    The standard is given by:
        - shape: (n_timesteps, n_features), where n_features = 3
        - (n_timesteps, 0) = timestamp in [ns]
        - (n_timesteps, 1) = direction in {-1, 1}
        - (n_timesteps, 2) = packet size in [bytes]

    """
    return np.load(path)
