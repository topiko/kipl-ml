import os

import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from kipl_ml.logging.logger import get_logger
from omegaconf import DictConfig

logger = get_logger(__name__)
load_dotenv()

DATA_DIR = os.getenv("WF_DATA_DIR", ".data")
CONFIG_DIR = os.getenv("CONFIG_DIR", "conf")
KIPL_SEQS_DATA_DIR = os.getenv("KIPL_SEQS_DATA_DIR")


def get_meta_df(dataset: str) -> pd.DataFrame:

    meta_path = os.path.join(KIPL_SEQS_DATA_DIR, dataset, "metadf.h5")

    if os.path.isfile(meta_path):
        logger.info(f"Loading metadata from {meta_path}...")
    else:
        raise FileNotFoundError(
            f"Metadata file not found at {meta_path} have you generated using convert?"
        )

    df = pd.read_hdf(meta_path)
    print(df.head())
    return df


def get_full_seq(path: str) -> np.ndarray:
    return np.load(path)


if __name__ == "__main__":
    meta_df = get_meta_df("ts5-monitored")

    _, (axp, ax_up, ax_dp) = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    meta_df.n_packets.hist(ax=axp)
    meta_df.n_packets_up.hist(ax=ax_up)
    meta_df.n_packets_down.hist(ax=ax_dp)

    plt.show()
