import os
import pickle as pkl

import hydra
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from kipl_ml.logging.logger import get_logger
from omegaconf import DictConfig

logger = get_logger(__name__)
load_dotenv()

DATA_DIR = os.getenv("WF_DATA_DIR")
KIPL_SEQS_DATA_DIR = os.getenv("KIPL_SEQS_DATA_DIR")
CONFIG_DIR = os.getenv("CONFIG_DIR")

assert DATA_DIR is not None
assert KIPL_SEQS_DATA_DIR is not None
assert CONFIG_DIR is not None


def _load_pickle_data(data_path: str) -> dict[int, list[list[np.ndarray | list]]]:
    """
    Load samples from pickle file
    """
    logger.info(f"Loading data from {data_path}...")
    with open(data_path, "rb") as fi:
        raw_data = pkl.load(fi)

    return raw_data


def _data_to_meta_row(
    dat: np.ndarray, label: int, path: str, dataset: str, flow_id: str
) -> pd.Series:
    """
    Convert data to metadata series
    """

    def _n_packets(ud: str) -> int:
        if len(dat) == 0:
            return 0
        if ud == "up":
            return (dat[:, 1] > 0).sum()
        if ud == "down":
            return (dat[:, 1] < 0).sum()

    def _time() -> int:
        if len(dat) == 0:
            return 0
        return dat[-1, 0] - dat[0, 0]

    def _path() -> str:
        if len(dat) == 0:
            return ""
        return path

    row_df = (
        pd.Series(
            {
                "label": label,
                "dataset": dataset,
                "n_packets": _n_packets("up") + _n_packets("down"),
                "time [ns]": _time(),
                "flow_id": flow_id,
                "n_packets_up": _n_packets("up"),
                "n_packets_down": _n_packets("down"),
                "path": path,
            }
        )
        .to_frame()
        .T.astype(
            {
                "label": int,
                "n_packets": int,
                "time [ns]": float,
                "flow_id": str,
                "path": str,
                "dataset": str,
                "n_packets_up": int,
                "n_packets_down": int,
            }
        )
    )

    return row_df


def _save_ts5_to_standard():
    """
    Convert data to standard format [seq_len, n_features], where features:
        - 0: t = time
        - 1: x = direction
        - 2: s = size
    """

    def save(raw_data: dict[int, list[list[np.ndarray | list]]], dataset: str):

        seq_rows = []
        L = 0
        for label, multisample in raw_data.items():
            for i, sample in enumerate(multisample):
                for j, seq in enumerate(sample):
                    if isinstance(seq, list):
                        seq = np.array(seq)

                    logger.info(f"Processing label {label}...")
                    times = np.abs(seq)
                    dirs = np.sign(seq)
                    sizes = np.ones_like(seq)
                    dat = np.vstack([times, dirs, sizes]).T

                    key = f"ms={i:04d}|seq={j:04d}"
                    fname = f"{key}.npy"
                    path_ = os.path.join(
                        KIPL_SEQS_DATA_DIR, dataset, f"{label:03d}", fname
                    )
                    dir_ = os.path.dirname(path_)
                    if not os.path.exists(dir_):
                        os.makedirs(dir_)

                    row = _data_to_meta_row(dat, label, path_, dataset, key)
                    seq_rows.append(row)

                    if len(dat) == 0:
                        logger.warning(f"Empty sequence: {path_}")
                        continue

                    np.save(path_, dat)
                    L += 1

        pd.concat(seq_rows, axis=0).reset_index().to_hdf(
            os.path.join(KIPL_SEQS_DATA_DIR, dataset, "metadf.h5"), key="metadf"
        )

    raw_monit = _load_pickle_data(os.path.join(DATA_DIR, "ts5-mon.pkl"))
    save(raw_monit, dataset="ts5-monitored")

    # raw_unmonit = _load_pickle_data(os.path.join(DATA_DIR, "ts5-unm.pkl"))
    # raw_unmonit = {-1: raw_unmonit}
    # save(raw_unmonit, dataset="ts5-unmonitored")


if __name__ == "__main__":
    _save_ts5_to_standard()
