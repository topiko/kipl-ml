import argparse
import os
import pickle as pkl

import kipl_ml.data.assets as assets
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from kipl_ml.data.utils import METADF_FNAME, get_dataset_root
from kipl_ml.logging.logger import get_logger

logger = get_logger(__name__)
load_dotenv()

DATA_DIR = os.getenv("WF_DATA_DIR")
STD_FLOWS_DATA_DIR = os.getenv("STD_FLOWS_DATA_DIR")

assert DATA_DIR is not None
assert STD_FLOWS_DATA_DIR is not None


def _load_pickle_data(data_path: str) -> dict[int, list[list[np.ndarray | list]]]:
    """
    Load samples from pickle file
    """
    logger.info(f"Loading data from {data_path}...")
    with open(data_path, "rb") as fi:
        raw_data = pkl.load(fi)

    return raw_data


def _data_to_meta_row(
    data: np.ndarray, label: int, path: str, dataset: str, trace_id: str
) -> pd.Series:
    """
    Convert data to metadata series
    """

    def _n_packets(ud: str) -> int:
        if len(data) == 0:
            return 0
        if ud == "up":
            return (data[:, 1] > 0).sum()
        if ud == "down":
            return (data[:, 1] < 0).sum()

    def _time() -> int:
        if len(data) == 0:
            return 0
        return data[-1, 0] - data[0, 0]

    def _path() -> str:
        if len(data) == 0:
            return ""
        return path

    row_df = (
        pd.Series(
            {
                assets.LABEL: label,
                "dataset": dataset,
                "n_packets": _n_packets("up") + _n_packets("down"),
                "time [ns]": _time(),
                "trace_id": trace_id,
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
                "trace_id": str,
                "path": str,
                "dataset": str,
                "n_packets_up": int,
                "n_packets_down": int,
            }
        )
    )

    return row_df


def _save_meta_df(seq_rows: list[pd.DataFrame], dataset: str):
    path = os.path.join(get_dataset_root(dataset), METADF_FNAME)
    pd.concat(seq_rows, axis=0).reset_index(drop=True).to_hdf(path, key="metadf")


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

                    # Check kipl_ml.data.assets for the indices!
                    dat = np.vstack([times, dirs, sizes]).T

                    key = f"ms={i:04d}|seq={j:04d}"
                    fname = f"{key}.npy"
                    path_ = os.path.join(
                        get_dataset_root(dataset), f"{label:03d}", fname
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

        _save_meta_df(seq_rows, dataset)

    raw_monit = _load_pickle_data(os.path.join(DATA_DIR, "ts5", "ts5-mon.pkl"))
    save(raw_monit, dataset="ts5-monitored")

    # raw_unmonit = _load_pickle_data(os.path.join(DATA_DIR, "ts5", "ts5-unm.pkl"))
    # raw_unmonit = {-1: raw_unmonit}
    # save(raw_unmonit, dataset="ts5-unmonitored")


def _save_big_enough_to_standard():
    """
    Convert data to standard format [seq_len, n_features], where features:
        - 0: t = time
        - 1: x = direction
        - 2: s = size
    """

    root = os.path.join(DATA_DIR, "bigenough-95x10x20-standard-rngsubpages")

    def parse_row(row: str, idx: int) -> str:
        return row.split(",")[idx]

    trace_dfs = []
    for dir_ in os.listdir(root):
        if not os.path.isdir(os.path.join(root, dir_)):
            continue

        label = int(dir_)
        trace_dir = os.path.join(root, dir_)
        for log_f in os.listdir(trace_dir):
            if not log_f.endswith(".log"):
                logger.warning(f"Skipping {log_f}")
            with open(os.path.join(trace_dir, log_f), "r") as fi:
                seq = fi.readlines()

            times = np.array([parse_row(p, 0) for p in seq], dtype=float)
            dirs = np.array(
                [{"s": -1, "r": 1}[parse_row(p, 1)] for p in seq], dtype=float
            )
            sizes = np.ones_like(times)

            # Check kipl_ml.data.assets for the indices!
            trace = np.vstack([times, dirs, sizes]).T
            log_f = log_f.replace(".log", "")
            path_ = os.path.join(
                STD_FLOWS_DATA_DIR, "bigenough", f"{label:04d}", f"{log_f}.npy"
            )
            trace_df = _data_to_meta_row(
                data=trace, label=label, path=path_, dataset="bigenough", trace_id=log_f
            )

            if not os.path.exists(os.path.dirname(path_)):
                os.makedirs(os.path.dirname(path_))
            np.save(path_, trace)

            trace_dfs.append(trace_df)

    _save_meta_df(trace_dfs, "bigenough")


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--dataset", type=str, required=True, choices=["ts5", "bigenough"]
    )
    args = argparser.parse_args()

    if args.dataset == "ts5":
        _save_ts5_to_standard()
    elif args.dataset == "bigenough":
        _save_big_enough_to_standard()
