from __future__ import annotations

import fcntl
import os
import random
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
import torch
from torch.utils.data import Dataset

from kipl_ml.data import assets
from kipl_ml.data.utils import load_dataset_meta_df
from kipl_ml.defences.base import NoDefence, _Def
from kipl_ml.logging.logger import get_logger
from kipl_ml.logging.utils import key_val_fmt
from kipl_ml.trace.features import Feats, FeatureTrs

logger = get_logger(__name__)


class WFDataset(Dataset):
    def __init__(
        self,
        meta_df: pd.DataFrame,
        feature_trs: FeatureTrs | None,
        label: str = assets.PAGE_LABEL,
        defence: _Def | None = None,
        defence_aug: int = 0,
        dataset_key: str | None = None,
        trim_raw: int = 0,
    ) -> None:
        logger.info("Buidling dataset...")

        self.trim_raw = trim_raw
        dataset_key = dataset_key or "".join(
            [random.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(10)]
        )
        dataset = (
            "|".join(meta_df.loc[:, "dataset"].unique().tolist()) + f"-{dataset_key}"
        )

        logger.info(key_val_fmt("name", dataset, suffix=""))

        self.meta_df = meta_df
        self.name = dataset
        self.label = label

        self.defence = defence or NoDefence(
            network_delay_millis=(0, 0), network_pps=(0, 0)
        )
        self.tmp_dir = None
        self.defence_aug = defence_aug

        if self.defence.FIXED_PER_TRACE and self.defence_aug == 0:
            logger.warning(
                "Def. augmentation is 0, i.e., infinite, however, you fix each defence to a trace."
            )
            logger.warning(
                "Infinite augmentation does not really make sense when you used fixed machines per trace."
            )

        self.feature_trs = feature_trs

    def report(self, to_log: bool = True) -> str:
        str_ = f"Dataset {self.name} w. {self.label}s:\n"
        str_ += key_val_fmt("n_traces", self.n_orig_traces)
        str_ += key_val_fmt("defence augmentation", self.defence_aug)
        str_ += key_val_fmt("n_traces (aug)", len(self))
        str_ += key_val_fmt("n_classes", self.n_classes)
        if self.trim_raw > 0:
            str_ += key_val_fmt("trimming from start", self.trim_raw)
        if self.feature_trs is not None:
            str_ += self.feature_trs.report(to_log=False)
        else:
            str_ += "No features -->"
        str_ += self.defence.report(to_log=False)

        if to_log:
            for i, line in enumerate(str_.split("\n")):
                tab = "\t"
                if i == 0:
                    tab = ""

                logger.info(tab + line)

        return str_

    @property
    def n_orig_traces(self) -> int:
        return len(self.meta_df)

    @property
    def defence_aug(self) -> int:
        return self._defence_aug

    @defence_aug.setter
    def defence_aug(self, aug_factor: int) -> None:
        if not isinstance(aug_factor, int):
            raise TypeError("Defence augmentation must be an integer")
        if aug_factor < 0:
            raise ValueError("Defence augmentation must be non-negative")
        if aug_factor > 0:
            # With statement is unnecessary here as the tmp_dir will share
            # its lifecykle w. the parent class and the TempDir class
            # hadles the deletion, when garbage collected...?
            self.tmp_dir = TemporaryDirectory(suffix=".traces", prefix=self.name)

        self._defence_aug = aug_factor

    @property
    def n_classes(self) -> int:
        return self.meta_df[self.label].nunique()

    @property
    def feature_trs(self) -> FeatureTrs | None:
        return self._feature_trs

    @feature_trs.setter
    def feature_trs(self, feature_trs: FeatureTrs | None) -> None:
        if not isinstance(feature_trs, FeatureTrs | None):
            raise TypeError("feature_trs must be an instance of FeatureTrs or None")
        self._feature_trs = feature_trs

        if feature_trs is not None:
            self.get_feature_shapes()

    @property
    def output_sizes(self) -> dict[str, dict[str, int]]:
        return self.feature_trs.output_sizes

    def get_feature_shapes(self) -> None:
        X = self._get_trace(0)
        self.feature_trs.get_shapes(X)

    def _get_idx(self, idx: int) -> tuple[int, int]:
        if self.defence_aug == 0:
            return (idx, 0)

        return (idx // self.defence_aug, idx % self.defence_aug)

    def _get_trace(self, idx: int) -> dict[Feats, torch.Tensor]:
        orig_idx, sub_idx = self._get_idx(idx)

        orig_trace_path = Path(self.meta_df.iloc[orig_idx][assets.TRACE_F_PATH])

        machine_idx = orig_idx if self.defence.FIXED_PER_TRACE else None

        if self.defence_aug == 0:
            trace = self.defence(orig_trace_path, machine_idx=machine_idx)
        else:
            if self.tmp_dir is None:
                raise ValueError("Temporary directory not initialized")

            tmp_trace_path = os.path.join(
                self.tmp_dir.name, f"{orig_trace_path.name}.{sub_idx:03d}"
            )

            def safe_load() -> dict[str, torch.tensor]:
                """
                When using dataloaders, several threads can call reading of the same
                trace file. This can cause some issues, here is an attempt to protect
                against simultaneous access.
                """
                with open(orig_trace_path, "rb") as f:
                    fcntl.flock(f, fcntl.LOCK_EX)  # Acquire an exclusive lock
                    try:
                        return self.defence(orig_trace_path, machine_idx=machine_idx)
                    finally:
                        fcntl.flock(f, fcntl.LOCK_UN)  # Release the lock

            if not os.path.exists(tmp_trace_path):
                trace = safe_load()
                with open(tmp_trace_path, "wb") as f:
                    torch.save(trace, f)
            else:
                with torch.serialization.safe_globals([Feats]):
                    with open(tmp_trace_path, "rb") as f:
                        trace = torch.load(f, weights_only=True)

        converted: dict[Feats, torch.Tensor] = {}
        for key, val in trace.items():
            if key == assets.PADDING:
                converted[key] = val.to(dtype=torch.bool)
            else:
                converted[key] = val.to(dtype=torch.float32)

        return converted

    def _get_label(self, idx: int) -> torch.Tensor:
        idx = self._get_idx(idx)[0]
        return torch.tensor(self.meta_df.iloc[idx][self.label], dtype=torch.long)

    def __len__(self) -> int:
        if self.defence_aug > 0:
            return len(self.meta_df) * self.defence_aug

        return len(self.meta_df)

    def __getitem__(self, idx: int) -> tuple[dict[Feats, torch.Tensor], torch.Tensor]:
        trace_dict = self._get_trace(idx)

        if self.trim_raw:
            trace_dict = {k: v[self.trim_raw :] for k, v in trace_dict.items()}
            # Set time to start from 0.
            if trace_dict[Feats.TIMES].numel():
                t0 = float(trace_dict[Feats.TIMES][0].item())
                # Avoid in-place ops: cached traces may contain inference tensors.
                trace_dict[Feats.TIMES] = trace_dict[Feats.TIMES] - t0

        if self.feature_trs is not None:
            trace_dict = self.feature_trs(trace_dict)

        label = self._get_label(idx)

        return trace_dict, label


def dict_to_device(
    X: dict[Feats, torch.tensor], device: torch.DeviceObjType, **kwargs
) -> dict[Feats, torch.tensor]:
    return {k: v.to(device, **kwargs) for k, v in X.items()}


def _exclude_long_traces(
    meta_df: pd.DataFrame,
    *,
    dataset: str,
    n_ref: int,
    t_max: float,
) -> pd.DataFrame:
    if (col := f"time_to_{n_ref}_packets") not in meta_df.columns:
        raise KeyError(
            f"Column '{col}' not found in meta_df. "
            + "Re-run conversion, e.g. 'python kipl_ml/data/conversion.py --dataset "
            + f"{dataset}'."
        )

    t_col = pd.Series(
        pd.to_numeric(meta_df.loc[:, col], errors="raise"),
        index=meta_df.index,
        dtype=float,
    )

    n_all = int(len(meta_df))
    m_ref = meta_df.loc[:, "n_packets"] >= n_ref
    n_ref_total = int(m_ref.sum())

    m_excl = t_col > t_max
    n_excl = (m_excl & m_ref).sum()

    logger.warning(
        "Filtering long traces by %s > %.3fs: excluded=%d/%d (%.2f%% all), "
        + "excluded among n_packets>=%d: %d/%d (%.2f%%)",
        col,
        t_max,
        n_excl,
        n_all,
        (100.0 * n_excl / max(1, n_all)),
        n_ref,
        n_excl,
        n_ref_total,
        (100.0 * n_excl / max(1, n_ref_total)),
    )

    return meta_df.loc[~m_excl]


def get_train_valid_test(
    dataset: str,
    label: str,
    n_splits: int,
    test_xv: int,
    random_state: int | None = None,
    defence_train: _Def | None = None,
    defence_valid: _Def | None = None,
    defence_test: _Def | None = None,
    defence_aug_valid: int = 1,
    n_min_packets: int | None = None,
    exclude_time_to_packets_n: int | None = None,
    exclude_time_to_packets_s: float | None = None,
    **kwargs,
) -> tuple[WFDataset, WFDataset, WFDataset]:
    meta_df = load_dataset_meta_df(dataset)

    if (n_min_packets := n_min_packets or 0) > 0:
        meta_df = meta_df[meta_df.loc[:, "n_packets"] >= n_min_packets]
        logger.warning("Short (<%d packets) flows removed!", n_min_packets)

    if exclude_time_to_packets_s is not None:
        meta_df = _exclude_long_traces(
            meta_df,
            dataset=dataset,
            n_ref=exclude_time_to_packets_n or 0,
            t_max=float(exclude_time_to_packets_s),
        )

    if len(meta_df) == 0:
        raise ValueError("No traces left after metadata filtering")

    if (col := assets.XV_SPLIT(n_splits, label)) not in meta_df.columns:
        raise KeyError(
            f"Column '{col}' not found in meta_df, \
            you can generate xv splits with kipl_ml.data.utils.generate_xv_splits"
        )

    valid_xv = test_xv - 1 if test_xv > 0 else n_splits - 1
    train_df = meta_df[~meta_df[col].isin((valid_xv, test_xv))].sample(
        frac=1, random_state=random_state
    )
    valid_df = meta_df[meta_df[col] == valid_xv]
    test_df = meta_df[meta_df[col] == test_xv]

    train_ds = WFDataset(
        label=label,
        meta_df=train_df,
        defence=defence_train,
        dataset_key="train",
        **kwargs,
    )
    train_ds.report()

    logger.info(
        "Setting %d fold augmentation for valid and test sets.", defence_aug_valid
    )
    kwargs["defence_aug"] = defence_aug_valid

    valid_ds = WFDataset(
        label=label,
        meta_df=valid_df,
        defence=defence_valid,
        dataset_key="valid",
        **kwargs,
    )
    valid_ds.report()

    test_ds = WFDataset(
        label=label,
        meta_df=test_df,
        defence=defence_test,
        dataset_key="test",
        **kwargs,
    )
    test_ds.report()

    return train_ds, valid_ds, test_ds
