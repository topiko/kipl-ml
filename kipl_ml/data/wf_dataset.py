from __future__ import annotations

import fcntl
import os
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
from kipl_ml.trace.features import FeatureTrs

logger = get_logger(__name__)


class WFDataset(Dataset):
    def __init__(
        self,
        dataset: str,
        meta_df: pd.DataFrame,
        feature_trs: FeatureTrs,
        label: str = assets.PAGE_LABEL,
        defence: _Def | None = None,
        defence_aug: int = 0,
    ) -> None:

        logger.info("Buidling dataset...")
        logger.info(key_val_fmt("name", dataset, suffix=""))

        self.meta_df = meta_df
        self.name = dataset
        self.label = label

        self.feature_trs = feature_trs

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

        self.get_feature_shapes()

    def report(self, to_log: bool = True) -> str:
        str_ = f"Dataset {self.name} w. {self.label}s:\n"
        str_ += key_val_fmt("n_traces", len(self))
        str_ += key_val_fmt("n_classes", self.n_classes)
        str_ += key_val_fmt("defence augmentation", self.defence_aug)
        str_ += self.feature_trs.report(to_log=False)
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
    def output_sizes(self) -> dict[str, dict[str, int]]:
        return self.feature_trs.output_sizes

    def get_feature_shapes(self) -> None:
        X = self._get_trace(0)
        self.feature_trs.get_shapes(X)

    def _get_idx(self, idx: int) -> tuple[int, int]:
        if self.defence_aug == 0:
            return (idx, 0)

        return (idx // self.defence_aug, idx % self.defence_aug)

    def _get_trace(self, idx: int) -> dict[str, torch.tensor]:

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
                with open(tmp_trace_path, "rb") as f:
                    trace = torch.load(f, weights_only=True)

        return {k: v.float() for k, v in trace.items()}

    def _get_label(self, idx: int) -> torch.Tensor:
        idx = self._get_idx(idx)[0]
        return torch.tensor(self.meta_df.iloc[idx][self.label], dtype=torch.long)

    def __len__(self) -> int:
        if self.defence_aug > 0:
            return len(self.meta_df) * self.defence_aug

        return len(self.meta_df)

    def __getitem__(self, idx: int) -> tuple[dict[str, torch.Tensor], torch.Tensor]:

        trace_dict = self._get_trace(idx)

        trace_dict = self.feature_trs(trace_dict)

        label = self._get_label(idx)

        return trace_dict, label


def dict_to_device(
    X: dict[str, torch.tensor], device: torch.DeviceObjType
) -> dict[str, torch.tensor]:
    return {k: v.to(device) for k, v in X.items()}


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
    **kwargs,
) -> tuple[WFDataset, WFDataset, WFDataset]:

    meta_df = load_dataset_meta_df(dataset)

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
        dataset=f"{dataset}-train",
        label=label,
        meta_df=train_df,
        defence=defence_train,
        **kwargs,
    )
    train_ds.report()

    logger.info(
        "Setting %d fold augmentation for valid and test sets.", defence_aug_valid
    )
    kwargs["defence_aug"] = defence_aug_valid

    valid_ds = WFDataset(
        dataset=f"{dataset}-valid",
        label=label,
        meta_df=valid_df,
        defence=defence_valid,
        **kwargs,
    )
    valid_ds.report()

    test_ds = WFDataset(
        dataset=f"{dataset}-test",
        label=label,
        meta_df=test_df,
        defence=defence_test,
        **kwargs,
    )
    test_ds.report()

    return train_ds, valid_ds, test_ds
