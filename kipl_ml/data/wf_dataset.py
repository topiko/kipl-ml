from __future__ import annotations

import fcntl
import os
import random
import shutil
from copy import deepcopy
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
from kipl_ml.network.network import NetworkContext, NetworkContextIntDict
from kipl_ml.trace.features import Feats, FeatureTrs

logger = get_logger(__name__)

_KEEP = object()


class WFDataset(Dataset):
    def __init__(
        self,
        meta_df: pd.DataFrame,
        feature_trs: FeatureTrs | None,
        network_context: NetworkContext,
        label: str = assets.PAGE_LABEL,
        defence: _Def | None = None,
        seed: int | None = 42,
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

        if defence is None:
            defence = NoDefence(seed=seed)

        self.defence = defence
        self.network_context = network_context.with_seed(seed)
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
        str_ += self.network_context.report() + "\n"
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

    def clone(
        self,
        *,
        feature_trs: FeatureTrs | None | object = _KEEP,
        network_context: NetworkContext | object = _KEEP,
        defence: _Def | None | object = _KEEP,
        defence_aug: int | object = _KEEP,
        dataset_key: str | None = None,
        trim_raw: int | object = _KEEP,
    ) -> WFDataset:
        if feature_trs is _KEEP:
            feature_trs = deepcopy(self.feature_trs)
        if network_context is _KEEP:
            network_context = deepcopy(self.network_context)
        if defence is _KEEP:
            defence = deepcopy(self.defence)
        if defence_aug is _KEEP:
            defence_aug = self.defence_aug
        if trim_raw is _KEEP:
            trim_raw = self.trim_raw

        return type(self)(
            meta_df=self.meta_df.copy(deep=True),
            feature_trs=feature_trs,
            network_context=network_context,
            label=self.label,
            defence=defence,
            seed=self.network_context.seed,
            defence_aug=defence_aug,
            dataset_key=dataset_key,
            trim_raw=trim_raw,
        )

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

    def wipe_cache(self) -> None:
        """Wipe the cached defended traces."""
        if self.tmp_dir is not None:
            shutil.rmtree(self.tmp_dir.name)
            self.tmp_dir = None
        self._defence_aug = 0

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

    def _sample_network_context(self) -> NetworkContextIntDict:
        return self.network_context.sample_params()

    def _get_trace_and_context(
        self, idx: int
    ) -> tuple[dict[Feats, torch.Tensor], NetworkContextIntDict]:
        orig_idx, sub_idx = self._get_idx(idx)

        orig_trace_path = Path(self.meta_df.iloc[orig_idx][assets.TRACE_F_PATH])

        machine_idx = orig_idx if self.defence.FIXED_PER_TRACE else None

        if self.defence_aug == 0:
            network_context = self._sample_network_context()
            trace = self.defence(
                orig_trace_path,
                machine_idx=machine_idx,
                trim_raw=self.trim_raw,
                network_context=network_context,
            )
        else:
            if self.tmp_dir is None:
                raise ValueError("Temporary directory not initialized")

            tmp_trace_path = os.path.join(
                self.tmp_dir.name, f"{orig_trace_path.name}.{sub_idx:03d}"
            )

            def safe_load() -> tuple[dict[Feats, torch.Tensor], NetworkContextIntDict]:
                """
                When using dataloaders, several threads can call reading of the same
                trace file. This can cause some issues, here is an attempt to protect
                against simultaneous access.
                """
                with open(orig_trace_path, "rb") as f:
                    fcntl.flock(f, fcntl.LOCK_EX)  # Acquire an exclusive lock
                    try:
                        network_context = self._sample_network_context()
                        return self.defence(
                            orig_trace_path,
                            machine_idx=machine_idx,
                            trim_raw=self.trim_raw,
                            network_context=network_context,
                        ), network_context
                    finally:
                        fcntl.flock(f, fcntl.LOCK_UN)  # Release the lock

            if not os.path.exists(tmp_trace_path):
                trace, network_context = safe_load()
                with open(tmp_trace_path, "wb") as f:
                    torch.save(
                        {"trace": trace, "network_context": network_context},
                        f,
                    )
            else:
                with torch.serialization.safe_globals([Feats]):
                    with open(tmp_trace_path, "rb") as f:
                        payload = torch.load(f, weights_only=True)
                if isinstance(payload, dict) and "trace" in payload:
                    trace = payload["trace"]
                    network_context = payload["network_context"]
                else:
                    trace = payload
                    network_context = self._sample_network_context()

        converted: dict[Feats, torch.Tensor] = {}
        for key, val in trace.items():
            if key == assets.DECOY:
                converted[key] = val.to(dtype=torch.bool)
            else:
                converted[key] = val.to(dtype=torch.float32)

        return converted, network_context

    def _get_trace(self, idx: int) -> dict[Feats, torch.Tensor]:
        trace, _ = self._get_trace_and_context(idx)
        return trace

    def _get_label(self, idx: int) -> torch.Tensor:
        idx = self._get_idx(idx)[0]
        return torch.tensor(self.meta_df.iloc[idx][self.label], dtype=torch.long)

    def get_meta(self, idx: int) -> pd.Series:
        orig_idx = self._get_idx(idx)[0]
        return self.meta_df.iloc[orig_idx]

    def __len__(self) -> int:
        if self.defence_aug > 0:
            return len(self.meta_df) * self.defence_aug

        return len(self.meta_df)

    def __getitem__(self, index: int) -> tuple[dict[Feats, torch.Tensor], torch.Tensor]:
        trace_dict = self._get_trace(index)

        if self.feature_trs is not None:
            trace_dict = self.feature_trs(trace_dict)

        label = self._get_label(index)

        return trace_dict, label


class InformativeDataset(Dataset):
    def __init__(self, base: WFDataset):
        self.base = base

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        # Override bases get_trace...

        if self.defence_aug != 0:
            raise ValueError("WithIDx dataset requires def aug == 0!")

        trace_dict, network_context = self.base._get_trace_and_context(idx)

        if self.feature_trs is not None:
            trace_dict = self.feature_trs(trace_dict)

        label = self._get_label(idx)

        return trace_dict, label, idx, network_context

    def get_meta(self, idx: int) -> pd.Series:
        return self.base.get_meta(idx)

    def __getattr__(self, name: str):
        return getattr(self.base, name)


def dict_to_device(
    X: dict[Feats, torch.tensor], device: torch.DeviceObjType, **kwargs
) -> dict[Feats, torch.tensor]:
    return {k: v.to(device, **kwargs) for k, v in X.items()}


def get_train_valid_test(
    dataset: str,
    label: str,
    network_context: NetworkContext,
    n_splits: int,
    test_xv: int,
    random_state: int | None = None,
    defence_train: _Def | None = None,
    defence_valid: _Def | None = None,
    defence_test: _Def | None = None,
    seed: int | None = 42,
    defence_aug_valid: int = 1,
    n_min_packets: int | None = None,
    **kwargs,
) -> tuple[WFDataset, WFDataset, WFDataset]:
    if network_context is None:
        raise ValueError("get_train_valid_test requires explicit network_context")

    meta_df = load_dataset_meta_df(dataset)

    if (n_min_packets := n_min_packets or 0) > 0:
        meta_df = meta_df[meta_df.loc[:, "n_packets"] >= n_min_packets]
        logger.warning("Short (<%d packets) flows removed!", n_min_packets)

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

    def _make_split(name: str, df, defence, split_seed, defence_aug):
        split_kwargs = {**kwargs, "defence_aug": defence_aug}
        ds = WFDataset(
            label=label,
            meta_df=df,
            defence=defence,
            network_context=network_context.with_seed(split_seed),
            seed=split_seed,
            dataset_key=name,
            **split_kwargs,
        )
        ds.report()
        return ds

    def _seed(delta: int) -> int | None:
        return None if seed is None else seed + delta

    train_ds = _make_split(
        "train", train_df, defence_train, _seed(1), kwargs.get("defence_aug", 0)
    )

    logger.info(
        "Setting %d fold augmentation for valid and test sets.", defence_aug_valid
    )
    valid_ds = _make_split(
        "valid", valid_df, defence_valid, _seed(2), defence_aug_valid
    )
    test_ds = _make_split("test", test_df, defence_test, _seed(3), defence_aug_valid)

    return train_ds, valid_ds, test_ds
