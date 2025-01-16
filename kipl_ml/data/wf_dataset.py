import atexit
import os
import shutil
from pathlib import Path

import kipl_ml.data.assets as assets
import pandas as pd
import torch
from kipl_ml.data.utils import load_dataset_meta_df, preserve_class_frac_sample
from kipl_ml.defences.base import NoDefence, _Def
from kipl_ml.logging.logger import get_logger
from kipl_ml.logging.utils import key_val_fmt
from kipl_ml.trace.features import FeatureTrs
from kipl_ml.trace.transforms import _TR
from torch.utils.data import Dataset

logger = get_logger(__name__)
TMP_TRACES = Path(".traces/")


class WFDataset(Dataset):
    def __init__(
        self,
        dataset: str,
        meta_df: pd.DataFrame,
        feature_trs: FeatureTrs,
        defence: _Def | None = None,
        defence_aug: int = 0,
    ) -> None:

        logger.info("Buidling dataset...")
        logger.info(key_val_fmt("name", dataset))

        self.meta_df = meta_df
        self.name = dataset

        self.feature_trs = feature_trs

        self.defence = defence or NoDefence(network_delay_millis=0)
        self.defence_aug = defence_aug
        self.tmp_dir = Path(os.path.join(TMP_TRACES, Path(dataset)))

        if defence_aug > 0:
            os.makedirs(self.tmp_dir, exist_ok=False)

        self.get_feature_shapes()
        self.report()

    def report(self, to_log: bool = True) -> str:
        str_ = f"Dataset: {self.name}...\n"
        str_ += key_val_fmt("n_traces", len(self)) + "\n"
        str_ += key_val_fmt("n_classes", self.n_classes) + "\n"
        str_ += key_val_fmt("defence augmentation", self.defence_aug) + "\n"
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
    def n_classes(self) -> int:
        return self.meta_df[assets.LABEL].nunique()

    @property
    def output_sizes(self) -> dict[str, int]:
        return self.feature_trs.output_sizes

    def get_feature_shapes(self) -> None:
        X = self._get_trace(0)
        self.feature_trs.get_shapes(X)

    def _get_trace(self, idx: int) -> dict[str, torch.Tensor]:

        orig_idx = idx
        if self.defence_aug > 0:
            orig_idx = idx // self.defence_aug

        orig_trace_path = Path(self.meta_df.iloc[orig_idx]["orig_path"])

        if self.defence_aug == 0:
            return self.defence(orig_trace_path)

        sub_idx = orig_idx % self.defence_aug
        tmp_trace_path = os.path.join(
            self.tmp_dir, f"{orig_trace_path.name}.{sub_idx:03d}"
        )

        if not os.path.exists(tmp_trace_path):
            trace = self.defence(orig_trace_path)
            with open(tmp_trace_path, "wb") as f:
                torch.save(trace, f)
        else:
            with open(tmp_trace_path, "rb") as f:
                trace = torch.load(f, weights_only=True)

        return trace

    def _get_label(self, idx: int) -> torch.Tensor:
        return torch.tensor(self.meta_df.iloc[idx][assets.LABEL], dtype=torch.long)

    def __len__(self) -> int:
        return len(self.meta_df)

    def __getitem__(self, idx: int) -> tuple[dict[str, torch.Tensor], torch.Tensor]:

        trace_dict = self._get_trace(idx)

        trace_dict = self.feature_trs(trace_dict)

        label = self._get_label(idx)

        return trace_dict, label


@atexit.register
def clean_tmp():
    logger.info("Cleaning tmp traces...")

    try:
        shutil.rmtree(TMP_TRACES)  # , ignore_errors=True)
    except FileNotFoundError:
        logger.error(f"{TMP_TRACES}/ not found")


def get_train_valid_test(
    dataset: str,
    n_samples: int | tuple[int, int, int],
    random_state: int | None = None,
    defence_train: _Def | None = None,
    defence_valid_test: _Def | None = None,
    **kwargs,
) -> tuple[WFDataset, WFDataset, WFDataset]:

    if isinstance(n_samples, int):
        n_samples = (n_samples,) * 3

    meta_df = load_dataset_meta_df(dataset)

    train_df = preserve_class_frac_sample(
        meta_df, n_samples[0], random_state=random_state
    )

    train_ds = WFDataset(
        dataset=f"{dataset}-train", meta_df=train_df, defence=defence_train, **kwargs
    )

    valid_mask = ~meta_df.loc[:, assets.TRACE_ID].isin(train_df.loc[:, assets.TRACE_ID])
    meta_df = meta_df[valid_mask]

    valid_df = preserve_class_frac_sample(
        meta_df,
        n_samples=n_samples[1],
        random_state=random_state,
        missing_classes="warn",
    )
    valid_ds = WFDataset(
        dataset=f"{dataset}-valid",
        meta_df=valid_df,
        defence=defence_valid_test,
        **kwargs,
    )

    test_mask = ~meta_df.loc[:, assets.TRACE_ID].isin(valid_df.loc[:, assets.TRACE_ID])
    meta_df = meta_df[test_mask]

    test_df = preserve_class_frac_sample(
        meta_df,
        n_samples=n_samples[2],
        random_state=random_state,
        missing_classes="warn",
    )
    test_ds = WFDataset(
        dataset=f"{dataset}-test", meta_df=test_df, defence=defence_valid_test, **kwargs
    )

    return train_ds, valid_ds, test_ds
