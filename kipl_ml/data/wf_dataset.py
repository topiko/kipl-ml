import kipl_ml.data.assets as assets
import pandas as pd
import torch
from kipl_ml.data.utils import (
    get_std_trace_array,
    load_dataset_meta_df,
    preserve_class_frac_sample,
)
from kipl_ml.logging.logger import get_logger
from kipl_ml.logging.utils import key_val_fmt
from kipl_ml.trace.features import FeatureTrs
from kipl_ml.trace.transforms import _TR
from torch.utils.data import Dataset

logger = get_logger(__name__)


class WFDataset(Dataset):
    def __init__(
        self,
        dataset: str,
        meta_df: pd.DataFrame,
        feature_trs: FeatureTrs,
        device: torch.device = torch.device("cpu"),
        short_trace_policy: str = "pad",
        n_samples: int | None = None,
    ) -> None:
        if short_trace_policy not in ("drop", "pad"):
            raise ValueError(
                f"short_trace_policy must be 'drop' or 'pad', got {short_trace_policy}"
            )

        logger.info("Buidling dataset...")
        logger.info(key_val_fmt("name", dataset))

        self.meta_df = meta_df
        self.name = dataset
        self.device = device

        if short_trace_policy == "drop":
            raise NotImplementedError("short_trace_policy='drop' not implemented.")
        self.feature_trs = feature_trs
        self.get_feature_shapes()
        for _l in self.feature_trs.report().split("\n"):
            logger.info(_l)

    @property
    def n_classes(self) -> int:
        return self.meta_df[assets.LABEL].nunique()

    @property
    def output_sizes(self) -> dict[str, int]:
        return self.feature_trs.output_sizes

    @property
    def device(self):
        return self._device

    @device.setter
    def device(self, device: torch.device) -> None:
        self._device = device

    def __len__(self) -> int:
        return len(self.meta_df)

    def get_feature_shapes(self) -> None:
        X = self._get_trace(0)
        self.feature_trs.get_shapes(X)

    def _get_trace(self, idx: int) -> dict[str, torch.Tensor]:

        path = self.meta_df.iloc[idx]["path"]

        np_trace = get_std_trace_array(path)

        trace = torch.Tensor(np_trace, device=self.device)

        return {
            assets.TIMES: trace[:, assets.TIMES_IDX],
            assets.DIRS: trace[:, assets.DIRS_IDX],
            assets.SIZES: trace[:, assets.SIZES_IDX],
        }

    def _get_label(self, idx: int) -> torch.Tensor:
        return torch.tensor(
            self.meta_df.iloc[idx][assets.LABEL], device=self.device, dtype=torch.long
        )

    def __getitem__(self, idx: int) -> tuple[dict[str, torch.Tensor], torch.tensor]:

        trace_dict = self._get_trace(idx)
        trace_dict = self.feature_trs(trace_dict)

        label = self._get_label(idx)

        return trace_dict, label


def get_train_valid_test(
    dataset: str, n_samples: int | tuple[int, int, int], **kwargs
) -> tuple[WFDataset, WFDataset, WFDataset]:

    if isinstance(n_samples, int):
        n_samples = (n_samples,) * 3

    meta_df = load_dataset_meta_df(dataset)

    train_df = preserve_class_frac_sample(meta_df, n_samples[0])
    train_ds = WFDataset(dataset=f"{dataset}-train", meta_df=train_df, **kwargs)

    valid_mask = ~meta_df.loc[:, assets.TRACE_ID].isin(train_df.loc[:, assets.TRACE_ID])
    meta_df = meta_df[valid_mask]

    valid_df = preserve_class_frac_sample(
        meta_df, n_samples=n_samples[1], missing_classes="warn"
    )
    valid_ds = WFDataset(dataset=f"{dataset}-valid", meta_df=valid_df, **kwargs)

    test_mask = ~meta_df.loc[:, assets.TRACE_ID].isin(valid_df.loc[:, assets.TRACE_ID])
    meta_df = meta_df[test_mask]

    test_df = preserve_class_frac_sample(
        meta_df, n_samples=n_samples[2], missing_classes="warn"
    )
    test_ds = WFDataset(dataset=f"{dataset}-test", meta_df=test_df, **kwargs)

    return train_ds, valid_ds, test_ds
