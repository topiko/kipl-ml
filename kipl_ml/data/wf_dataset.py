import os

import pandas as pd
import torch
from kipl_ml.data.assets import assets
from kipl_ml.data.utils import (
    get_std_flow_array,
    load_dataset_meta_df,
    preserve_class_frac_sample,
)
from kipl_ml.flow.transforms import _TR, FeatureTrs
from kipl_ml.logging.logger import get_logger
from kipl_ml.logging.utils import key_val_fmt
from torch.utils.data import Dataset

logger = get_logger(__name__)


class WFDataset(Dataset):
    def __init__(
        self,
        dataset: str,
        meta_df: pd.DataFrame,
        feature_trs: FeatureTrs,
        n_packets: int = 500,
        device: torch.device = torch.device("cpu"),
        short_flow_policy: str = "pad",
        n_samples: int | None = None,
    ) -> None:
        if short_flow_policy not in ("drop", "pad"):
            raise ValueError(
                f"short_flow_policy must be 'drop' or 'pad', got {short_flow_policy}"
            )

        logger.info("Buidling dataset...")
        logger.info(key_val_fmt("name", dataset))

        self.meta_df = meta_df
        self.name = dataset
        self.device = device
        self.n_packets = n_packets

        if short_flow_policy == "drop":
            packet_mask = self.meta_df.n_packets >= n_packets
            if packet_mask.sum() < len(self.meta_df):
                n_rem = len(self.meta_df) - packet_mask.sum()
                logger.info(
                    f"Removing fraction {n_rem / len(self.meta_df): .2f} "
                    f"of flows due to less than {n_packets} packets."
                )
            self.meta_df = self.meta_df[packet_mask]
        self.feature_trs = feature_trs
        self.get_feature_shapes()
        for _l in self.feature_trs.report().split("\n"):
            logger.info(_l)

    @property
    def n_classes(self) -> int:
        return self.meta_df[assets.LABEL].nunique()

    @property
    def outputs(self) -> dict[str, int]:
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
        X = self._get_flow(self.meta_df.iloc[0]["path"])
        self.feature_trs.get_shapes(X)

    def _get_flow(self, path: os.PathLike) -> dict[str, torch.Tensor]:

        np_flow = get_std_flow_array(path)

        flow = torch.Tensor(np_flow, device=self.device)[: self.n_packets]

        if len(flow) < self.n_packets:
            flow = torch.cat([flow, torch.zeros(self.n_packets - len(flow), 3)])

        return {
            assets.TIME: flow[:, assets.TIME_IDX],
            assets.DIR: flow[:, assets.DIR_IDX],
            assets.SIZE: flow[:, assets.SIZE_IDX],
        }

    def _get_label(self, idx: int) -> torch.Tensor:
        return torch.tensor(
            self.meta_df.iloc[idx][assets.LABEL], device=self.device, dtype=torch.long
        )

    def __getitem__(self, idx: int) -> tuple[dict[str, torch.Tensor], torch.tensor]:
        row = self.meta_df.iloc[idx]

        flow_dict = self._get_flow(row["path"])
        flow_dict = self.feature_trs(flow_dict)

        label = self._get_label(idx)

        return flow_dict, label


def get_train_valid_test(
    dataset: str, n_samples: int | tuple[int, int, int], **kwargs
) -> tuple[WFDataset, WFDataset, WFDataset]:

    if isinstance(n_samples, int):
        n_samples = (n_samples,) * 3

    meta_df = load_dataset_meta_df(dataset)

    train_df = preserve_class_frac_sample(meta_df, n_samples[0])
    train_ds = WFDataset(dataset=f"{dataset}-train", meta_df=train_df, **kwargs)

    valid_mask = ~meta_df.loc[:, assets.FLOW_ID].isin(train_df.loc[:, assets.FLOW_ID])
    meta_df = meta_df[valid_mask]

    valid_df = preserve_class_frac_sample(
        meta_df, n_samples=n_samples[1], missing_classes="warn"
    )
    valid_ds = WFDataset(dataset=f"{dataset}-valid", meta_df=valid_df, **kwargs)

    test_mask = ~meta_df.loc[:, assets.FLOW_ID].isin(valid_df.loc[:, assets.FLOW_ID])
    meta_df = meta_df[test_mask]

    test_df = preserve_class_frac_sample(
        meta_df, n_samples=n_samples[2], missing_classes="warn"
    )
    test_ds = WFDataset(dataset=f"{dataset}-test", meta_df=test_df, **kwargs)

    return train_ds, valid_ds, test_ds
