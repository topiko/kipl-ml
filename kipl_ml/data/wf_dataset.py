import os

import pandas as pd
import torch
from kipl_ml.data.assets import assets
from kipl_ml.data.utils import get_std_flow_array, load_dataset_meta_df
from kipl_ml.flow.transforms import _TR, FeatureTrs
from kipl_ml.logging.logger import get_logger
from kipl_ml.logging.utils import key_val_fmt
from torch.utils.data import Dataset

logger = get_logger(__name__)


class WFDataset(Dataset):
    def __init__(
        self,
        dataset: str,
        feature_trs: FeatureTrs,
        n_packets: int = 500,
        device: torch.device = torch.device("cpu"),
    ) -> None:

        logger.info("Buidling dataset...")
        logger.info(key_val_fmt("name", dataset))

        meta_df = load_dataset_meta_df(dataset)
        self.name = dataset
        self.device = device
        self.n_packets = n_packets

        packet_mask = meta_df.n_packets >= n_packets

        if packet_mask.sum() < len(meta_df):
            n_rem = len(meta_df) - packet_mask.sum()
            logger.info(
                f"Removing fraction {n_rem / len(meta_df): .2f} "
                f"of flows due to less than {n_packets} packets."
            )
        self.meta_df = meta_df[packet_mask]
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
