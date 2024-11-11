import pandas as pd
import torch
from kipl_ml.data.assets import assets
from kipl_ml.data.utils import get_std_flow_array, load_dataset_meta_df
from kipl_ml.flow.transforms import _TR, Identity, get_tr_seq_outputs
from kipl_ml.logging.logger import get_logger
from torch.utils.data import Dataset

logger = get_logger(__name__)


class WFDataset(Dataset):
    def __init__(
        self,
        dataset: str,
        n_packets: int = 500,
        transform: list[_TR] = [Identity()],
        device: torch.device = torch.device("cpu"),
    ) -> None:

        meta_df = load_dataset_meta_df(dataset)
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
        self.transform = transform

    @property
    def n_classes(self) -> int:
        return self.meta_df[assets.LABEL].nunique()

    @property
    def outputs(self) -> list[tuple[str, int]]:
        return get_tr_seq_outputs(self.transform)

    @property
    def device(self):
        return self._device

    @device.setter
    def device(self, device: torch.device) -> None:
        self._device = device

    def __len__(self) -> int:
        return len(self.meta_df)

    def __getitem__(self, idx: int) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        row = self.meta_df.iloc[idx]

        path = row["path"]
        label = row[assets.LABEL]

        flow = get_std_flow_array(path)

        flow = torch.tensor(flow, device=self.device)[: self.n_packets]
        label = torch.tensor(label, device=self.device, dtype=torch.long)

        flow_dict = {
            assets.TIME: flow[:, assets.TIME_IDX],
            assets.DIR: flow[:, assets.DIR_IDX],
            assets.SIZE: flow[:, assets.SIZE_IDX],
        }

        if self.transform is not None:
            flow_dict = self.transform(flow_dict)

        return flow_dict, label
