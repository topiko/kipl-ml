import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from kipl_ml.data.wf_dataset import dict_to_device
from kipl_ml.logging.logger import TQDM_W, get_logger
from kipl_ml.metrics.overhead_metrics import _BurstOverheadLoss
from kipl_ml.tools.cuda_tools import get_device

logger = get_logger(__name__)


def evaluate_obs(
    obsfuscator: nn.Module,
    dataloader: DataLoader,
    metrics: list[_BurstOverheadLoss],
    key: str | None = None,
    model: nn.Module | None = None,
) -> dict[str, float | torch.Tensor]:
    logger.info(f"Evaluate... {dataloader.dataset.name}")

    metric_vals = {m.name: 0.0 for m in metrics}
    with torch.no_grad():
        with tqdm(dataloader, ncols=TQDM_W) as pbar:
            n = 1
            for X, _ in pbar:
                X = dict_to_device(X, get_device())
                Xobs = obsfuscator(X)

                for m in metrics:
                    mv = metric_vals[m.name]
                    metric_vals[m.name] = mv + m(Xobs, X).item() / n

                n += 1

    if key is not None:
        metric_vals = {f"{key}-{k}": v for k, v in metric_vals.items()}
    return metric_vals
