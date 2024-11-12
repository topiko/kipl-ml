from collections.abc import Callable

import torch
from kipl_ml.logging.logger import get_logger
from kipl_ml.metrics.clf_metrics import Metric
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

logger = get_logger(__name__)


def run_inference(
    model: nn.Module, dataloader: torch.utils.data.DataLoader
) -> tuple[torch.Tensor, torch.Tensor]:
    logger.info(f"Run inference...")
    model.eval()

    preds = []
    labels = []
    with torch.no_grad():
        with tqdm(dataloader) as pbar:
            for X, y in pbar:
                preds.append(model(X))
                labels.append(y)

    pred_y_prob = torch.cat(preds)
    y_true = torch.cat(labels)
    return pred_y_prob, y_true


def evaluate_model(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    metrics: list[Metric],
    loss_fn: Callable | None = None,
) -> dict[str, float | torch.Tensor]:
    logger.info(f"Evaluate...")

    y_prob, y_true = run_inference(model, dataloader)

    pred_class = y_prob.argmax(dim=1)

    metric_vals: dict[str, float | torch.Tensor] = {}
    for m in metrics:
        if m.pred_type == "classes":
            metric_vals[m.name] = m(y_pred=pred_class, y_true=y_true)
        elif m.pred_type == "logits":
            metric_vals[m.name] = m(y_pred=y_prob, y_true=y_true)
        else:
            raise ValueError(f"Unknown prediction type {m.pred_type}")

    if loss_fn is not None:
        loss = loss_fn(y_prob, y_true)
        metric_vals["loss"] = loss.item()

    return metric_vals
