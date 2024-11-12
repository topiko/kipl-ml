"""
The usual train loops...
"""

from collections.abc import Callable

import torch
from kipl_ml.logging.logger import get_logger
from kipl_ml.metrics.clf_metrics import Metric, get_objective
from kipl_ml.model_eval.evaluate import evaluate_model
from torch import nn
from tqdm import tqdm

logger = get_logger(__name__)


def _one_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    n_epoch: int = 0,
):

    logger.info(f"Running epoch {n_epoch: 03d}")

    model.train()
    with tqdm(dataloader, desc=f"epoch {n_epoch: 03d}") as pbar:
        for X, y in pbar:
            optimizer.zero_grad()
            output = model(X)
            loss = loss_fn(output, y)

            loss.backward()
            optimizer.step()

            pbar.set_postfix({"loss": f"{loss.item():1.4f}"})

    return model


def train_model(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    valid_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: Callable,
    metrics: list[Metric],
    early_stop_metric: str = "loss",
    patience: int = 2,
) -> nn.Module:

    objective = get_objective(early_stop_metric)
    if objective == "min":
        best_early_stop_val = float("inf")
    elif objective == "max":
        best_early_stop_val = float("-inf")

    epoch = 0
    best_epoch = 0
    while True:
        metrics_vals = evaluate_model(model, valid_loader, metrics, loss_fn)

        if early_stop_metric == "loss":
            early_stop_m_val = metrics_vals["loss"]
        else:
            early_stop_m_val = metrics_vals[early_stop_metric]

        if objective == "min":
            if early_stop_m_val < best_early_stop_val:
                best_early_stop_val = early_stop_m_val
                best_epoch = epoch
                # best_model = model.copy()
        elif objective == "max":
            if early_stop_m_val > best_early_stop_val:
                best_early_stop_val = early_stop_m_val
                best_epoch = epoch
                # best_model = model.copy()

        logger.info(f"{early_stop_metric}: {early_stop_m_val:1.4f}")
        logger.info(
            f"Current best {early_stop_metric}: {best_early_stop_val:1.4f} at epoch {best_epoch}"
        )
        if epoch - best_epoch >= patience:
            logger.info("Terminate; early stopping")
            break

        epoch += 1
        model = _one_epoch(model, train_loader, optimizer, loss_fn, n_epoch=epoch)

    return best_model
