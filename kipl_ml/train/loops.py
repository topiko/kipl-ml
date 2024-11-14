"""
The usual train loops...
"""

from collections.abc import Callable

import torch
from kipl_ml.logging.logger import get_logger
from kipl_ml.logging.utils import key_val_fmt
from kipl_ml.metrics.clf_metrics import (
    ClassMetric,
    GeneralMetric,
    Objective,
)
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

    logger.info(f"Running epoch {n_epoch: 03d}...")

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
    metrics: list[GeneralMetric | ClassMetric],
    early_stop_metric: GeneralMetric | ClassMetric | str = "loss",
    patience: int = 2,
) -> nn.Module:
    if early_stop_metric == "loss":
        best_early_stop_val = float("inf")
        early_stop_metric_str = "loss"
    elif isinstance(early_stop_metric, (GeneralMetric, ClassMetric)):
        if early_stop_metric.OBJECTIVE == Objective.MIN:
            best_early_stop_val = float("inf")
        elif early_stop_metric.OBJECTIVE == Objective.MAX:
            best_early_stop_val = float("-inf")

        early_stop_metric_str = early_stop_metric.name
    else:
        raise ValueError(f"Invalid early_stop_metric {early_stop_metric}")

    logger.info("Training model...")
    logger.info(key_val_fmt("model", model.name))
    logger.info(key_val_fmt("dataset", train_loader.dataset.name))

    epoch = 0
    best_epoch = 0
    while True:
        metrics_vals = evaluate_model(model, valid_loader, metrics, loss_fn)

        if early_stop_metric == "loss":
            early_stop_m_val = metrics_vals["loss"]
            objective = Objective.MIN
        elif isinstance(early_stop_metric, (GeneralMetric, ClassMetric)):
            early_stop_m_val = metrics_vals[early_stop_metric.name]
            objective = early_stop_metric.OBJECTIVE
        else:
            raise ValueError(f"Invalid early_stop_metric: {early_stop_metric}")

        if not isinstance(early_stop_m_val, float):
            raise ValueError(
                f"Early stop metric value must be a float, got {type(early_stop_m_val)}"
            )
        if objective == Objective.MIN:
            if early_stop_m_val < best_early_stop_val:
                best_early_stop_val = early_stop_m_val
                best_epoch = epoch
                # best_model = model.copy()
        elif objective == Objective.MAX:
            if early_stop_m_val > best_early_stop_val:
                best_early_stop_val = early_stop_m_val
                best_epoch = epoch
                # best_model = model.copy()

        logger.info("Valid metrics...")
        logger.info("Current epoch:")
        for k, v in metrics_vals.items():
            logger.info(key_val_fmt(k, f"{v:1.4f}"))
        logger.info("Best:")
        logger.info(
            key_val_fmt(
                early_stop_metric_str,
                f"{best_early_stop_val:1.4f} at epoch {best_epoch: 03d}",
            )
        )

        if epoch - best_epoch >= patience:
            logger.info("Terminate; early stopping")
            break

        epoch += 1
        model = _one_epoch(model, train_loader, optimizer, loss_fn, n_epoch=epoch)

    return model
