"""
The usual train loops...
"""

import torch
from kipl_ml.metrics.clf_metrics import Metric, get_objective
from torch import nn
from tqdm import tqdm


def _one_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    n_epoch: int = 0,
):

    with tqdm(dataloader, desc=f"Running epoch {n_epoch: 03d}") as pbar:
        for X, y in pbar:
            optimizer.zero_grad()
            output = model(X)
            loss = loss_fn(output, y)

            loss.backward()
            optimizer.step()

            pbar.set_postfix({"loss": loss.item()})

    return model


def train_model(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    valid_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    metrics: list[Metric],
    early_stopping: str = "loss",
    patience: int = 2,
) -> nn.Module:

    epoch = 0
    objective = get_objective(early_stopping)
    if objective == "min":
        best = float("inf")
    elif objective == "max":
        best = float("-inf")
    best_epoch = 0

    while True:
        metrics_vals = evaluate(model, valid_loader, loss_fn, metrics)

        if early_stopping == "loss":
            early_stop = metrics_vals["loss"]
        else:
            early_stop = metrics_vals[early_stopping]

        if objective == "min":
            if early_stop < best:
                best = early_stop
                best_epoch = epoch
                best_model = model.copy()
        elif objective == "max":
            if early_stop > best:
                best = early_stop
                best_epoch = epoch
                best_model = model.copy()

        if epoch - best_epoch >= patience:
            break

        model = _one_epoch(model, train_loader, optimizer, loss_fn, n_epoch=epoch)

        epoch += 1
    return best_model
