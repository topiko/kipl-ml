"""
The usual train loops...
"""

from collections.abc import Callable
from copy import deepcopy

import mlflow
import torch
from kipl_ml.data.wf_dataset import dict_to_device
from kipl_ml.logging.logger import TQDM_W, get_logger
from kipl_ml.logging.utils import key_val_fmt
from kipl_ml.metrics.clf_metrics import ClassMetric, GeneralMetric, Objective
from kipl_ml.model_eval.evaluate import evaluate_model
from torch import nn
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau
from tqdm import tqdm

logger = get_logger(__name__)


def _get_val_and_metric_str(
    early_stop_metric: GeneralMetric | ClassMetric | str,
) -> tuple[str, float]:
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

    return early_stop_metric_str, best_early_stop_val


def _one_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: Callable,
    lr_scheduler: ReduceLROnPlateau | LambdaLR | None = None,
    n_epoch: int = 0,
) -> tuple[nn.Module, float]:
    """
    Run one epoch of training.

    Args:
        model:
        dataloader:
        optimizer:
        loss_fn:
        n_epoch:

    Returns:
        model: nn.Module
        train_loss: float

    """

    logger.info(f"Running epoch {n_epoch: 03d}...")

    if lr_scheduler and not isinstance(lr_scheduler, (ReduceLROnPlateau, LambdaLR)):
        raise ValueError(
            f"Invalid scheduler {lr_scheduler}, must be ReduceLROnPlateau or LambdaLR"
        )

    if torch.cuda.is_available():
        logger.info("Found cuda device - training on it")
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model.train()
    model.to(device)

    loss_val = 0
    with tqdm(dataloader, desc=f"epoch {n_epoch: 03d}", ncols=TQDM_W) as pbar:
        for X, y in pbar:

            X = dict_to_device(X, device)
            y = y.to(device)

            optimizer.zero_grad()
            output = model(X)
            loss = loss_fn(output, y)

            loss.backward()
            optimizer.step()

            if isinstance(lr_scheduler, ReduceLROnPlateau):
                lr_scheduler.step(loss.item())
            elif isinstance(lr_scheduler, LambdaLR):
                lr_scheduler.step()
            else:
                pass

            lr = optimizer.param_groups[0]["lr"]
            pbar.set_postfix({"loss": f"{loss.item():1.4f}", "lr": f"{lr:1.4e}"})
            # TODO: do this prpoerly, the last batch is smaller than the rest
            # however that effect should be insignificant.
            loss_val += loss.item() * dataloader.batch_size

    loss_val /= len(dataloader.dataset)

    model.to("cpu")
    return model, loss_val


def train_model(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    valid_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: Callable,
    metrics: list[GeneralMetric | ClassMetric],
    lr_scheduler: ReduceLROnPlateau | LambdaLR | None = None,
    early_stop_metric: GeneralMetric | ClassMetric | str = "loss",
    patience: int = 2,
) -> nn.Module:
    logger.info("Training model...")
    logger.info(key_val_fmt("model", model.name, suffix=""))
    logger.info(key_val_fmt("dataset", train_loader.dataset.name, suffix=""))

    early_stop_metric_str, best_early_stop_val = _get_val_and_metric_str(
        early_stop_metric
    )

    epoch = 0
    best_epoch = 0
    best_model_state = None
    train_loss = None
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

        for m, mv in metrics_vals.items():
            if epoch == 0 and "loss" in m:
                continue
            if isinstance(mv, float):
                mlflow.log_metric(f"valid_{m}", mv, step=epoch)

        if objective == Objective.MIN:
            if early_stop_m_val < best_early_stop_val:
                best_early_stop_val = early_stop_m_val
                best_epoch = epoch
                best_model_state = deepcopy(model.state_dict())
        elif objective == Objective.MAX:
            if early_stop_m_val > best_early_stop_val:
                best_early_stop_val = early_stop_m_val
                best_epoch = epoch
                best_model_state = deepcopy(model.state_dict())

        logger.info("Valid metrics...")
        logger.info("Current epoch:")
        for k, v in metrics_vals.items():
            logger.info(key_val_fmt(k, f"{v:1.4f}", suffix=""))
        logger.info("Best:")
        logger.info(
            key_val_fmt(
                early_stop_metric_str,
                f"{best_early_stop_val:1.4f} at epoch {best_epoch:d} (/ {epoch:d})",
                suffix="",
            )
        )

        if epoch - best_epoch >= patience:
            logger.info("Terminate; early stopping")
            break

        epoch += 1
        model, train_loss = _one_epoch(
            model, train_loader, optimizer, loss_fn, lr_scheduler, n_epoch=epoch
        )
        mlflow.log_metric("train_loss", train_loss, step=epoch)

        logger.info(key_val_fmt("Train loss", f"{train_loss:1.4f}", suffix=""))

    if best_model_state is None:
        raise ValueError("No best model state found.")

    model.load_state_dict(best_model_state, strict=True)

    metrics_vals = evaluate_model(model, valid_loader, metrics, loss_fn)

    return model
