from collections.abc import Callable

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from kipl_ml.data.assets import PAGE_LABEL, PRED, PRED_CLS_PROB
from kipl_ml.data.wf_dataset import dict_to_device
from kipl_ml.logging.logger import TQDM_W, get_logger
from kipl_ml.metrics.clf_metrics import ClassMetric, GeneralMetric, PredType
from kipl_ml.tools.cuda_tools import get_device

logger = get_logger(__name__)


def get_clf_df(model: nn.Module, dataloader: DataLoader) -> pd.DataFrame:
    no_shuffle_dl = DataLoader(
        dataloader.dataset, batch_size=dataloader.batch_size, shuffle=False
    )

    logits, pred_class, y_true = run_inference(model, no_shuffle_dl)

    pred_probs = torch.softmax(logits, dim=1)
    pred_class = logits.argmax(dim=1)

    df = dataloader.dataset.meta_df.copy()

    df.loc[:, PRED] = pred_class.cpu().numpy()
    df.loc[:, PRED_CLS_PROB] = pred_probs.cpu().numpy().tolist()
    df.loc[:, PAGE_LABEL] = y_true.cpu().numpy()

    return df


def run_inference(
    model: nn.Module, dataloader: DataLoader
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    logger.info("Run inference...")
    model.eval()
    model.to(get_device())

    logits = []
    preds = []
    labels = []
    with torch.no_grad():
        with tqdm(dataloader, ncols=TQDM_W) as pbar:
            for X, y in pbar:
                X_ = dict_to_device(X, get_device())
                logits_, preds_ = model.predict(X_)
                logits.append(logits_)
                preds.append(preds_)
                labels.append(y)

    logits = torch.cat(logits).to("cpu")
    preds = torch.cat(preds).to("cpu")
    y_true = torch.cat(labels).to("cpu")

    return logits, preds, y_true


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    metrics: list[GeneralMetric | ClassMetric],
    loss_fn: Callable | None = None,
    key: str | None = None,
) -> dict[str, float | torch.Tensor]:
    logger.info(f"Evaluate... {dataloader.dataset.name}")

    logits, pred_class, y_true = run_inference(model, dataloader)

    metric_vals: dict[str, float | torch.Tensor] = {}
    for m in metrics:
        if m.PRED_TYPE == PredType.CLASSES:
            metric_vals[m.name] = m(y_pred=pred_class, y_true=y_true)
        elif m.PRED_TYPE == PredType.LOGITS:
            metric_vals[m.name] = m(y_pred=logits, y_true=y_true)
        else:
            raise ValueError(f"Unknown prediction type {m.PRED_TYPE}")

    if loss_fn is not None:
        loss = loss_fn(logits, y_true)
        metric_vals["loss"] = loss.item()

    if key is not None:
        metric_vals = {f"{key}-{k}": v for k, v in metric_vals.items()}
    return metric_vals
