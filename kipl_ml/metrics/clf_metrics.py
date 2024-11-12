"""
Classification metrics.
"""

from typing import ClassVar, Protocol

import torch
from torch import nn


class Metric(Protocol):
    name: ClassVar[str]
    objective: ClassVar[str]
    pred_type: ClassVar[str]

    def __call__(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
        raise NotImplementedError


class CrossEntropyLoss:
    name: str = "CrossEntropyLoss"
    objective: str = "min"
    pred_type: str = "logits"

    def __init__(self, *args, **kwargs):
        self.loss = nn.CrossEntropyLoss(*args, **kwargs)

    def __call__(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
        return self.loss(y_pred, y_true).item()


class Accuracy:
    name: str = "accuracy"
    objective: str = "max"
    pred_type: str = "classes"

    def __call__(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
        return (y_pred == y_true).float().mean().item()


METRICS: set[Metric] = {Accuracy, CrossEntropyLoss}


def get_objective(metric_name: str) -> str:
    for metric in METRICS:
        if metric.name == metric_name:
            return metric.objective
    raise ValueError(f"Metric {metric_name} not found.")


def evaluate(
    metrics: list[Metric], y_pred: torch.Tensor, y_true: torch.Tensor
) -> dict[str, float]:
    return {metric.name: metric(y_pred, y_true) for metric in metrics}
