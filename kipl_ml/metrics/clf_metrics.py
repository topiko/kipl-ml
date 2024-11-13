"""
Classification metrics.
"""

from dataclasses import dataclass
from typing import ClassVar, Protocol

import torch
from torch import nn


@dataclass
class Objective:
    MIN: str = "min"
    MAX: str = "max"


@dataclass
class PredType:
    CLASSES: str = "classes"
    LOGITS: str = "logits"


@dataclass
class CLFMetrics:
    ACCURACY: str = "accuracy"
    CROSS_ENTROPY_LOSS: str = "CrossEntropyLoss"


class Metric(Protocol):
    name: ClassVar[str]
    objective: ClassVar[str]
    pred_type: ClassVar[str]

    def __call__(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
        raise NotImplementedError


class CrossEntropyLoss:
    name: str = CLFMetrics.CROSS_ENTROPY_LOSS
    objective: str = Objective.MIN
    pred_type: str = PredType.LOGITS

    def __init__(self, *args, **kwargs):
        self.loss = nn.CrossEntropyLoss(*args, **kwargs)

    def __call__(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
        return self.loss(y_pred, y_true).item()


class Accuracy:
    name: str = CLFMetrics.ACCURACY
    objective: str = Objective.MAX
    pred_type: str = PredType.CLASSES

    def __call__(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
        return (y_pred == y_true).float().mean().item()


METRICS: set[Metric] = {Accuracy, CrossEntropyLoss}


def get_objective(metric_name: str) -> str:
    for metric in METRICS:
        if metric.name == metric_name:
            return metric.objective
    raise ValueError(f"Metric {metric_name} not found.")
