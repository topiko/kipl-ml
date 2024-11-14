"""
Classification metrics.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar

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
    PROBS: str = "probs"


class GeneralMetric(ABC):
    OBJECTIVE: ClassVar[str]
    PRED_TYPE: ClassVar[str]

    @property
    def name(self) -> str:
        return self.__class__.__name__.lower()

    @abstractmethod
    def __call__(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
        raise NotImplementedError


class ClassMetric(ABC):
    OBJECTIVE: ClassVar[str]
    PRED_TYPE: ClassVar[str]
    CLASS_IDX: int

    def __init__(self, class_idx: int):
        self.CLASS_IDX = class_idx

    @property
    def name(self):
        return f"{self.CLASS_IDX:04d} - {self.__class__.__name__.lower()}"

    @abstractmethod
    def __call__(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
        raise NotImplementedError


class CrossEntropyLoss(GeneralMetric):
    OBJECTIVE: ClassVar[str] = Objective.MIN
    PRED_TYPE: ClassVar[str] = PredType.LOGITS

    def __init__(self, *args, **kwargs):
        self.loss = nn.CrossEntropyLoss(*args, **kwargs)

    def __call__(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
        return self.loss(y_pred, y_true).item()


class Accuracy(GeneralMetric):
    OBJECTIVE: ClassVar[str] = Objective.MAX
    PRED_TYPE: ClassVar[str] = PredType.CLASSES

    def __call__(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
        return (y_pred == y_true).float().mean().item()


def _recall(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    tp = (y_pred == y_true).sum()
    fn = (y_pred != y_true).sum()
    return (tp / (tp + fn)).item()


class Recall(GeneralMetric):
    OBJECTIVE: ClassVar[str] = Objective.MAX
    PRED_TYPE: ClassVar[str] = PredType.CLASSES

    def __call__(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
        return _recall(y_pred, y_true)


class ClassRecall(ClassMetric):
    OBJECTIVE: ClassVar[str] = Objective.MAX
    PRED_TYPE: ClassVar[str] = PredType.CLASSES
    CLASS_IDX: int

    def __call__(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
        mask = y_true == self.CLASS_IDX
        y_true_ = y_true[mask]
        y_pred_ = y_pred[mask]

        return _recall(y_pred_, y_true_)
