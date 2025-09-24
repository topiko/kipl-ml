"""
Classification metrics.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar

import pandas as pd
import torch
from torch import nn

from kipl_ml.data.assets import PAGE_LABEL, PRED
from kipl_ml.logging.logger import get_logger

logger = get_logger(__name__)


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
    # Note! This is to be called from within ClassRecall or Recall only!!!
    # See handling of mask in those classes.
    tp = (y_pred == y_true).sum()
    fn = (y_pred != y_true).sum()
    return (tp / (tp + fn)).item()


class Recall(GeneralMetric):
    OBJECTIVE: ClassVar[str] = Objective.MAX
    PRED_TYPE: ClassVar[str] = PredType.CLASSES

    def __call__(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
        if (nc := torch.unique(y_true).numel()) > 2:
            logger.warning(
                f"Overall recall is not very meaningful for multi-class {nc} problems."
            )
        mask = y_true != 0
        return _recall(y_pred[mask], y_true[mask])


class ClassRecall(ClassMetric):
    OBJECTIVE: ClassVar[str] = Objective.MAX
    PRED_TYPE: ClassVar[str] = PredType.CLASSES
    CLASS_IDX: int

    def __call__(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
        mask = y_true == self.CLASS_IDX
        y_true_ = y_true[mask]
        y_pred_ = y_pred[mask]

        return _recall(y_pred_, y_true_)


def metric_from_df(df: pd.DataFrame, metric: ClassMetric) -> float:
    preds = torch.tensor(df.loc[:, PRED].values, dtype=torch.int64)
    labels = torch.tensor(df.loc[:, PAGE_LABEL].values, dtype=torch.int64)

    return metric(preds, labels)
