from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar

import torch


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
