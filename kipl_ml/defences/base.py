import os
from abc import ABC, abstractmethod

import torch
from kipl_ml.data.utils import get_std_trace_dict
from kipl_ml.logging.logger import get_logger
from kipl_ml.logging.utils import key_val_fmt

logger = get_logger(__name__)


class _Def(ABC):
    def _report(self, to_log: bool = True, **kwargs) -> str:
        str_ = f"{self.name}\n"
        for key, value in kwargs.items():
            str_ += key_val_fmt(key, value) + "\n"

        if to_log:
            for line in str_.split("\n"):
                logger.info(line)

        return str_

    @property
    def name(self) -> str:
        return self.__class__.__name__

    def __call__(self, trace_path: os.PathLike) -> dict[str, torch.Tensor]:
        if not isinstance(trace_path, os.PathLike):
            raise TypeError(
                f"Expected trace to be os.PathLike, got: {type(trace_path)}"
            )

        return self._simulate(trace_path)

    def load_data(self, trace_path: os.PathLike) -> dict[str, torch.Tensor]:
        return get_std_trace_dict(trace_path)

    @abstractmethod
    def _simulate(self, trace_path: os.PathLike) -> dict[str, torch.Tensor]:
        raise NotImplementedError

    @abstractmethod
    def report(self, to_log: bool = True) -> str:
        raise NotImplementedError


class NoDefence(_Def):

    def report(self, to_log: bool = True) -> str:
        str_ = "No defence applied"
        if to_log:
            logger.info(str_)
        return str_

    def _simulate(self, trace_path: os.PathLike) -> dict[str, torch.Tensor]:
        return self.load_data(trace_path)
