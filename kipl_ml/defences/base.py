from abc import ABC, abstractmethod

import torch
from kipl_ml.logging.logger import get_logger
from kipl_ml.logging.utils import key_val_fmt

logger = get_logger(__name__)


class _Def(ABC):
    def _report(self, to_log: bool = True, **kwargs) -> str:
        str_ = f"{self.name}\n"
        for key, value in kwargs.items():
            str_ += key_val_fmt(key, value)

        if to_log:
            logger.info(str_)

        return str_

    @property
    def name(self) -> str:
        return self.__class__.__name__

    @abstractmethod
    def report(self, to_log: bool = True) -> str:
        raise NotImplementedError

    @abstractmethod
    def __call__(self, trace: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        raise NotImplementedError
