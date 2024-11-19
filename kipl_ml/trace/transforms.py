"""
Base class for all transforms.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch


class _TR(ABC):
    NAME: str

    def __init__(self, asset: str):
        self.asset = asset
        self._output_size: int | None = None
        self._input_size: int | None = None

    @property
    def input_size(self) -> int:
        if self._input_size is None:
            raise ValueError("Input size not set. Need to call fit first.")
        return self._input_size

    @property
    def output_size(self) -> int:
        if self._output_size is None:
            raise ValueError("Output size not set. Need to call fit first.")
        return self._output_size

    @property
    def name(self) -> str:
        return self.NAME + f"_{self.asset}"

    @abstractmethod
    def get_shapes(self, trace: dict[str, torch.Tensor]) -> _TR:
        raise NotImplementedError

    @abstractmethod
    def __call__(self, trace: dict[str, torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError
