"""
Base class for all transforms.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch


class _TR(ABC):
    NAME: str
    _output_sizes: dict[str, int] | None = None

    @property
    def output_sizes(self) -> dict[str, int]:
        if self._output_sizes is None:
            raise ValueError("Output size not set. Need to call 'get_shapes(X)' first.")
        return self._output_sizes

    @property
    def name(self) -> str:
        return self.NAME + f"_{self.asset}"

    @property
    def output(self) -> str:
        return self.name

    @abstractmethod
    def get_shapes(self, trace: dict[str, torch.Tensor]) -> _TR:
        raise NotImplementedError

    @abstractmethod
    def __call__(self, trace: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        raise NotImplementedError
