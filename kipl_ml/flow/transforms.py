"""
Various transforms for for flows.
"""

import torch
from kipl_ml.flow.assets import assets


class _Base:
    def __call__(self, flow: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class Select(_Base):
    def __init__(self, keys: list[str]) -> None:
        self.keys = keys

    def __call__(self, flow: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {k: flow[k] for k in self.keys}


class Drop(_Base):
    def __init__(self, keys: list[str]) -> None:
        self.keys = keys

    def __call__(self, flow: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {k: flow[k] for k in flow.keys() if k not in self.keys}


class DropPacketSize(Drop):
    def __init__(self) -> None:
        super().__init__(keys=[assets.SIZE])
