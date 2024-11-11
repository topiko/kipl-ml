"""
Various transforms for for flows.
"""

import torch
from kipl_ml.data.assets import assets
from kipl_ml.logging.logger import get_logger

logger = get_logger(__name__)


class _TR:
    def __call__(self, flow: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class Identity(_TR):
    def __call__(self, flow: torch.Tensor) -> torch.Tensor:
        return flow


class Select(_TR):
    def __init__(self, keys: list[str]) -> None:
        self.keys = keys

    def __call__(self, flow: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {k: flow[k] for k in self.keys}


class Drop(_TR):
    def __init__(self, keys: list[str]) -> None:
        self.keys = keys

    def __call__(self, flow: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {k: flow[k] for k in flow.keys() if k not in self.keys}


class DropPacketSize(Drop):
    def __init__(self) -> None:
        super().__init__(keys=[assets.SIZE])


def parse_transforms(transforms: list[str]) -> list[_TR]:
    logger.warning("Transforms not implemented yet.")
    return [Identity()]


def get_tr_seq_outputs(tranforms: list[_TR]) -> list[tuple[str, int]]:
    logger.warning("Transforms not implemented yet.")
    return [(assets.TIME, 100), (assets.DIR, 300), (assets.SIZE, 300)]
