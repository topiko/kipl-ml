"""
Various transforms for for flows.
"""

from __future__ import annotations

import torch
from kipl_ml.data.assets import assets
from kipl_ml.logging.logger import get_logger
from kipl_ml.logging.utils import key_val_fmt

logger = get_logger(__name__)


class _TR:
    NAME: str

    def __init__(self, asset: str):
        self.asset = asset
        self._output_size: int | None = None
        self._input_size: int | None = None

    @property
    def output_size(self) -> int:
        if self._output_size is None:
            raise ValueError("Output size not set. Need to call fit first.")
        return self._output_size

    @property
    def name(self) -> str:
        return self.NAME + f"_{self.asset}"

    def get_shapes(self, flow: dict[str, torch.Tensor]) -> _TR:
        raise NotImplementedError

    def __call__(self, flow: dict[str, torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError


class Select(_TR):
    NAME = "identity"

    def get_shapes(self, flow: dict[str, torch.Tensor]) -> Select:
        self._input_size = flow[self.asset].shape[0]
        self._output_size = flow[self.asset].shape[0]
        return self

    def __call__(self, flow: dict[str, torch.Tensor]) -> torch.Tensor:
        return flow[self.asset]


class UDPackets(_TR):
    NAME = "up/down_packets"

    def __init__(self, up_down: str):
        if up_down not in {"up", "down"}:
            raise ValueError("up_down must be either 'up' or 'down'")
        self.up_down = up_down
        super().__init__(assets.DIR)

    @property
    def name(self) -> str:
        return self.up_down + "_packets"

    def get_shapes(self, flow: dict[str, torch.Tensor]) -> UpPackets:
        self._input_size = flow[self.asset].shape[0]
        self._output_size = flow[self.asset].shape[0]
        return self

    def __call__(self, flow: dict[str, torch.Tensor]) -> torch.Tensor:
        if self.up_down == "up":
            mask = flow[self.asset] > 1
        elif self.up_down == "down":
            mask = flow[self.asset] < 1
        return mask.float()


class FeatureTrs:
    def __init__(
        self,
        feature_trs: list[_TR] | None = None,
        feature_names: list[str] | None = None,
    ):
        if feature_trs is None and feature_names is None:
            raise ValueError(
                "Either 'feature_trs' or 'feature_names' must be provided."
            )
        if feature_trs is None:
            feature_trs = build_feature_trs(feature_names)
        elif not all(isinstance(tr, _TR) for tr in feature_trs):
            raise ValueError("All elements in 'feature_trs' must be of type _TR.")

        self._feature_trs = feature_trs

    def get_shapes(self, flow: dict[str, torch.Tensor]) -> FeatureTrs:
        for tr in self._feature_trs:
            tr.get_shapes(flow)
        return self

    @property
    def output_sizes(self) -> dict[str, int]:
        return {tr.name: tr.output_size for tr in self._feature_trs}

    @property
    def features(self) -> list[str]:
        return [tr.name for tr in self._feature_trs]

    def report(self) -> str:

        return "Feature Transforms:\n" + "\n".join(
            [key_val_fmt(tr.name, tr.output_size) for tr in self._feature_trs]
        )

    def __call__(self, flow: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {tr.name: tr(flow) for tr in self._feature_trs}


def build_feature_trs(feature_name: list[str]) -> list[_TR]:
    return [get_feature_tr(f) for f in feature_name]


def get_feature_tr(feature_name: str) -> _TR:
    match feature_name:
        case "dirs":
            return Select(assets.DIR)
        case "sizes":
            return Select(assets.SIZE)
        case "times":
            return Select(assets.TIME)
        case "up_packets":
            return UDPackets("up")
        case "down_packets":
            return UDPackets("down")
        case _:
            raise ValueError(f"Unknown feature name: {feature_name}")
