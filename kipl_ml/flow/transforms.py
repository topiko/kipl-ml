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


class Normalize(_TR):
    NAME = "normalized"

    def get_shapes(self, flow: dict[str, torch.Tensor]) -> Normlaize:
        self._input_size = flow[self.asset].shape[0]
        self._output_size = flow[self.asset].shape[0]
        return self

    def __call__(self, flow: dict[str, torch.Tensor]) -> torch.Tensor:
        if (std := flow[self.asset].std()) == 0:
            logger.warning(f"Std zeron when standardizing! {self.name}")
            if all(flow[self.asset] == 0):
                return flow[self.asset]
            raise ValueError("Standard deviation is zero. Cannot normalize.")
        return (flow[self.asset] - flow[self.asset].mean()) / std


class IAT(_TR):
    NAME = "iat"

    def __init__(self, dir_key: str):
        if dir_key not in {"up", "down", "any"}:
            raise ValueError("Dir key must be either 'up' or 'down', or 'any'")
        self.dir_key = dir_key
        super().__init__(assets.TIME)

    @property
    def name(self) -> str:
        if self.dir_key == "any":
            return self.NAME

        return f"{self.dir_key}_{self.NAME}"

    def get_shapes(self, flow: dict[str, torch.Tensor]) -> IAT:
        self._input_size = flow[self.asset].shape[0]
        self._output_size = flow[self.asset].shape[0]
        return self

    def __call__(self, flow: dict[str, torch.Tensor]) -> torch.Tensor:

        if self.dir_key == "up":
            mask = flow[assets.DIR] > 1
        elif self.dir_key == "down":
            mask = flow[assets.DIR] < 1
        else:
            mask = torch.ones_like(flow[assets.DIR], dtype=torch.bool)

        idxs = torch.where(mask)[0]
        iats = torch.zeros_like(flow[assets.TIME])
        iats[idxs[1:]] = torch.diff(flow[assets.TIME][mask], dim=0)
        return iats


class Compose(_TR):
    NAME = "compose"

    def __init__(self, *transforms: _TR):
        self.transforms = transforms
        self._output_size = None
        self._input_size = None

    @property
    def name(self) -> str:
        return "pipe:" + "-->".join(tr.name for tr in self.transforms)

    def get_shapes(self, flow: dict[str, torch.Tensor]) -> Compose:
        outs = None
        for i, tr in enumerate(self.transforms):
            tr.get_shapes(flow)
            outs = tr.output_size
            flow[tr.name] = flow[tr.asset][:outs]

            if i == 0:
                self._input_size = tr.input_size

        self._output_size = outs

        return self

    def __call__(self, flow: dict[str, torch.Tensor]) -> torch.Tensor:
        for i, tr in enumerate(self.transforms):
            tensor = tr(flow)
            if i < len(self.transforms) - 1:
                flow[self.transforms[i].name] = tensor

        return tensor


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
        max_l = max(len(tr.name) for tr in self._feature_trs) + 3
        return "Feature Transforms:\n" + "\n".join(
            [
                key_val_fmt(tr.name, tr.output_size, key_len=max_l)
                for tr in self._feature_trs
            ]
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
        case "iats":
            return IAT("any")
        case "up_iats":
            return IAT("up")
        case "down_iats":
            return IAT("down")
        case "times_normalized":
            return Normalize(assets.TIME)
        case "iats_normalized":
            return Compose(IAT("any"), Normalize("iat"))
        case _:
            raise ValueError(f"Unknown feature name: {feature_name}")
