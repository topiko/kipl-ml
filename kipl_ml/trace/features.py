from __future__ import annotations

import kipl_ml.data.assets as assets
import mlflow
import torch
from kipl_ml.logging.logger import get_logger
from kipl_ml.logging.utils import key_val_fmt
from kipl_ml.trace.transforms import _TR
from mlflow.models import set_model

logger = get_logger(__name__)


def _pad_short_trace(
    trace: torch.Tensor, n_packets: int, asset_key: str, cut: bool = True
) -> torch.Tensor:
    if cut and (trace.shape[0] >= n_packets):
        return trace[:n_packets]

    if asset_key == assets.TIMES:
        pad_val = trace[-1].item()
    elif asset_key in {assets.DIRS, assets.SIZES}:
        pad_val = 0.0
    elif asset_key == assets.ORIG_PACKETS:
        pad_val = 0
    else:
        raise ValueError(f"Unknown asset key: {asset_key}")

    return torch.cat(
        [
            trace,
            torch.ones(
                n_packets - trace.shape[0],
                *trace.shape[1:],
                dtype=trace.dtype,
                device=trace.device,
            )
            * pad_val,
        ]
    )


class PadOrCutTrace(_TR):
    NAME = "sel_packets"

    def __init__(self, n_packets: int):
        self.n_packets = n_packets

    @property
    def name(self) -> str:
        return f"cut|{self.n_packets}"

    def get_shapes(self, trace: dict[str, torch.Tensor]) -> PadOrCutTrace:
        self._output_sizes = {key: self.n_packets for key in trace.keys()}

        return self

    def __call__(self, trace: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:

        trace_ = {
            key: _pad_short_trace(val, self.n_packets, asset_key=key)
            for key, val in trace.items()
        }

        return trace_


class Select(_TR):
    NAME = "identity"

    def __init__(self, asset: str):
        self.asset = asset

    def get_shapes(self, trace: dict[str, torch.Tensor]) -> Select:
        self._output_sizes = {self.asset: trace[self.asset].shape[0]}
        return self

    def __call__(self, trace: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {self.asset: trace[self.asset]}


class UDPackets(_TR):
    NAME = "up/down_packets"

    def __init__(self, up_down: str, dir_asset: str = assets.DIRS):
        if up_down not in {"up", "down"}:
            raise ValueError("up_down must be either 'up' or 'down'")
        self.up_down = up_down
        self.dir_asset = dir_asset

    @property
    def name(self) -> str:
        return self.up_down + "_packets"

    def get_shapes(self, trace: dict[str, torch.Tensor]) -> UDPackets:
        self._output_sizes = {self.name: trace[self.dir_asset].shape[0]}
        return self

    def __call__(self, trace: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        if self.up_down == "up":
            mask = trace[self.dir_asset] == 1
        elif self.up_down == "down":
            mask = trace[self.dir_asset] == -1

        return {self.name: mask.float()}


class Normalize(_TR):
    NAME = "normalized"

    def __init__(self, normalized_asset: str, input_asset: str, division: str = "std"):

        if division not in {"std", "max"}:
            raise ValueError("Division must be either 'std' or 'max'")

        self.normalized_asset = normalized_asset
        self.input_asset = input_asset
        self.division = division

    @property
    def name(self) -> str:
        match self.normalized_asset:
            case assets.TIMES:
                name_ = assets.TIMES_NORMALIZED
            case assets.IATS:
                name_ = assets.IATS_NORMALIZED
            case assets.UP_IATS:
                name_ = assets.UP_IATS_NORMALIZED
            case assets.DOWN_IATS:
                name_ = assets.DOWN_IATS_NORMALIZED
            case _:
                raise ValueError(f"Unknown asset: {self.input_asset}")

        return name_

    def get_shapes(self, trace: dict[str, torch.Tensor]) -> Normalize:
        self._output_sizes = {self.name: trace[self.input_asset].shape[0]}
        return self

    def __call__(self, trace: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        trace_ = trace[self.input_asset] - trace[self.input_asset].mean()

        if self.division == "std":
            if (div := trace_.std()) == 0:
                logger.warning(f"Std zero when standardizing! {self.name}")
                if all(trace_ == 0):
                    return trace_
                raise ValueError("Standard deviation is zero. Cannot normalize.")
        elif self.division == "max":
            div = torch.max(torch.abs(trace_))
            if div == 0:
                logger.warning(f"Max zero when max normalizing! {self.name}")
                if all(trace_ == 0):
                    return trace_

        trace_ = trace_ / div
        return {self.name: trace_}


class IAT(_TR):
    NAME = "iat"

    def __init__(
        self, dir_key: str, time_asset: str = assets.TIMES, dir_asset: str = assets.DIRS
    ):
        if dir_key not in {"up", "down", "any"}:
            raise ValueError("Dir key must be either 'up' or 'down', or 'any'")
        self.dir_key = dir_key
        self.time_asset = time_asset
        self.dir_asset = dir_asset

    @property
    def name(self) -> str:
        if self.dir_key == "any":
            return assets.IATS
        if self.dir_key == "up":
            return assets.UP_IATS

        return assets.DOWN_IATS

    def get_shapes(self, trace: dict[str, torch.Tensor]) -> IAT:
        self._output_sizes = {self.name: trace[self.time_asset].shape[0]}
        return self

    def __call__(self, trace: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:

        if self.dir_key == "up":
            mask = trace[self.dir_asset] == 1
        elif self.dir_key == "down":
            mask = trace[self.dir_asset] == -1
        else:
            mask = torch.ones_like(trace[self.dir_asset], dtype=torch.bool)

        idxs = torch.where(mask)[0]
        iats = torch.zeros_like(trace[self.time_asset])
        if len(idxs) > 1:
            iats[idxs[1:]] = torch.diff(trace[self.time_asset][mask], dim=0)
        return {self.name: iats}


class _TimeWeight(_TR):

    def __init__(self, w_asset: str, time_asset: str):
        self.time_asset = time_asset
        self.w_asset = w_asset

    def get_shapes(self, trace: dict[str, torch.Tensor]) -> _TimeWeight:
        self._output_sizes = {self.name: trace[self.time_asset].shape[0]}
        return self

    def __call__(self, trace: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {self.name: trace[self.time_asset] * trace[self.w_asset]}


class TimeDirs(_TimeWeight):
    NAME = "time_dirs"

    def __init__(self, time_asset: str = assets.TIMES, dir_asset: str = assets.DIRS):
        super().__init__(dir_asset, time_asset)

    @property
    def name(self) -> str:
        return self.time_asset + "_dirs"


class IATDirs(_TimeWeight):
    NAME = "iat_dirs"

    def __init__(self, iat_asset: str = assets.IATS, dir_asset: str = assets.DIRS):
        super().__init__(dir_asset, iat_asset)

    @property
    def name(self) -> str:
        return self.time_asset + "_dirs"

    def __call__(self, trace: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        trace[self.time_asset] = trace[self.time_asset] + 1.0
        return super().__call__(trace)


class NormalizedIATDirs(_TimeWeight):
    NAME = "normalized_iat_dirs"

    def __init__(
        self, iat_asset: str = assets.IATS_NORMALIZED, dir_asset: str = assets.DIRS
    ):
        super().__init__(dir_asset, iat_asset)

    @property
    def name(self) -> str:
        return assets.NORMALIZED_IAT_DIRS


class Compose(_TR):
    NAME = "compose"

    def __init__(self, *transforms: _TR):
        self.transforms = transforms

    @property
    def name(self) -> str:
        return "pipe:" + "-->".join(tr.name for tr in self.transforms)

    def get_shapes(self, trace: dict[str, torch.Tensor]) -> Compose:
        for tr in self.transforms:
            tr.get_shapes(trace)
            if tr == self.transforms[-1]:
                trace = tr(trace)
            else:
                trace.update(tr(trace))

        self._output_sizes = tr.output_sizes

        return self

    def __call__(self, trace: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:

        for tr in self.transforms:
            if tr == self.transforms[-1]:
                trace = tr(trace)
            else:
                trace.update(tr(trace))

        return trace


class FeatureTrs:
    def __init__(
        self,
        feature_trs: list[_TR] | None = None,
        feature_names: list[str] | None = None,
        n_packets: int | None = None,
    ):
        if feature_trs is None and feature_names is None:
            raise ValueError(
                "Either 'feature_trs' or 'feature_names' must be provided."
            )
        if feature_trs is None:
            if n_packets is None:
                raise ValueError("n_packets must be provided if 'feature_names' is.")
            feature_trs = build_feature_trs(feature_names, n_packets)
        elif not all(isinstance(tr, _TR) for tr in feature_trs):
            raise ValueError("All elements in 'feature_trs' must be of type _TR.")

        self._feature_trs = feature_trs

    def get_shapes(self, trace: dict[str, torch.Tensor]) -> FeatureTrs:
        for tr in self._feature_trs:
            tr.get_shapes(trace)
        return self

    @property
    def output_sizes(self) -> dict[str, int]:
        return {tr.name: tr.output_sizes for tr in self._feature_trs}

    @property
    def features(self) -> list[str]:
        return [tr.name for tr in self._feature_trs]

    def report(self, to_log: bool = True) -> str:
        max_l = max(len(tr.name) for tr in self._feature_trs) + 3
        report = "Feature Transforms:\n"
        for tr in self._feature_trs:
            report += key_val_fmt(tr.name, tr.output_sizes, key_len=max_l) + "\n"

        if to_log:
            for line in report.split("\n"):
                logger.info(line)

        return report

    def __call__(self, trace: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:

        trace_: dict[str, torch.Tensor] = {}
        for tr in self._feature_trs:
            out = tr(trace)
            if len(out) != 1:
                raise ValueError(f"Transform {tr.name} returned more than one tensor.")

            trace_.update(out)

        return trace_


def build_feature_trs(feature_name: list[str], n_packets: int) -> list[_TR]:
    return [get_feature_tr(f, n_packets) for f in feature_name]


def get_feature_tr(feature_name: str, n_packets: int) -> _TR:
    match feature_name:
        case assets.DIRS:
            return Compose(PadOrCutTrace(n_packets), Select(assets.DIRS))
        case assets.SIZES:
            return Compose(PadOrCutTrace(n_packets), Select(assets.SIZES))
        case assets.TIMES:
            return Compose(PadOrCutTrace(n_packets), Select(assets.TIMES))
        case assets.UP_PACKETS:
            return Compose(
                PadOrCutTrace(n_packets),
                UDPackets("up", assets.DIRS),
            )
        case assets.DOWN_PACKETS:
            return Compose(
                PadOrCutTrace(n_packets),
                UDPackets("down", assets.DIRS),
            )
        case assets.IATS:
            return Compose(
                PadOrCutTrace(n_packets),
                IAT("any", time_asset=assets.TIMES, dir_asset=assets.DIRS),
            )
        case assets.UP_IATS:
            return Compose(
                PadOrCutTrace(n_packets),
                IAT("up", time_asset=assets.TIMES, dir_asset=assets.DIRS),
            )
        case assets.DOWN_IATS:
            return Compose(
                PadOrCutTrace(n_packets),
                IAT("down", time_asset=assets.TIMES, dir_asset=assets.DIRS),
            )
        case assets.TIMES_NORMALIZED:
            return Compose(
                PadOrCutTrace(n_packets),
                Normalize(normalized_asset=assets.TIMES, input_asset=assets.TIMES),
            )
        case assets.IATS_NORMALIZED:
            return Compose(
                PadOrCutTrace(n_packets),
                IAT("any", time_asset=assets.TIMES, dir_asset=assets.DIRS),
                Normalize(normalized_asset=assets.IATS, input_asset=assets.IATS),
            )
        case assets.TIME_DIRS:
            return Compose(
                PadOrCutTrace(n_packets),
                TimeDirs(time_asset=assets.TIMES, dir_asset=assets.DIRS),
            )
        case assets.IAT_DIRS:
            return Compose(
                PadOrCutTrace(n_packets),
                IAT("any", time_asset=assets.TIMES, dir_asset=assets.DIRS),
                IATDirs(iat_asset=assets.IATS, dir_asset=assets.DIRS),
            )
        case assets.NORMALIZED_IAT_DIRS:
            return Compose(
                PadOrCutTrace(n_packets),
                IAT("any", time_asset=assets.TIMES, dir_asset=assets.DIRS),
                Normalize(normalized_asset=assets.IATS, input_asset=assets.IATS),
                IATDirs(iat_asset=assets.IATS_NORMALIZED, dir_asset=assets.DIRS),
            )
        case _:
            raise ValueError(f"Unknown feature name: {feature_name}")
