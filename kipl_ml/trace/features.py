from __future__ import annotations

from dataclasses import dataclass

import kipl_ml.data.assets as assets
import mlflow
import torch
from kipl_ml.logging.logger import get_logger
from kipl_ml.logging.utils import key_val_fmt
from kipl_ml.trace.params import DOWNLOAD, UPLOAD
from kipl_ml.trace.transforms import _TR
from mlflow.models import set_model

logger = get_logger(__name__)


@dataclass
class Feats:
    DIRS: str = assets.DIRS
    SIZES: str = assets.SIZES
    TIMES: str = assets.TIMES
    TIMES_NORMALIZED: str = f"normalized_{TIMES}"
    TIMES_MAX_NORMALIZED: str = f"max_normalized_{TIMES}"
    CUM_TIMES: str = f"cum_{TIMES}"
    CUM_SIZES: str = f"cum_{SIZES}"
    CUM_SIZES_NORMALIZED: str = f"normalized_{CUM_SIZES}"
    LABEL: str = "label"
    IATS: str = "iats"
    IATS_NORMALIZED: str = f"normalized_{IATS}"
    IATS_MAX_NORMALIZED: str = f"max_normalized_{IATS}"
    UP_IATS: str = f"up_{IATS}"
    UP_IATS_NORMALIZED: str = f"up_{IATS_NORMALIZED}"
    DOWN_IATS: str = f"down_{IATS}"
    DOWN_IATS_NORMALIZED: str = f"down_{IATS_NORMALIZED}"
    UP_PACKETS: str = "up_packets"
    DOWN_PACKETS: str = "down_packets"
    TIME_DIRS: str = f"{TIMES}_dirs"
    IAT_DIRS: str = f"{IATS}_dirs"
    IAT_DIRS_NORMALIZED: str = f"{IATS_NORMALIZED}_dirs"
    CUM_SIZES_MAX_NORMALIZED = f"max_normalized_{CUM_SIZES}"
    BURST_EDGES: str = "burst_edges"
    FLOW_IATS: str = "flow_iats"
    FLOW_IATS_NORMALIZED: str = f"normalized_{FLOW_IATS}"
    LOG_INV_FLOW_IATS: str = f"log_inv_{FLOW_IATS}"
    LOG_INV_FLOW_IATS_NORMALIZED: str = f"log_inv_{FLOW_IATS_NORMALIZED}"
    LOG_INV_FLOW_IATS_NORMALIZED_DIRS: str = f"log_inv_{FLOW_IATS_NORMALIZED}_dirs"
    LOG_INV_FLOW_IAT_DIRS: str = f"log_inv_{IAT_DIRS}"
    RUNNING_RATE_SIZES: str = f"running_rate_{SIZES}"
    SIZE_DIRS: str = f"{SIZES}_dirs"
    CUM_SIZE_DIRS: str = f"cum_{SIZE_DIRS}"
    CUM_SIZE_DIRS_MAX_NORMALIZED: str = f"max_normalized_{CUM_SIZE_DIRS}"


def _pad_short_trace(
    trace: torch.Tensor, n_packets: int, asset_key: str, cut: bool = True
) -> torch.Tensor:
    if cut and (trace.shape[0] >= n_packets):
        return trace[:n_packets]

    if asset_key == Feats.TIMES:
        pad_val = trace[-1].item()
    elif asset_key in {Feats.DIRS, Feats.SIZES}:
        pad_val = 0.0
    elif asset_key == Feats.ORIG_PACKETS:
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

    def __init__(self, up_down: str, dir_asset: str = Feats.DIRS):
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
            mask = trace[self.dir_asset] == UPLOAD
        elif self.up_down == "down":
            mask = trace[self.dir_asset] == DOWNLOAD

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

        if self.division == "std":
            return {
                Feats.TIMES: Feats.TIMES_NORMALIZED,
                Feats.IATS: Feats.IATS_NORMALIZED,
                Feats.UP_IATS: Feats.UP_IATS_NORMALIZED,
                Feats.DOWN_IATS: Feats.DOWN_IATS_NORMALIZED,
                Feats.CUM_SIZES: Feats.CUM_SIZES_NORMALIZED,
                Feats.FLOW_IATS: Feats.FLOW_IATS_NORMALIZED,
            }[self.normalized_asset]

        return {
            Feats.TIMES: Feats.TIMES_MAX_NORMALIZED,
            Feats.IATS: Feats.IATS_MAX_NORMALIZED,
            Feats.CUM_SIZES: Feats.CUM_SIZES_MAX_NORMALIZED,
            Feats.CUM_SIZE_DIRS: Feats.CUM_SIZE_DIRS_MAX_NORMALIZED,
        }[self.normalized_asset]

    def get_shapes(self, trace: dict[str, torch.Tensor]) -> Normalize:
        self._output_sizes = {self.name: trace[self.input_asset].shape[0]}
        return self

    def __call__(self, trace: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        trace_ = trace[self.input_asset] - trace[self.input_asset].mean()

        if self.division == "std":
            if (div := trace_.std()) == 0:
                logger.warning(f"Std zero when standardizing! {self.name}")
                if all(trace_ == 0):
                    return {self.name: trace_}
                raise ValueError("Standard deviation is zero. Cannot normalize.")
        elif self.division == "max":
            if (div := torch.max(torch.abs(trace_))) == 0:
                logger.warning(f"Max zero when max normalizing! {self.name}")
                if all(trace_ == 0):
                    return {self.name: trace_}

        trace_ = trace_ / div
        return {self.name: trace_}


class IAT(_TR):
    NAME = "iat"

    def __init__(
        self, dir_key: str, time_asset: str = Feats.TIMES, dir_asset: str = Feats.DIRS
    ):
        if dir_key not in {"up", "down", "any"}:
            raise ValueError("Dir key must be either 'up' or 'down', or 'any'")
        self.dir_key = dir_key
        self.time_asset = time_asset
        self.dir_asset = dir_asset

    @property
    def name(self) -> str:
        if self.dir_key == "any":
            return Feats.IATS
        if self.dir_key == "up":
            return Feats.UP_IATS

        return Feats.DOWN_IATS

    def get_shapes(self, trace: dict[str, torch.Tensor]) -> IAT:
        self._output_sizes = {self.name: trace[self.time_asset].shape[0]}
        return self

    def __call__(self, trace: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:

        if self.dir_key == "up":
            mask = trace[self.dir_asset] == UPLOAD
        elif self.dir_key == "down":
            mask = trace[self.dir_asset] == DOWNLOAD
        else:
            mask = torch.ones_like(trace[self.dir_asset], dtype=torch.bool)

        idxs = torch.where(mask)[0]
        iats = torch.zeros_like(trace[self.time_asset])
        if len(idxs) > 1:
            iats[idxs[1:]] = torch.diff(trace[self.time_asset][mask], dim=0)
        return {self.name: iats}


class _DirWeight(_TR):

    def __init__(self, dir_asset: str, w_asset: str):
        self.dir_asset = dir_asset
        self.w_asset = w_asset

    @property
    def name(self) -> str:
        return self.w_asset + "_dirs"

    def get_shapes(self, trace: dict[str, torch.Tensor]) -> _DirWeight:
        self._output_sizes = {self.name: trace[self.dir_asset].shape[0]}
        return self

    def __call__(self, trace: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {self.name: trace[self.dir_asset] * trace[self.w_asset]}


class TimeDirs(_DirWeight):
    NAME = "time_dirs"

    def __init__(self, dir_asset: str = Feats.DIRS, time_asset: str = Feats.TIMES):
        super().__init__(dir_asset, time_asset)


class IATDirs(_DirWeight):
    NAME = "iat_dirs"

    def __init__(self, dir_asset: str = Feats.DIRS, iat_asset: str = Feats.TIMES):
        super().__init__(dir_asset, iat_asset)

    def __call__(self, trace: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        trace[self.w_asset] = trace[self.w_asset] + 1.0
        return super().__call__(trace)


class SizeDirs(_DirWeight):
    NAME = "size_dirs"

    def __init__(self, dir_asset: str = Feats.DIRS, size_asset: str = Feats.SIZES):
        super().__init__(dir_asset, size_asset)


class Cumulative(_TR):
    NAME = "cumulative"

    def __init__(self, asset: str):
        self.asset = asset

    @property
    def name(self) -> str:
        return f"cum_{self.asset}"

    def get_shapes(self, trace: dict[str, torch.Tensor]) -> Cumulative:
        self._output_sizes = {self.name: trace[self.asset].shape[0]}

    def __call__(self, trace: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {self.name: torch.cumsum(trace[self.asset], dim=0)}


class BurstEdges(_TR):
    NAME = "burst_edges"

    @property
    def name(self) -> str:
        return "burst_edges"

    def get_shapes(self, trace: dict[str, torch.Tensor]) -> BurstEdges:
        self._output_sizes = {self.name: trace[Feats.DIRS].shape[0]}
        return self

    def __call__(self, trace: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        edges = torch.diff(trace[Feats.DIRS], dim=0, prepend=torch.Tensor([0]))
        return {self.name: edges}


class FlowIATS(_TR):
    NAME = "flow_iats"

    @property
    def name(self) -> str:
        return Feats.FLOW_IATS

    def get_shapes(self, trace: dict[str, torch.Tensor]) -> FlowIATS:
        self._output_sizes = {self.name: trace[Feats.UP_IATS].shape[0]}
        return self

    def __call__(self, trace: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:

        dirs = trace[Feats.DIRS]
        flow_iats = torch.zeros_like(dirs)
        flow_iats[dirs == UPLOAD] = trace[Feats.UP_IATS][dirs == UPLOAD]
        flow_iats[dirs == DOWNLOAD] = trace[Feats.DOWN_IATS][dirs == DOWNLOAD]
        return {self.name: flow_iats}


class LogInv(_TR):
    NAME = "log_inv"

    def __init__(self, asset: str):
        self.asset = asset

    @property
    def name(self) -> str:
        return f"log_inv_{self.asset}"

    def get_shapes(self, trace: dict[str, torch.Tensor]) -> LogInv:
        self._output_sizes = {self.name: trace[self.asset].shape[0]}
        return self

    def __call__(self, trace: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        log_inv = torch.log(torch.nan_to_num(1 / trace[self.asset] + 1, posinf=1e4))

        return {self.name: log_inv}


class RunningRate(_TR):
    NAME = "running_rate"

    def __init__(self, asset: str, time_asset: str):
        self.asset = asset
        self.time_asset = time_asset

    @property
    def name(self) -> str:
        return f"running_rate_{self.asset}"

    def get_shapes(self, trace: dict[str, torch.Tensor]) -> RunningRate:
        self._output_sizes = {self.name: trace[self.asset].shape[0]}
        return self

    def __call__(self, trace: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        times = trace[self.time_asset]
        values = trace[self.asset]

        # First time is 0.0
        values[0] = 0.0
        times[0] = 1.0

        running_rate = torch.cumsum(values, dim=0) / times
        return {self.name: running_rate}


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
        case Feats.DIRS:
            return Compose(PadOrCutTrace(n_packets), Select(Feats.DIRS))
        case Feats.SIZES:
            return Compose(PadOrCutTrace(n_packets), Select(Feats.SIZES))
        case Feats.TIMES:
            return Compose(PadOrCutTrace(n_packets), Select(Feats.TIMES))
        case Feats.UP_PACKETS:
            return Compose(
                PadOrCutTrace(n_packets),
                UDPackets("up", Feats.DIRS),
            )
        case Feats.DOWN_PACKETS:
            return Compose(
                PadOrCutTrace(n_packets),
                UDPackets("down", Feats.DIRS),
            )
        case Feats.IATS:
            return Compose(
                PadOrCutTrace(n_packets),
                IAT("any", time_asset=Feats.TIMES, dir_asset=Feats.DIRS),
            )
        case Feats.UP_IATS:
            return Compose(
                PadOrCutTrace(n_packets),
                IAT("up", time_asset=Feats.TIMES, dir_asset=Feats.DIRS),
            )
        case Feats.DOWN_IATS:
            return Compose(
                PadOrCutTrace(n_packets),
                IAT("down", time_asset=Feats.TIMES, dir_asset=Feats.DIRS),
            )
        case Feats.TIMES_NORMALIZED:
            return Compose(
                PadOrCutTrace(n_packets),
                Normalize(normalized_asset=Feats.TIMES, input_asset=Feats.TIMES),
            )
        case Feats.TIMES_MAX_NORMALIZED:
            return Compose(
                PadOrCutTrace(n_packets),
                Normalize(
                    normalized_asset=Feats.TIMES,
                    input_asset=Feats.TIMES,
                    division="max",
                ),
            )
        case Feats.IATS_NORMALIZED:
            return Compose(
                PadOrCutTrace(n_packets),
                IAT("any", time_asset=Feats.TIMES, dir_asset=Feats.DIRS),
                Normalize(normalized_asset=Feats.IATS, input_asset=Feats.IATS),
            )
        case Feats.IATS_MAX_NORMALIZED:
            return Compose(
                PadOrCutTrace(n_packets),
                IAT("any", time_asset=Feats.TIMES, dir_asset=Feats.DIRS),
                Normalize(
                    normalized_asset=Feats.IATS,
                    input_asset=Feats.IATS,
                    division="max",
                ),
            )
        case Feats.TIME_DIRS:
            return Compose(
                PadOrCutTrace(n_packets),
                TimeDirs(time_asset=Feats.TIMES, dir_asset=Feats.DIRS),
            )
        case Feats.IAT_DIRS:
            return Compose(
                PadOrCutTrace(n_packets),
                IAT("any", time_asset=Feats.TIMES, dir_asset=Feats.DIRS),
                IATDirs(dir_asset=Feats.DIRS, iat_asset=Feats.IATS),
            )
        case Feats.IAT_DIRS_NORMALIZED:
            return Compose(
                PadOrCutTrace(n_packets),
                IAT("any", time_asset=Feats.TIMES, dir_asset=Feats.DIRS),
                Normalize(normalized_asset=Feats.IATS, input_asset=Feats.IATS),
                IATDirs(dir_asset=Feats.DIRS, iat_asset=Feats.IATS_NORMALIZED),
            )
        case Feats.CUM_SIZES:
            return Compose(
                PadOrCutTrace(n_packets),
                Cumulative(Feats.SIZES),
            )
        case Feats.CUM_SIZES_MAX_NORMALIZED:
            return Compose(
                PadOrCutTrace(n_packets),
                Cumulative(Feats.SIZES),
                Normalize(
                    normalized_asset=Feats.CUM_SIZES,
                    input_asset=Feats.CUM_SIZES,
                    division="max",
                ),
            )
        case Feats.BURST_EDGES:
            return Compose(
                PadOrCutTrace(n_packets),
                BurstEdges(),
            )
        case Feats.FLOW_IATS:
            return Compose(
                PadOrCutTrace(n_packets),
                IAT("up", time_asset=Feats.TIMES, dir_asset=Feats.DIRS),
                IAT("down", time_asset=Feats.TIMES, dir_asset=Feats.DIRS),
                FlowIATS(),
            )
        case Feats.FLOW_IATS_NORMALIZED:
            return Compose(
                PadOrCutTrace(n_packets),
                IAT("up", time_asset=Feats.TIMES, dir_asset=Feats.DIRS),
                IAT("down", time_asset=Feats.TIMES, dir_asset=Feats.DIRS),
                FlowIATS(),
                Normalize(
                    normalized_asset=Feats.FLOW_IATS,
                    input_asset=Feats.FLOW_IATS,
                ),
            )
        case Feats.LOG_INV_FLOW_IATS:
            return Compose(
                PadOrCutTrace(n_packets),
                IAT("up", time_asset=Feats.TIMES, dir_asset=Feats.DIRS),
                IAT("down", time_asset=Feats.TIMES, dir_asset=Feats.DIRS),
                FlowIATS(),
                LogInv(Feats.FLOW_IATS),
            )
        case Feats.LOG_INV_FLOW_IATS_NORMALIZED:
            return Compose(
                PadOrCutTrace(n_packets),
                IAT("up", time_asset=Feats.TIMES, dir_asset=Feats.DIRS),
                IAT("down", time_asset=Feats.TIMES, dir_asset=Feats.DIRS),
                FlowIATS(),
                Normalize(
                    normalized_asset=Feats.FLOW_IATS,
                    input_asset=Feats.FLOW_IATS,
                ),
                LogInv(Feats.FLOW_IATS_NORMALIZED),
            )
        case Feats.LOG_INV_FLOW_IAT_DIRS:
            return Compose(
                PadOrCutTrace(n_packets),
                IAT("up", time_asset=Feats.TIMES, dir_asset=Feats.DIRS),
                IAT("down", time_asset=Feats.TIMES, dir_asset=Feats.DIRS),
                FlowIATS(),
                LogInv(Feats.FLOW_IATS),
                IATDirs(dir_asset=Feats.DIRS, iat_asset=Feats.LOG_INV_FLOW_IATS),
            )
        case Feats.LOG_INV_FLOW_IATS_NORMALIZED_DIRS:
            return Compose(
                PadOrCutTrace(n_packets),
                IAT("up", time_asset=Feats.TIMES, dir_asset=Feats.DIRS),
                IAT("down", time_asset=Feats.TIMES, dir_asset=Feats.DIRS),
                FlowIATS(),
                Normalize(
                    normalized_asset=Feats.FLOW_IATS,
                    input_asset=Feats.FLOW_IATS,
                ),
                LogInv(Feats.FLOW_IATS_NORMALIZED),
                IATDirs(
                    dir_asset=Feats.DIRS,
                    iat_asset=Feats.LOG_INV_FLOW_IATS_NORMALIZED,
                ),
            )
        case Feats.RUNNING_RATE_SIZES:
            return Compose(
                PadOrCutTrace(n_packets),
                RunningRate(Feats.SIZES, Feats.TIMES),
            )
        case Feats.SIZE_DIRS:
            return Compose(
                PadOrCutTrace(n_packets),
                SizeDirs(dir_asset=Feats.DIRS, size_asset=Feats.SIZES),
            )
        case Feats.CUM_SIZE_DIRS:
            return Compose(
                PadOrCutTrace(n_packets),
                SizeDirs(dir_asset=Feats.DIRS, size_asset=Feats.CUM_SIZES),
                Cumulative(Feats.SIZE_DIRS),
            )
        case Feats.CUM_SIZE_DIRS_MAX_NORMALIZED:
            return Compose(
                PadOrCutTrace(n_packets),
                SizeDirs(dir_asset=Feats.DIRS, size_asset=Feats.SIZES),
                Cumulative(Feats.SIZE_DIRS),
                Normalize(
                    normalized_asset=Feats.CUM_SIZE_DIRS,
                    input_asset=Feats.SIZE_DIRS,
                    division="max",
                ),
            )
        case _:
            raise ValueError(f"Unknown feature name: {feature_name}")
