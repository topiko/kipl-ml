from __future__ import annotations

import pickle

import mlflow
import torch
from kipl_ml.data.assets import assets
from kipl_ml.logging.logger import get_logger
from kipl_ml.logging.utils import key_val_fmt
from kipl_ml.trace.transforms import _TR

logger = get_logger(__name__)


def _pad_short_trace(trace: torch.Tensor, n_packets: int) -> torch.Tensor:
    if trace.shape[0] >= n_packets:
        return trace[:n_packets]

    return torch.cat(
        [
            trace,
            torch.zeros(
                n_packets - trace.shape[0],
                *trace.shape[1:],
                dtype=trace.dtype,
                device=trace.device,
            ),
        ]
    )


class CutTrace(_TR):
    NAME = "sel_packets"

    def __init__(self, asset: str, n_packets: int):
        self.n_packets = n_packets
        super().__init__(asset)

    @property
    def name(self) -> str:
        return f"{self.n_packets}-{self.asset}"

    def get_shapes(self, trace: dict[str, torch.Tensor]) -> CutTrace:
        self._input_size = trace[self.asset].shape[0]
        self._output_size = self.n_packets

        return self

    def __call__(self, trace: dict[str, torch.Tensor]) -> torch.Tensor:
        trace_ = _pad_short_trace(trace[self.asset], self.n_packets)
        print(self.asset)
        print(trace_.shape)
        print()
        return trace_


class Select(_TR):
    NAME = "identity"

    def get_shapes(self, trace: dict[str, torch.Tensor]) -> Select:
        self._input_size = trace[self.asset].shape[0]
        self._output_size = trace[self.asset].shape[0]
        return self

    def __call__(self, trace: dict[str, torch.Tensor]) -> torch.Tensor:
        return trace[self.asset]


class UDPackets(_TR):
    NAME = "up/down_packets"

    def __init__(self, up_down: str, dir_asset: str = assets.DIR):
        if up_down not in {"up", "down"}:
            raise ValueError("up_down must be either 'up' or 'down'")
        self.up_down = up_down
        super().__init__(dir_asset)

    @property
    def name(self) -> str:
        return self.up_down + "_packets"

    def get_shapes(self, trace: dict[str, torch.Tensor]) -> UpPackets:
        self._input_size = trace[self.asset].shape[0]
        self._output_size = trace[self.asset].shape[0]
        return self

    def __call__(self, trace: dict[str, torch.Tensor]) -> torch.Tensor:
        if self.up_down == "up":
            mask = trace[self.asset] == 1
        elif self.up_down == "down":
            mask = trace[self.asset] == -1
        return mask.float()


class Normalize(_TR):
    NAME = "normalized"

    def get_shapes(self, trace: dict[str, torch.Tensor]) -> Normlaize:
        self._input_size = trace[self.asset].shape[0]
        self._output_size = trace[self.asset].shape[0]
        return self

    def __call__(self, trace: dict[str, torch.Tensor]) -> torch.Tensor:
        if (std := trace[self.asset].std()) == 0:
            logger.warning(f"Std zeron when standardizing! {self.name}")
            if all(trace[self.asset] == 0):
                return trace[self.asset]
            raise ValueError("Standard deviation is zero. Cannot normalize.")
        return (trace[self.asset] - trace[self.asset].mean()) / std


class IAT(_TR):
    NAME = "iat"

    def __init__(self, dir_key: str, time_asset: str = assets.TIME):
        if dir_key not in {"up", "down", "any"}:
            raise ValueError("Dir key must be either 'up' or 'down', or 'any'")
        self.dir_key = dir_key
        super().__init__(asset=time_asset)

    @property
    def name(self) -> str:
        if self.dir_key == "any":
            return self.NAME

        return f"{self.dir_key}_{self.NAME}"

    def get_shapes(self, trace: dict[str, torch.Tensor]) -> IAT:
        self._input_size = trace[self.asset].shape[0]
        self._output_size = trace[self.asset].shape[0]
        return self

    def __call__(self, trace: dict[str, torch.Tensor]) -> torch.Tensor:

        if self.dir_key == "up":
            mask = trace[assets.DIR] == 1
        elif self.dir_key == "down":
            mask = trace[assets.DIR] == -1
        else:
            mask = torch.ones_like(trace[assets.DIR], dtype=torch.bool)

        idxs = torch.where(mask)[0]
        iats = torch.zeros_like(trace[assets.TIME])
        if len(idxs) > 1:
            iats[idxs[1:]] = torch.diff(trace[assets.TIME][mask], dim=0)
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

    def get_shapes(self, trace: dict[str, torch.Tensor]) -> Compose:
        outs = None
        for i, tr in enumerate(self.transforms):
            tr.get_shapes(trace)
            outs = tr.output_size
            trace[tr.name] = trace[tr.asset][:outs]

            if i == 0:
                self._input_size = tr.input_size

        self._output_size = outs

        return self

    def __call__(self, trace: dict[str, torch.Tensor]) -> torch.Tensor:
        for i, tr in enumerate(self.transforms):
            tensor = tr(trace)
            if i < len(self.transforms) - 1:
                trace[self.transforms[i].name] = tensor

        return tensor


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

    def __call__(self, trace: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {tr.name: tr(trace) for tr in self._feature_trs}


def build_feature_trs(feature_name: list[str], n_packets: int) -> list[_TR]:
    return [get_feature_tr(f, n_packets) for f in feature_name]


def get_feature_tr(feature_name: str, n_packets: int) -> _TR:
    match feature_name:
        case "dirs":
            return CutTrace(assets.DIR, n_packets)
        case "sizes":
            return CutTrace(assets.SIZE, n_packets)
        case "times":
            return CutTrace(assets.TIME, n_packets)
        case "up_packets":
            return Compose(
                CutTrace(assets.DIR, n_packets),
                UDPackets("up", f"{n_packets}-{assets.DIR}"),
            )
        case "down_packets":
            return Compose(
                CutTrace(assets.DIR, n_packets),
                UDPackets("down", f"{n_packets}-{assets.DIR}"),
            )
        case "iats":
            return Compose(
                CutTrace(assets.TIME, n_packets),
                IAT("any", f"{n_packets}-{assets.TIME}"),
            )
        case "up_iats":
            return Compose(
                CutTrace(assets.TIME, n_packets),
                IAT("up", f"{n_packets}-{assets.TIME}"),
            )
        case "down_iats":
            return Compose(
                CutTrace(assets.TIME, n_packets),
                IAT("down", f"{n_packets}-{assets.TIME}"),
            )
        case "times_normalized":
            return Compose(
                CutTrace(assets.TIME, n_packets),
                Normalize(f"{n_packets}-{assets.TIME}"),
            )
        case "iats_normalized":
            return Compose(
                CutTrace(assets.TIME, n_packets),
                IAT("any", f"{n_packets}-{assets.TIME}"),
                Normalize("iat"),
            )
        case _:
            raise ValueError(f"Unknown feature name: {feature_name}")


def log_feature_trs_to_mlflow(feature_trs: FeatureTrs):
    if (run := mlflow.active_run()) is None:
        raise ValueError("No active MLFlow run.")

    run_id = run.info.run_id

    fname = f"feature_trs_{run_id}.pkl"
    with open(fname, "wb") as f:
        pickle.dump(feature_trs, f)

    logger.info(f"Logging feature transforms to MLFlow.")
    mlflow.log_artifact(fname, artifact_path="feature_trs")
