from __future__ import annotations

from copy import deepcopy
from typing import Any

import torch
from torch import nn

from kipl_ml.data import assets
from kipl_ml.logging.logger import get_logger
from kipl_ml.logging.utils import key_val_fmt, log_multiline
from kipl_ml.trace.enums import Feats
from kipl_ml.trace.params import DOWNLOAD, UPLOAD
from kipl_ml.trace.transforms import _TR

logger = get_logger(__name__)


FEAT_NAME_MAP = {
    "dirs": Feats.DIRS,
    "flow_iats": Feats.FLOW_IATS,
    "time_dirs": Feats.TIME_DIRS,
    "size_dirs": Feats.SIZE_DIRS,
    "cumul": Feats.CUM_SIZE_DIRS,
    "times_norm": Feats.TIMES_MAX_NORMALIZED,
    "cumul_norm": Feats.CUM_SIZE_DIRS_MAX_NORMALIZED,
    "iat_dirs": Feats.IAT_DIRS,
    "inv_iat_log_dirs": Feats.LOG_INV_FLOW_IAT_DIRS,
    "inv_iat_logs": Feats.LOG_INV_FLOW_IATS,
    "running_rates": Feats.RUNNING_RATE_SIZES,
    "max_normalized_running_rates": Feats.RUNNING_RATE_SIZES_MAX_NORMALIZED,
}


def _pad_short_trace(
    trace: torch.Tensor,
    n_packets: int,
    asset_key: str,
    cut: bool = True,
) -> torch.Tensor:
    if cut and len(trace) > n_packets:
        return trace[:n_packets]

    if asset_key == Feats.TIMES:
        pad_val = trace[-1].item()
    elif asset_key in {Feats.DIRS, Feats.SIZES}:
        pad_val = 0.0
    elif asset_key == Feats.PADDING:
        pad_val = False
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

    def __init__(self, n_packets: int | None):
        if n_packets is None:
            logger.info("Disabled padding/cutting traces - n_packets is None")
        self.n_packets = n_packets

    @property
    def name(self) -> str:
        if self.n_packets is None:
            return "cut/pad DISABLED"
        return f"slice|:{self.n_packets}"

    def get_shapes(self, trace: dict[Feats, torch.Tensor]) -> PadOrCutTrace:
        self._output_sizes = {key: self.n_packets for key in trace.keys()}

        return self

    def __call__(self, trace: dict[Feats, torch.Tensor]) -> dict[Feats, torch.Tensor]:
        if self.n_packets is None:
            return trace

        trace_ = {
            key: _pad_short_trace(val, self.n_packets, asset_key=key)
            for key, val in trace.items()
        }

        return trace_


class Select(_TR):
    NAME = "select"

    def __init__(self, asset: Feats):
        self.asset = asset

    def get_shapes(self, trace: dict[Feats, torch.Tensor]) -> Select:
        self._output_sizes = {self.asset: trace[self.asset].shape[0]}
        return self

    def __call__(self, trace: dict[Feats, torch.Tensor]) -> dict[Feats, torch.Tensor]:
        return {self.asset: trace[self.asset]}


class DirProbs(_TR):
    NAME = Feats.DIR_PROBS

    @property
    def name(self) -> Feats:
        return Feats.DIR_PROBS

    def get_shapes(self, trace: dict[Feats, torch.Tensor]) -> DirProbs:
        self._output_sizes = {self.name: trace[Feats.DIRS].shape[0]}
        return self

    def __call__(self, trace: dict[Feats, torch.Tensor]) -> dict[Feats, torch.Tensor]:
        dirs = trace[Feats.DIRS]

        dirps = nn.functional.one_hot(dirs.long() + 1, num_classes=3).float()

        return {self.name: dirps}


class UDPackets(_TR):
    NAME = "up/down_packets"

    def __init__(self, up_down: str, dir_asset: Feats = Feats.DIRS):
        if up_down not in {"up", "down"}:
            raise ValueError("up_down must be either 'up' or 'down'")
        self.up_down = up_down
        self.dir_asset = dir_asset

    @property
    def name(self) -> Feats:
        match self.up_down:
            case "down":
                return Feats.DOWN_PACKETS
            case "up":
                return Feats.UP_PACKETS
            case _:
                raise ValueError("up_down must be either 'up' or 'down'")

    def get_shapes(self, trace: dict[Feats, torch.Tensor]) -> UDPackets:
        self._output_sizes = {self.name: trace[self.dir_asset].shape[0]}
        return self

    def __call__(self, trace: dict[Feats, torch.Tensor]) -> dict[Feats, torch.Tensor]:
        if self.up_down == "up":
            mask = trace[self.dir_asset] == UPLOAD
        elif self.up_down == "down":
            mask = trace[self.dir_asset] == DOWNLOAD

        return {self.name: mask.float()}


class Normalize(_TR):
    NAME = "normalized"

    def __init__(
        self, normalized_asset: Feats, input_asset: Feats, division: str = "std"
    ):
        if division not in {"std", "max"}:
            raise ValueError("Division must be either 'std' or 'max'")

        self.normalized_asset = normalized_asset
        self.input_asset = input_asset
        self.division = division

    @property
    def name(self) -> Feats:
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
            Feats.RUNNING_RATE_SIZES: Feats.RUNNING_RATE_SIZES_MAX_NORMALIZED,
            Feats.TAM_UP_COUNTS: Feats.TAM_UP_COUNTS_MAX_NORMALIZED,
            Feats.TAM_DOWN_COUNTS: Feats.TAM_DOWN_COUNTS_MAX_NORMALIZED,
        }[self.normalized_asset]

    def get_shapes(self, trace: dict[Feats, torch.Tensor]) -> Normalize:
        self._output_sizes = {self.name: trace[self.input_asset].shape[0]}
        return self

    def __call__(self, trace: dict[Feats, torch.Tensor]) -> dict[Feats, torch.Tensor]:
        trace_ = trace[self.normalized_asset] - trace[self.normalized_asset].mean()

        if self.division == "std":
            if (div := trace_.std()) == 0:
                logger.warning("Std zero when standardizing! %s" % self.name)
                if all(trace_ == 0):
                    return {self.name: trace_}
                raise ValueError("Standard deviation is zero. Cannot normalize.")
        elif self.division == "max":
            if (div := torch.max(torch.abs(trace_))) == 0:
                logger.warning("Max zero when max normalizing! %s" % self.name)
                if all(trace_ == 0):
                    return {self.name: trace_}

                raise ValueError("Absmax == 0, however, nonzero value encountered!?")

        trace_ = trace_ / div
        return {self.name: trace_}


class IAT(_TR):
    NAME = "iat"

    def __init__(
        self,
        dir_key: str,
        time_asset: Feats = Feats.TIMES,
        dir_asset: Feats = Feats.DIRS,
    ):
        if dir_key not in {"up", "down", "any"}:
            raise ValueError("Dir key must be either 'up' or 'down', or 'any'")
        self.dir_key = dir_key
        self.time_asset = time_asset
        self.dir_asset = dir_asset

    @property
    def name(self) -> Feats:
        match self.dir_key:
            case "any":
                return Feats.IATS
            case "up":
                return Feats.UP_IATS
            case "down":
                return Feats.DOWN_IATS
            case _:
                raise ValueError("Dir key must be either 'up' or 'down', or 'any'")

    def get_shapes(self, trace: dict[Feats, torch.Tensor]) -> IAT:
        self._output_sizes = {self.name: trace[self.time_asset].shape[0]}
        return self

    def __call__(self, trace: dict[Feats, torch.Tensor]) -> dict[Feats, torch.Tensor]:
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

    def get_shapes(self, trace: dict[Feats, torch.Tensor]) -> _DirWeight:
        self._output_sizes = {self.name: trace[self.dir_asset].shape[0]}
        return self

    def __call__(self, trace: dict[Feats, torch.Tensor]) -> dict[Feats, torch.Tensor]:
        return {self.name: trace[self.dir_asset] * trace[self.w_asset]}


class TimeDirs(_DirWeight):
    NAME = "time_dirs"

    def __init__(self, dir_asset: str = Feats.DIRS, time_asset: str = Feats.TIMES):
        super().__init__(dir_asset=dir_asset, w_asset=time_asset)


class IATDirs(_DirWeight):
    NAME = "iat_dirs"

    def __init__(
        self,
        dir_asset: str = Feats.DIRS,
        iat_asset: str = Feats.TIMES,
        add_unit_to_weight: bool = True,
    ):
        super().__init__(dir_asset=dir_asset, w_asset=iat_asset)
        self.add = 0.0
        if add_unit_to_weight:
            self.add = 1.0

    def __call__(self, trace: dict[Feats, torch.Tensor]) -> dict[Feats, torch.Tensor]:
        trace[self.w_asset] = trace[self.w_asset] + self.add
        return super().__call__(trace)


class SizeDirs(_DirWeight):
    NAME = "size_dirs"

    def __init__(self, dir_asset: str = Feats.DIRS, size_asset: str = Feats.SIZES):
        super().__init__(dir_asset=dir_asset, w_asset=size_asset)


class Cumulative(_TR):
    NAME = "cumulative"

    def __init__(self, asset: str):
        self.asset = asset

    @property
    def name(self) -> str:
        return f"cum_{self.asset}"

    def get_shapes(self, trace: dict[Feats, torch.Tensor]) -> Cumulative:
        self._output_sizes = {self.name: trace[self.asset].shape[0]}

    def __call__(self, trace: dict[Feats, torch.Tensor]) -> dict[Feats, torch.Tensor]:
        return {self.name: torch.cumsum(trace[self.asset], dim=0)}


class BurstEdges(_TR):
    NAME = Feats.BURST_EDGES

    @property
    def name(self) -> Feats:
        return self.NAME

    def get_shapes(self, trace: dict[Feats, torch.Tensor]) -> BurstEdges:
        self._output_sizes = {self.name: trace[Feats.DIRS].shape[0]}
        return self

    def __call__(self, trace: dict[Feats, torch.Tensor]) -> dict[Feats, torch.Tensor]:
        edges = torch.diff(trace[Feats.DIRS], dim=0, prepend=torch.Tensor([0]))
        return {self.name: edges}


class BurstLens(_TR):
    NAME = Feats.BURST_LENS

    @property
    def name(self) -> Feats:
        return self.NAME

    def get_shapes(self, trace: dict[Feats, torch.Tensor]) -> BurstLens:
        self._output_sizes = {self.name: None}  # Variable length
        return self

    def __call__(self, trace: dict[Feats, torch.Tensor]) -> dict[Feats, torch.Tensor]:
        edges = trace[Feats.BURST_EDGES]

        # Example:
        # dirs:    -1, 1, 1, -1, -1, -1, 1, 1, 1, 1, -1
        # edges:   -1, 2, 0, -2,  0,  0, 2, 0, 0, 0, -2
        # idxs:        1,     3,         6,          10,
        # lens:     1,    2,          3,          4, ...

        down2up = torch.argwhere(edges == 2).squeeze()
        up2down = torch.argwhere(edges == -2).squeeze()

        idxs = torch.cat((down2up, up2down, torch.Tensor([len(edges)]))).sort()[0]

        burst_lens = torch.diff(idxs, prepend=torch.Tensor([0]))

        return {self.name: burst_lens}


class BurstDurs(_TR):
    NAME = Feats.BURST_DURS

    @property
    def name(self) -> Feats:
        return self.NAME

    def get_shapes(self, trace: dict[Feats, torch.Tensor]) -> BurstLens:
        self._output_sizes = {self.name: None}  # Variable length
        return self

    def __call__(self, trace: dict[Feats, torch.Tensor]) -> dict[Feats, torch.Tensor]:
        edges = trace[Feats.BURST_EDGES]

        # Example:
        # dirs:    -1, 1, 1, -1, -1, -1,  1, 1, 1, 1, -1
        # edges:   -1, 2, 0, -2,  0,  0,  2, 0, 0, 0, -2
        # idxs:        1,     3,          6,          10,
        # times:    0, 1, 2,  4,  8, 12, 14, 0, 0, 0, -2
        # bt:       0, 1,     3,         10, ...

        down2up = torch.argwhere(edges == 2).squeeze()
        up2down = torch.argwhere(edges == -2).squeeze()

        idxs = (
            torch.cat((down2up, up2down, torch.Tensor([len(edges) - 1])))
            .sort()[0]
            .long()
        )

        burst_durs = torch.diff(trace[Feats.TIMES][idxs], prepend=torch.Tensor([0]))

        return {self.name: burst_durs}


class BurstRelativeDurs(_TR):
    NAME = Feats.BURST_RELDURS

    @property
    def name(self) -> Feats:
        return self.NAME

    def get_shapes(self, trace: dict[Feats, torch.Tensor]) -> BurstLens:
        self._output_sizes = {self.name: None}  # Variable length
        return self

    def __call__(self, trace: dict[Feats, torch.Tensor]) -> dict[Feats, torch.Tensor]:
        # Example:
        # dirs:    -1, 1, 1, -1, -1, -1,  1, 1, 1, 1, -1
        # edges:   -1, 2, 0, -2,  0,  0,  2, 0, 0, 0, -2
        # idxs:        1,     3,          6,          10,
        # times:    0, 1, 2,  4,  8, 12, 14, 0, 0, 0, -2
        # bt:       0, 1,     3,         10, ...
        bt = trace[Feats.BURST_DURS]

        bt_down = bt[::2]
        bt_up = bt[1::2]

        cummax_down = bt_down.cummax(0)[0]
        cummax_up = bt_up.cummax(0)[0]

        bt_down = torch.where(cummax_down != 0, bt_down / cummax_down, 1)
        bt_up = torch.where(cummax_up != 0, bt_up / cummax_up, 1)

        burst_rel_durs = torch.zeros_like(bt)
        burst_rel_durs[::2] = bt_down
        burst_rel_durs[1::2] = bt_up

        return {self.name: burst_rel_durs}


class BurstDirs(_TR):
    NAME = Feats.BURST_DIRS

    @property
    def name(self) -> Feats:
        return self.NAME

    def get_shapes(self, trace: dict[Feats, torch.Tensor]) -> BurstDirs:
        self._output_sizes = {self.name: None}  # Variable length
        return self

    def __call__(self, trace: dict[Feats, torch.Tensor]) -> dict[Feats, torch.Tensor]:
        edges = trace[Feats.BURST_EDGES]

        # Example:
        # dirs:    -1, 1, 1, -1, -1, -1, 1, 1, 1, 1, -1
        # edges:   -1, 2, 0, -2,  0,  0, 2, 0, 0, 0, -2
        # lens:     1,    2,          3,          4, ...

        down2up = torch.argwhere(edges == 2).squeeze()
        up2down = torch.argwhere(edges == -2).squeeze()

        idxs = torch.cat((down2up, up2down, torch.Tensor([len(edges)]))).sort()[0]

        burst_dirs = torch.ones_like(idxs)
        if trace[Feats.DIRS][0] == DOWNLOAD:
            burst_dirs[0::2] = -1
        elif trace[Feats.DIRS][0] == UPLOAD:
            burst_dirs[1::2] = -1
        else:
            raise ValueError("First dir cannot be 0")

        return {self.name: burst_dirs}


class FlowIATS(_TR):
    NAME = Feats.FLOW_IATS

    @property
    def name(self) -> Feats:
        return Feats.FLOW_IATS

    def get_shapes(self, trace: dict[Feats, torch.Tensor]) -> FlowIATS:
        self._output_sizes = {self.name: trace[Feats.UP_IATS].shape[0]}
        return self

    def __call__(self, trace: dict[Feats, torch.Tensor]) -> dict[Feats, torch.Tensor]:
        dirs = trace[Feats.DIRS]
        flow_iats = torch.zeros_like(dirs)
        flow_iats[dirs == UPLOAD] = trace[Feats.UP_IATS][dirs == UPLOAD]
        flow_iats[dirs == DOWNLOAD] = trace[Feats.DOWN_IATS][dirs == DOWNLOAD]
        return {self.name: flow_iats}


class LogInv(_TR):
    NAME = "log_inv"

    def __init__(self, asset: Feats):
        self.asset = asset

    @property
    def name(self) -> Feats:
        return f"log_inv_{self.asset}"

    def get_shapes(self, trace: dict[Feats, torch.Tensor]) -> LogInv:
        self._output_sizes = {self.name: trace[self.asset].shape[0]}
        return self

    def __call__(self, trace: dict[Feats, torch.Tensor]) -> dict[Feats, torch.Tensor]:
        log_inv = torch.log(torch.nan_to_num(1 / trace[self.asset] + 1, posinf=1e4))

        return {self.name: log_inv}


class Log1p(_TR):
    NAME = "log1p"

    def __init__(self, asset: Feats):
        self.asset = asset

    def get_shapes(self, trace: dict[Feats, torch.Tensor]) -> LogInv:
        self._output_sizes = {self.name: trace[self.asset].shape[0]}
        return self

    def __call__(self, trace: dict[Feats, torch.Tensor]) -> dict[Feats, torch.Tensor]:
        log_inv = torch.log1p(torch.nan_to_num(1 / trace[self.asset] + 1, posinf=1e4))

        return {self.name: log_inv}


class RunningRate(_TR):
    NAME = "running_rate"

    def __init__(self, asset: Feats, time_asset: Feats):
        self.asset = asset
        self.time_asset = time_asset

    @property
    def name(self) -> Feats:
        return f"running_rate_{self.asset}"

    def get_shapes(self, trace: dict[Feats, torch.Tensor]) -> RunningRate:
        self._output_sizes = {self.name: None}
        return self

    def __call__(self, trace: dict[Feats, torch.Tensor]) -> dict[Feats, torch.Tensor]:
        times = trace[self.time_asset]
        values = trace[self.asset].clone()

        values[times == 0] = 0.0

        running_rate = torch.where(
            times != 0, torch.cumsum(values, dim=0) / times, torch.zeros_like(values)
        )

        return {self.name: running_rate}


class _TAM(_TR):
    NAME = "tam"
    DIR: str
    TIMES: bool
    PADDING: bool

    def __init__(
        self,
        max_matrix_len: int = 1800,
        max_load_time_s: float = 80.0,
        window_width_s: float | None = None,
        prune_empty_bins: bool = False,
        max_len: int | None = None,
    ):
        self.max_load_time_s = max_load_time_s
        self.max_len = max_len
        self.prune_empty = prune_empty_bins
        if window_width_s is not None:
            logger.warning(
                f"window_width_s overrides max_matrix_len {max_matrix_len}"
                + f"-> {max_load_time_s / window_width_s}"
            )
            self.max_matrix_len = int(self.max_load_time_s // window_width_s + 1)
            self.window_width_s = window_width_s
        else:
            self.max_matrix_len = max_matrix_len

            # Keep TAM on an integer grid; derive bin times from bin indices.
            # This avoids float32 spacing jitter and keeps binning stable.
            self.window_width_s = float(self.max_load_time_s) / float(self.max_matrix_len)

        # Integer bin indices [0..max_matrix_len].
        self.bin_idx = torch.arange(0, int(self.max_matrix_len) + 1, dtype=torch.long)
        # Bin start times in seconds (float).
        self.bins = self.bin_idx.to(dtype=torch.float32) * float(self.window_width_s)

        self.padding_warned = False
        # To ensure the capture of "outside bins values"
        # self.bins[0] = -1
        # self.bins[-1] = float("inf")

    @property
    def name(self) -> Feats:
        if self.TIMES:
            return Feats.TAM_TIMES
        if self.DIR == "upload" and not self.TIMES:
            if self.PADDING:
                return Feats.TAM_UP_PAD
            return Feats.TAM_UP_COUNTS
        if self.DIR == "download" and not self.TIMES:
            if self.PADDING:
                return Feats.TAM_DOWN_PAD
            return Feats.TAM_DOWN_COUNTS

        raise KeyError(f"Invalid dir {self.DIR}")

    def get_shapes(self, trace: dict[Feats, torch.Tensor]) -> _TAM:
        if self.prune_empty:
            len_ = None  # Variable length, depends on trace
        else:
            len_ = min(self.max_len or float("inf"), self.max_matrix_len)

        self._output_sizes = {self.name: len_}
        return self

    def __call__(self, trace: dict[Feats, torch.Tensor]) -> dict[Feats, torch.Tensor]:
        times = trace[assets.TIMES]
        dirs = trace[assets.DIRS]

        match self.DIR:
            case "upload":
                mask = dirs == UPLOAD
            case "download":
                mask = dirs == DOWNLOAD
            case "up/download":
                mask = (dirs == DOWNLOAD) | (dirs == UPLOAD)
            case _:
                raise KeyError(f"Invalid dir {self.DIR}")

        if self.TIMES:
            mask = dirs != 0

        if self.PADDING:
            try:
                mask = mask & (trace[Feats.PADDING] == 1)
            except KeyError:
                if not self.padding_warned:
                    logger.warning(
                        "PADDING key not found in trace, but PADDING is True. "
                        + "Proceeding without padding mask."
                    )
                    self.padding_warned = True
                mask = torch.zeros_like(mask, dtype=torch.bool)

        # NOTE: we expect the time to be in "s"!
        # counts2, edges = torch.histogram(
        #     times[mask], bins=len(self.bins) - 1, range=(0, self.bins[-1].item())
        # )
        counts = torch.histc(
            times[mask], bins=len(self.bins) - 1, min=0.0, max=self.bins[-1]
        )

        mask = torch.ones_like(counts, dtype=torch.bool)
        if self.prune_empty:
            mask = counts > 0

        if self.TIMES:
            # We can also return the time of each bin
            bin_times = self.bins[:-1].to(times.device)[mask]

            return {self.name: bin_times}

        counts = counts[mask]
        return {self.name: counts}


class TAM_UP(_TAM):
    DIR = "upload"
    TIMES = False
    PADDING = False


class TAM_DOWN(_TAM):
    DIR = "download"
    TIMES = False
    PADDING = False


class TAM_UP_PAD(_TAM):
    DIR = "upload"
    TIMES = False
    PADDING = True


class TAM_DOWN_PAD(_TAM):
    DIR = "download"
    TIMES = False
    PADDING = True


class TAM_TIMES(_TAM):
    DIR = "up/download"
    TIMES = True
    PADDING = False


class TAM_BINS(_TAM):
    DIR = "up/download"
    TIMES = True
    PADDING = False

    @property
    def name(self) -> Feats:
        return Feats.TAM_BINS

    def __call__(self, trace: dict[Feats, torch.Tensor]) -> dict[Feats, torch.Tensor]:
        # Same masking and pruning behavior as TAM_TIMES, but return integer bins.
        times = trace[assets.TIMES]
        dirs = trace[assets.DIRS]

        mask = dirs != 0

        counts = torch.histc(
            times[mask], bins=len(self.bins) - 1, min=0.0, max=self.bins[-1]
        )

        keep = torch.ones_like(counts, dtype=torch.bool)
        if self.prune_empty:
            keep = counts > 0

        bin_idx = self.bin_idx[:-1].to(device=times.device)
        return {self.name: bin_idx[keep]}


class Compose(_TR):
    NAME = "compose"

    def __init__(self, *transforms: _TR):
        self.transforms = transforms

    @property
    def name(self) -> str:
        return "pipe:" + "-->".join(tr.name for tr in self.transforms)

    @property
    def output(self) -> str:
        return self.transforms[-1].name

    def get_shapes(self, trace: dict[Feats, torch.Tensor]) -> Compose:
        # Avoid inplace changes
        _trace = deepcopy(trace)
        for tr in self.transforms:
            tr.get_shapes(_trace)

            if tr == self.transforms[-1]:
                _trace = tr(_trace)
            else:
                _trace.update(tr(_trace))

        self._output_sizes = tr.output_sizes

        return self

    def __call__(self, trace: dict[Feats, torch.Tensor]) -> dict[Feats, torch.Tensor]:
        # Here inplace changes are fine?
        trace = {k: v.clone() for k, v in trace.items()}
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
        feature_names: list[Feats] | None = None,
        n_packets: int | None = None,
    ):
        if feature_trs is None and feature_names is None:
            raise ValueError(
                "Either 'feature_trs' or 'feature_names' must be provided."
            )
        if feature_trs is None:
            if n_packets is None:
                logger.warning("No 'n_packets' provided, padding/cutting disabled.")
            feature_trs = build_feature_trs(feature_names, n_packets)
        elif not all(isinstance(tr, _TR) for tr in feature_trs):
            raise ValueError("All elements in 'feature_trs' must be of type _TR.")

        self._feature_trs = feature_trs

    def get_shapes(self, trace: dict[Feats, torch.Tensor]) -> FeatureTrs:
        for tr in self._feature_trs:
            tr.get_shapes(trace)
        return self

    @property
    def output_sizes(self) -> dict[str, dict[str, int]]:
        return {tr.name: tr.output_sizes for tr in self._feature_trs}

    @property
    def features(self) -> list[str]:
        return [tr.name for tr in self._feature_trs]

    def report(self, to_log: bool = True) -> str:
        max_l = max(len(tr.name) for tr in self._feature_trs) + 3
        report = "Feature Transforms:\n"
        for tr in self._feature_trs:
            report += key_val_fmt(tr.name, tr.output_sizes, key_len=max_l)

        if to_log:
            log_multiline(report)

        return report

    def __call__(self, trace: dict[Feats, torch.Tensor]) -> dict[Feats, torch.Tensor]:
        trace_: dict[Feats, torch.Tensor] = {}
        for tr in self._feature_trs:
            out = tr(trace)

            if len(out) != 1:
                raise ValueError(f"Transform {tr.name} returned more than one tensor.")

            trace_.update(out)

        return trace_

    def transform_batch(
        self, trace_batch: dict[Feats, torch.Tensor], pad_val: float | None = None
    ) -> dict[Feats, torch.Tensor]:
        # Very inefficient implementation, but ok for now
        bs = trace_batch[next(iter(trace_batch))].shape[0]

        if trace_batch[next(iter(trace_batch))].ndim < 2:
            raise ValueError("Batch trasform called, but no batch dim found?")

        trace_batch_l: list[dict[Feats, torch.Tensor]] = []

        max_len = 0
        for i in range(bs):
            current_trace = {key: val[i] for key, val in trace_batch.items()}
            trace_: dict[Feats, torch.Tensor] = {}
            for tr in self._feature_trs:
                out = tr(current_trace)
                if len(out) != 1:
                    raise ValueError(
                        f"Transform {tr.name} returned more than one tensor."
                    )

                trace_.update(out)
            trace_batch_l.append(trace_)

            if pad_val is not None:
                v_l = []
                for k, v in trace_.items():
                    if v.ndim != 1:
                        raise ValueError(
                            "Padding batch only supports 1D tensors, "
                            + "but got {v.ndim}D tensor for key {k}"
                        )
                    v_l.append(v.shape[0])

                    if len(set(v_l)) != 1:
                        raise ValueError(
                            "All tensors in a trace must have the same "
                            + f"length for padding, but got lengths {v_l}"
                        )

                max_len = max(max_len, v_l[0])

        if pad_val is not None:
            for trace_ in trace_batch_l:
                for k, v in trace_.items():
                    trace_[k] = torch.cat(
                        v, torch.full((max_len - v.shape[0],), pad_val)
                    )

        transformed_trace_batch: dict[Feats, torch.Tensor] = {
            k: torch.stack([d[k] for d in trace_batch_l], dim=0)
            for k in trace_batch_l[0].keys()
        }

        return transformed_trace_batch


def build_feature_trs(feature_name: list[Feats], n_packets: int) -> list[_TR]:
    return [get_feature_tr(f, n_packets) for f in feature_name]


def get_feature_tr(
    feature_name: Feats,
    n_packets: int | None,
    tam_kwargs: dict[str, Any] | None = None,
) -> _TR:
    tam_kwargs = tam_kwargs or {}
    match feature_name:
        case Feats.DIRS:
            return Compose(
                PadOrCutTrace(n_packets),
                Select(Feats.DIRS),
            )
        case Feats.DIR_PROBS:
            return Compose(PadOrCutTrace(n_packets), DirProbs())
        case Feats.SIZES:
            return Compose(
                PadOrCutTrace(n_packets),
                Select(Feats.SIZES),
            )
        case Feats.TIMES:
            return Compose(
                PadOrCutTrace(n_packets),
                Select(Feats.TIMES),
            )
        case Feats.PADDING:
            return Compose(
                PadOrCutTrace(n_packets),
                Select(Feats.PADDING),
            )
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
        case Feats.LOG1P_IATS:
            return Compose(
                PadOrCutTrace(n_packets),
                IAT("any", time_asset=Feats.TIMES, dir_asset=Feats.DIRS),
                Log1p(Feats.IATS),
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
            return Compose(PadOrCutTrace(n_packets), BurstEdges())
        case Feats.BURST_LENS:
            return Compose(BurstEdges(), BurstLens())
        case Feats.BURST_DURS:
            return Compose(BurstEdges(), BurstDurs())
        case Feats.BURST_RELDURS:
            return Compose(BurstEdges(), BurstDurs(), BurstRelativeDurs())
        case Feats.BURST_DIRS:
            return Compose(BurstEdges(), BurstDirs())
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
                IATDirs(
                    dir_asset=Feats.DIRS,
                    iat_asset=Feats.LOG_INV_FLOW_IATS,
                    add_unit_to_weight=False,
                ),
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
                    add_unit_to_weight=False,
                ),
            )
        case Feats.RUNNING_RATE_SIZES:
            return Compose(
                PadOrCutTrace(n_packets),
                RunningRate(Feats.SIZES, Feats.TIMES),
            )
        case Feats.RUNNING_RATE_SIZES_MAX_NORMALIZED:
            return Compose(
                PadOrCutTrace(n_packets),
                RunningRate(Feats.SIZES, Feats.TIMES),
                Normalize(
                    normalized_asset=Feats.RUNNING_RATE_SIZES,
                    input_asset=Feats.RUNNING_RATE_SIZES,
                    division="max",
                ),
            )
        case Feats.SIZE_DIRS:
            return Compose(
                PadOrCutTrace(n_packets),
                SizeDirs(dir_asset=Feats.DIRS, size_asset=Feats.SIZES),
            )
        case Feats.CUM_SIZE_DIRS:
            return Compose(
                PadOrCutTrace(n_packets),
                SizeDirs(dir_asset=Feats.DIRS, size_asset=Feats.SIZES),
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
        case Feats.TAM_UP_COUNTS:
            return TAM_UP(**tam_kwargs)
        case Feats.TAM_UP_PAD:
            return TAM_UP_PAD(**tam_kwargs)
        case Feats.TAM_UP_COUNTS_MAX_NORMALIZED:
            return Compose(
                TAM_UP(**tam_kwargs),
                Normalize(
                    normalized_asset=Feats.TAM_UP_COUNTS,
                    input_asset=Feats.TAM_UP_COUNTS,
                    division="max",
                ),
            )
        case Feats.TAM_DOWN_COUNTS:
            return TAM_DOWN(**tam_kwargs)
        case Feats.TAM_DOWN_PAD:
            return TAM_DOWN_PAD(**tam_kwargs)
        case Feats.TAM_BINS:
            return TAM_BINS(**tam_kwargs)
        case Feats.TAM_TIMES:
            return TAM_TIMES(**tam_kwargs)
        case Feats.TAM_DOWN_COUNTS_MAX_NORMALIZED:
            return Compose(
                TAM_DOWN(**tam_kwargs),
                Normalize(
                    normalized_asset=Feats.TAM_DOWN_COUNTS,
                    input_asset=Feats.TAM_DOWN_COUNTS,
                    division="max",
                ),
            )
        case _:
            raise ValueError(f"Unknown feature name: {feature_name}")
