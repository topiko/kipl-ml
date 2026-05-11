from __future__ import annotations

import os
from abc import ABC, abstractmethod

import torch
from mbnt import sim_trace_from_file_advanced

from kipl_ml.data.utils import get_std_trace_dict, parse_trace_to_tensor_dict
from kipl_ml.logging.logger import get_logger
from kipl_ml.logging.utils import key_val_fmt
from kipl_ml.tools.rng_samplers import TraceSimulRng
from kipl_ml.trace.features import Feats
from kipl_ml.trace.params import EVENTS_MULTIPLIER, MAX_TRACE_LENGTH

logger = get_logger(__name__)
DEFENCE_TYPE_KW = "defence-type"
NET_DELAY_KW = "network_delay_millis"
NET_PPS_KW = "network_packets_per_second"
NetworkContext = dict[str, int]


class _Def(ABC):
    FIXED_PER_TRACE = False

    def __init__(
        self,
        seed: int | None = 42,
        fixed_per_trace: bool = False,
    ):
        self.simul_rng = TraceSimulRng(seed=seed)
        self.FIXED_PER_TRACE = fixed_per_trace
        self.seed = seed

    def _report(self, to_log: bool = True, **kwargs) -> str:
        str_ = f"{self.name}\n"
        for key, value in kwargs.items():
            str_ += key_val_fmt(key, value)

        if to_log:
            for line in str_.split("\n"):
                logger.info(line)

        return str_

    @property
    def name(self) -> str:
        return self.__class__.__name__

    def __call__(
        self,
        trace_path: os.PathLike,
        machine_idx: int | None = None,
        trim_raw: int = 0,
        network_context: NetworkContext | None = None,
    ) -> dict[Feats, torch.Tensor]:
        if not isinstance(trace_path, os.PathLike):
            raise TypeError(
                f"Expected trace to be os.PathLike, got: {type(trace_path)}"
            )

        return self._simulate(
            trace_path,
            machine_idx,
            trim_raw=trim_raw,
            network_context=network_context,
        )

    def load_data(self, trace_path: os.PathLike, trim_raw: int = 0) -> dict[Feats, torch.Tensor]:
        return get_std_trace_dict(trace_path, trim_raw=trim_raw)

    @abstractmethod
    def _simulate(
        self,
        trace_path: os.PathLike,
        machine_idx: int | None = None,
        trim_raw: int = 0,
        network_context: NetworkContext | None = None,
    ) -> dict[Feats, torch.Tensor]:
        raise NotImplementedError

    @abstractmethod
    def report(self, to_log: bool = True) -> str:
        raise NotImplementedError

    def mlflow_log_params(self) -> dict[str, str]:
        return {f"defence.{k}": v for k, v in self._mlflow_log_params().items()}

    @abstractmethod
    def _mlflow_log_params(self) -> dict[str, str]:
        raise NotImplementedError

    def _require_network_context(
        self, network_context: NetworkContext | None
    ) -> tuple[int, int]:
        if network_context is None:
            raise ValueError(
                f"{self.__class__.__name__} requires explicit network_context with "
                f"'{NET_DELAY_KW}' and '{NET_PPS_KW}'"
            )
        try:
            delay_ms = int(network_context[NET_DELAY_KW])
            pps = int(network_context[NET_PPS_KW])
        except KeyError as exc:
            raise KeyError(
                f"network_context must contain '{NET_DELAY_KW}' and '{NET_PPS_KW}'"
            ) from exc
        return delay_ms, pps


class NoDefence(_Def):
    def __init__(
        self,
        seed: int | None = 42,
        simul_kwargs: dict | None = None,
    ):
        super().__init__(seed=seed, fixed_per_trace=False)
        self.simul_kwargs = simul_kwargs or {}

    def report(self, to_log: bool = True) -> str:
        str_ = "No defence applied\n"

        if self.simul_kwargs:
            str_ += "Simul. args\n"
            for k, v in self.simul_kwargs.items():
                str_ += f"\t{k} : {v}\n"

        if to_log:
            logger.info(str_)
        return str_

    def _simulate(
        self,
        trace_path: os.PathLike,
        machine_idx: int | None = None,
        trim_raw: int = 0,
        network_context: NetworkContext | None = None,
    ) -> dict[Feats, torch.Tensor]:
        network_delay_millis, network_packets_per_second = self._require_network_context(
            network_context
        )
        times, dirs, paddings = sim_trace_from_file_advanced(
            str(trace_path),
            [],  # Empty machines --> no defence
            [],  # Empty machines --> no defence
            network_delay_millis,
            network_packets_per_second,
            max_padding_frac_client=0,
            max_padding_frac_server=0,
            max_blocking_frac_client=0,
            max_blocking_frac_server=0,
            max_trace_length=self.simul_kwargs.get(
                "max_trace_length", MAX_TRACE_LENGTH
            ),
            random_state=self.simul_rng(),
            events_multiplier=self.simul_kwargs.get(
                "events_multiplier", EVENTS_MULTIPLIER
            ),
            trim_raw=trim_raw,
        )
        trace_d = parse_trace_to_tensor_dict(times, dirs, paddings, None)
        return trace_d

    def _mlflow_log_params(self) -> dict[str, str]:
        d = {}
        d[DEFENCE_TYPE_KW] = self.__class__.__name__.lower()

        return d
