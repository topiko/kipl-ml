import os
from abc import ABC, abstractmethod

import torch
from kipl_ml.data.utils import get_std_trace_dict, parse_trace_to_tensor_dict
from kipl_ml.logging.logger import get_logger
from kipl_ml.logging.utils import key_val_fmt
from kipl_ml.trace.params import MAX_TRACE_LENGTH
from rustbindings import sim_trace_from_file_advanced

logger = get_logger(__name__)


class _Def(ABC):
    def _report(self, to_log: bool = True, **kwargs) -> str:
        str_ = f"{self.name}\n"
        for key, value in kwargs.items():
            str_ += key_val_fmt(key, value) + "\n"

        if to_log:
            for line in str_.split("\n"):
                logger.info(line)

        return str_

    @property
    def name(self) -> str:
        return self.__class__.__name__

    def __call__(self, trace_path: os.PathLike) -> dict[str, torch.Tensor]:
        if not isinstance(trace_path, os.PathLike):
            raise TypeError(
                f"Expected trace to be os.PathLike, got: {type(trace_path)}"
            )

        return self._simulate(trace_path)

    def load_data(self, trace_path: os.PathLike) -> dict[str, torch.Tensor]:
        return get_std_trace_dict(trace_path)

    @abstractmethod
    def _simulate(self, trace_path: os.PathLike) -> dict[str, torch.Tensor]:
        raise NotImplementedError

    @abstractmethod
    def report(self, to_log: bool = True) -> str:
        raise NotImplementedError


class NoDefence(_Def):

    def __init__(self, network_delay_millis: int):
        self.network_delay_millis = network_delay_millis

    def report(self, to_log: bool = True) -> str:
        str_ = "No defence applied\n"
        str_ += f"\tNetwork delay [ms]: {self.network_delay_millis}\n"
        if to_log:
            logger.info(str_)
        return str_

    def _simulate(self, trace_path: os.PathLike) -> dict[str, torch.Tensor]:

        times, dirs, paddings = sim_trace_from_file_advanced(
            str(trace_path),
            [],  # Empty machines --> no defence
            [],  # Empty machines --> no defence
            self.network_delay_millis,
            max_padding_frac_client=0,
            max_padding_frac_server=0,
            max_blocking_frac_client=0,
            max_blocking_frac_server=0,
            max_trace_length=MAX_TRACE_LENGTH,
        )
        trace_d = parse_trace_to_tensor_dict(times, dirs, paddings, None)
        return trace_d
