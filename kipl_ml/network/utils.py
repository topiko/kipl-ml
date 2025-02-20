from __future__ import annotations

import numpy as np

from kipl_ml.logging.logger import get_logger

logger = get_logger(__name__)


class RandSampler:
    def __init__(self, low_lim: int, up_lim: int, seed: int | None = 42):
        self.low_lim = low_lim
        self.up_lim = up_lim

        if low_lim == up_lim:
            self.way = "fixed"
        elif up_lim > low_lim:
            self.way = "random"
        else:
            raise ValueError("Invalid up low lim combination")

        if self.way == "random":
            self.rng = np.random.default_rng(seed)

    def __call__(self):
        match self.way:
            case "random":
                return self._cast_to_uint64(
                    self.rng.integers(self.low_lim, self.up_lim + 1)
                )
            case "fixed":
                return self._cast_to_uint64(self.low_lim)
            case _:
                raise NotImplementedError()

    def _cast_to_uint64(self, val: int) -> np.uint64:
        return np.uint64(val)


class NetwkDelay(RandSampler):
    def __init__(self, min_delay_ms: int, max_delay_ms: int, seed: int | None = 42):

        super().__init__(min_delay_ms, max_delay_ms, seed=seed)

    def __str__(self):
        return f"Netwk delay fun: {self.way} [{self.low_lim}, {self.up_lim}]ms."

    def mlflow_log_params(self) -> dict[str, str]:
        return {
            "netwk_delay_type": self.way,
            "netwk_delay_min_ms": str(self.low_lim),
            "netwk_delay_max_ms": str(self.up_lim),
        }


class NetwkPps(RandSampler):
    def __init__(self, min_pps: int, max_pps: int, seed: int | None = 42):

        super().__init__(min_pps, max_pps, seed=seed)

    def __str__(self):
        return f"Netwk pps fun: {self.way} [{self.low_lim}, {self.up_lim}]ms."

    def mlflow_log_params(self) -> dict[str, str]:
        return {
            "netwk_pps_type": self.way,
            "netwk_pps_min_pps": str(self.low_lim),
            "netwk_pps_max_pps": str(self.up_lim),
        }
