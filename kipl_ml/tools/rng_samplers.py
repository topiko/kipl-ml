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


class TraceSimulRng(RandSampler):
    def __init__(self, seed: int | None = 42):
        super().__init__(0, 2**63 - 1, seed=seed)

    def __str__(self):
        return f"Trace simul rng: {self.way} [{self.low_lim}, {self.up_lim}]."


class MachineRng(RandSampler):
    def __init__(self, n_machines: int, seed: int | None = 42):
        super().__init__(0, n_machines - 1, seed=seed)

    def __str__(self):
        return f"Machine rng: {self.way} [{self.low_lim}, {self.up_lim}]."


class NetwkRtt(RandSampler):
    def __init__(self, min_rtt_ms: int, max_rtt_ms: int, seed: int | None = 42):

        if min_rtt_ms <= 0 or max_rtt_ms <= 0:
            raise ValueError("Network delay cannot be 0 -> issues in simul.")
        super().__init__(min_rtt_ms, max_rtt_ms, seed=seed)

    def __str__(self):
        return f"Netwk rtt fun: {self.way} [{self.low_lim}, {self.up_lim}] ms."

    def mlflow_log_params(self) -> dict[str, str]:
        return {
            "netwk_rtt_type": self.way,
            "netwk_rtt_min_ms": str(self.low_lim),
            "netwk_rtt_max_ms": str(self.up_lim),
        }


class NetwkMbps(RandSampler):
    def __init__(self, min_mbps: int, max_mbps: int, seed: int | None = 42):

        super().__init__(min_mbps, max_mbps, seed=seed)

    def __str__(self):
        return f"Netwk mbps fun: {self.way} [{self.low_lim}, {self.up_lim}] mbps."

    def mlflow_log_params(self) -> dict[str, str]:
        return {
            "netwk_mbps_type": self.way,
            "netwk_mbps_min_mbps": str(self.low_lim),
            "netwk_mbps_max_mbps": str(self.up_lim),
        }
