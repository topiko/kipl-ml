import os

import kipl_ml.data.assets as assets
import numpy as np
import torch
import yaml
from kipl_ml.data.utils import parse_trace_to_tensor_dict
from kipl_ml.defences.base import _Def
from kipl_ml.logging.logger import get_logger
from rustbindings import sim_trace_from_file
from torch.distributions.chi2 import Chi2

logger = get_logger(__name__)


class Maybenot(_Def):
    def __init__(self, deck_path: os.PathLike, network_delay_millis: int):
        self.machines: list[dict[str, str]] = self._load_machines(deck_path)
        self.network_delay_millis: np.uint64 = np.uint64(network_delay_millis)

    def _load_machines(self, deck_path: os.PathLike) -> list[dict[str, str]]:
        logger.info(f"Loading machines from {deck_path}")
        with open(deck_path, "r") as fi:
            defenses = yaml.safe_load(fi)["defenses"]

        return [d[0] for d in defenses]

    def report(self, to_log: bool = True) -> str:
        return "Maybenot"

    def sim_defence(self, trace_path: os.PathLike) -> dict[str, torch.Tensor]:

        machine_idx = np.random.choice(len(self.machines))
        times, dirs, paddings = sim_trace_from_file(
            trace_path,
            self.machines[machine_idx]["client"],
            self.machines[machine_idx]["server"],
            self.network_delay_millis,
        )

        trace_d = parse_trace_to_tensor_dict(times, dirs, paddings, None)
        return trace_d

    def __call__(self, trace: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        raise NotImplementedError("Maybenot has its own perks...")
