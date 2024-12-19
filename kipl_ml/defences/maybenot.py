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


def load_machines(deck_path: os.PathLike) -> list[dict[str, list[str]]]:
    logger.info(f"Loading machines from {deck_path}")
    with open(deck_path, "r") as fi:
        defenses = yaml.safe_load(fi)["defenses"]

    return [d[0] for d in defenses]


class Maybenot(_Def):
    def __init__(self, deck_path: os.PathLike, network_delay_millis: int):
        self.machines: list[dict[str, list[str]]] = load_machines(deck_path)
        self.network_delay_millis: np.uint64 = np.uint64(network_delay_millis)

    def report(self, to_log: bool = True) -> str:
        str_ = "Maybenot Defence\n"
        str_ += f"\tNumber of machines: {len(self.machines)}\n"
        str_ += f"\tclient: {len(self.machines[0]['client']):02d}\n"
        str_ += f"\tserver: {len(self.machines[0]['server']):02d}\n"
        str_ += f"\tNetwork delay: {self.network_delay_millis} ms\n"
        return str_

    def sim_defence(
        self, trace_path: os.PathLike, machine_idx: int | None = None
    ) -> dict[str, torch.Tensor]:

        machine_idx = machine_idx or np.random.choice(len(self.machines))
        times, dirs, paddings = sim_trace_from_file(
            trace_path,
            self.machines[machine_idx]["client"],
            self.machines[machine_idx]["server"],
            self.network_delay_millis,
        )

        trace_d = parse_trace_to_tensor_dict(times, dirs, paddings, None)

        if trace_d[assets.TIMES].shape[0] == 0:
            logger.warning(f"Empty trace for {trace_path}")
            logger.warning(f"machine_idx: {machine_idx}")

        return trace_d

    def __call__(self, trace: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        raise NotImplementedError("Maybenot has its own perks...")
