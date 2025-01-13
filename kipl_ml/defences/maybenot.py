import os

import kipl_ml.data.assets as assets
import numpy as np
import torch
import yaml
from kipl_ml.data.utils import parse_trace_to_tensor_dict
from kipl_ml.defences.base import _Def
from kipl_ml.logging.logger import get_logger
from kipl_ml.trace.params import MAX_TRACE_LENGTH
from rustbindings import sim_trace_from_file_advanced

logger = get_logger(__name__)
MAX_PADDING_FRAC = 1.0


def load_machines(
    deck_path: os.PathLike, machine_idxs: list[int] | None = None
) -> list[dict[str, list[str]]]:
    logger.info(f"Loading machines from {deck_path}")
    with open(deck_path, "r") as fi:
        defenses = yaml.safe_load(fi)["defenses"]

    machine_idxs = machine_idxs or list(range(len(defenses)))

    if max(machine_idxs) >= len(defenses):
        raise ValueError(
            f"Machine idxs out of bounds: {max(machine_idxs)} >= {len(defenses)}"
        )

    return [d[0] for i, d in enumerate(defenses) if i in machine_idxs]


class Maybenot(_Def):
    def __init__(
        self,
        deck_path: os.PathLike,
        network_delay_millis: int,
        max_padding_frac_client: str = "random",
        max_padding_frac_server: str = "random",
        max_blocking_frac_client: str = "no-blocking",
        max_blocking_frac_server: str = "no-blocking",
        machine_idxs: list[int] | None = None,
    ):
        self.machines: list[dict[str, list[str]]] = load_machines(
            deck_path, machine_idxs
        )
        self.network_delay_millis: np.uint64 = np.uint64(network_delay_millis)

        if any(
            f != "random" for f in (max_padding_frac_client, max_padding_frac_server)
        ):
            raise ValueError("Only random padding is supported for now")
        if any(
            f != "no-blocking"
            for f in (max_blocking_frac_client, max_blocking_frac_server)
        ):
            raise ValueError("Only no-blocking is supported for now")

        self.max_padding_frac_client = max_padding_frac_client
        self.max_padding_frac_server = max_padding_frac_server
        self.max_blocking_frac_client = max_blocking_frac_client
        self.max_blocking_frac_server = max_blocking_frac_server

    def report(self, to_log: bool = True) -> str:
        str_ = "Maybenot Defence\n"
        str_ += f"\tNumber of machines: {len(self.machines)}\n"
        str_ += f"\tclient: {len(self.machines[0]['client']):02d}\n"
        str_ += f"\tserver: {len(self.machines[0]['server']):02d}\n"
        str_ += f"\tNetwork delay: {self.network_delay_millis} ms\n"
        return str_

    def _get_paddings(self) -> tuple[float, float]:
        def get_padding_frac(way: str) -> float:
            if way == "random":
                return np.random.uniform(0.0, MAX_PADDING_FRAC)
            raise NotImplementedError(f"Padding way {way} not implemented")

        client_padding = get_padding_frac(self.max_padding_frac_client)
        server_padding = get_padding_frac(self.max_padding_frac_server)

        return client_padding, server_padding

    def _get_blocking_fracs(self) -> tuple[float, float]:
        def get_blocking_frac(way: str) -> float:
            if way == "no-blocking":
                return 0.0
            raise NotImplementedError(f"Blocking way {way} not implemented")

        client_blocking = get_blocking_frac(self.max_blocking_frac_client)
        server_blocking = get_blocking_frac(self.max_blocking_frac_server)

        return client_blocking, server_blocking

    def sim_defence(self, trace_path: os.PathLike) -> dict[str, torch.Tensor]:

        machine_idx = np.random.choice(len(self.machines))

        max_padding_client, max_padding_server = self._get_paddings()
        max_blocking_client, max_blocking_server = self._get_blocking_fracs()

        times, dirs, paddings = sim_trace_from_file_advanced(
            trace_path,
            self.machines[machine_idx]["client"],
            self.machines[machine_idx]["server"],
            self.network_delay_millis,
            max_padding_frac_client=max_padding_client,
            max_padding_frac_server=max_padding_server,
            max_blocking_frac_client=max_blocking_client,
            max_blocking_frac_server=max_blocking_server,
            max_trace_length=MAX_TRACE_LENGTH,
        )

        trace_d = parse_trace_to_tensor_dict(times, dirs, paddings, None)

        if trace_d[assets.TIMES].shape[0] == 0:
            logger.warning(f"Empty trace for {trace_path}")
            logger.warning(f"machine_idx: {machine_idx}")

        return trace_d

    def __call__(self, trace: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        raise NotImplementedError("Maybenot has its own perks...")
