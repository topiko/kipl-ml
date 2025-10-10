from __future__ import annotations

import os

import torch
from mbnt import deal_machines, sim_trace_from_file_advanced

from kipl_ml.data import assets
from kipl_ml.data.utils import parse_trace_to_tensor_dict
from kipl_ml.defences.base import DEFENCE_TYPE_KW, _Def
from kipl_ml.logging.logger import get_logger
from kipl_ml.logging.utils import log_multiline
from kipl_ml.tools.rng_samplers import MachineRng
from kipl_ml.trace.params import EVENTS_MULTIPLIER, MAX_TRACE_LENGTH

logger = get_logger(__name__)


class Maybenot(_Def):
    def __init__(
        self,
        deck_path: os.PathLike,
        network_delay_millis: tuple[int, int],
        network_pps: tuple[int, int],
        n_machines: int,
        scale: float,
        client_padding_budget: tuple[int, int],
        client_blocking_budget: tuple[int, int],
        client_padding_frac: tuple[float, float],
        client_blocking_frac: tuple[float, float],
        server_padding_budget: tuple[int, int],
        server_blocking_budget: tuple[int, int],
        server_padding_frac: tuple[float, float],
        server_blocking_frac: tuple[float, float],
        seed: int | None = 42,
        fixed_per_trace: bool = False,
        simul_kwargs: dict | None = None,
    ):
        super().__init__(
            network_delay_millis=network_delay_millis,
            network_pps=network_pps,
            seed=seed,
            fixed_per_trace=fixed_per_trace,
        )

        self.limits = {
            "client": {
                "padding_budget": client_padding_budget,
                "blocking_budget": client_blocking_budget,
                "padding_frac": client_padding_frac,
                "blocking_frac": client_blocking_frac,
            },
            "server": {
                "padding_budget": server_padding_budget,
                "blocking_budget": server_blocking_budget,
                "padding_frac": server_padding_frac,
                "blocking_frac": server_blocking_frac,
            },
        }

        self.scale = scale
        self.n_machines = n_machines

        if not os.path.isfile(deck_path):
            raise ValueError(f"No deck found in: {deck_path}")

        self.machines = deal_machines(
            str(deck_path), self.limits, n_machines, scale, seed=seed
        )
        self.machine_rng = MachineRng(len(self.machines))
        self.deck_path = deck_path
        self.simul_kwargs = simul_kwargs or {}

    def report(self, to_log: bool = True) -> str:
        str_ = "Maybenot Defence:\n"
        str_ += f"\tDeck: {self.deck_path}\n"
        str_ += f"\tN machines: {len(self.machines)}\n"
        str_ += f"\tScale: {self.scale}\n"
        str_ += f"\t{self.network_delay_millis}\n"
        str_ += f"\t{self.network_pps}\n"
        str_ += f"\tFixed per trace: {self.FIXED_PER_TRACE}\n"

        if self.simul_kwargs:
            str_ += "Simul. args\n"
            for k, v in self.simul_kwargs.items():
                str_ += f"\t{k} : {v}\n"

        if to_log:
            log_multiline(str_)
        return str_

    def _get_machines(
        self, idx: int | None = None
    ) -> tuple[dict[str, float], tuple[list[str], list[str]]]:
        idx = idx or self.machine_rng()

        try:
            d = self.machines[idx].copy()
        except IndexError as e:
            raise IndexError(
                f"You provided machine {idx}, however there is only \
                {len(self.machines)} machines available!"
            ) from e

        server_machines = d.pop("server_machines")
        client_machines = d.pop("client_machines")

        return d, (client_machines, server_machines)

    def _simulate(
        self, trace_path: os.PathLike, machine_idx: int | None = None
    ) -> dict[Feats, torch.Tensor]:
        pad_bloc_fracs, (client_machines, server_machines) = self._get_machines(
            machine_idx
        )
        times, dirs, paddings = sim_trace_from_file_advanced(
            str(trace_path),
            client_machines,
            server_machines,
            self.network_delay_millis(),
            self.network_pps(),
            **pad_bloc_fracs,
            max_trace_length=self.simul_kwargs.get(
                "max_trace_length", MAX_TRACE_LENGTH
            ),
            random_state=self.simul_rng(),
            events_multiplier=self.simul_kwargs.get(
                "events_multiplier", EVENTS_MULTIPLIER
            ),
        )

        trace_d = parse_trace_to_tensor_dict(times, dirs, paddings, None)

        if trace_d[assets.TIMES].shape[0] == 0:
            raise ValueError("Empty trace!")

        return trace_d

    def _mlflow_log_params(self) -> dict[str, str]:
        d = {}
        d[DEFENCE_TYPE_KW] = f"{self.__class__.__name__.lower()}"
        d["fixed_per_trace"] = str(self.FIXED_PER_TRACE)
        d["scale"] = str(self.scale)
        d["deck"] = str(self.deck_path).rsplit("/", maxsplit=1)[-1]
        d["n_machines"] = str(len(self.machines))

        return d
