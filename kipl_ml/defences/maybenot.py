from __future__ import annotations

import json
import os
import random

import numpy as np
import torch
from mbnt import apply_machine_budget, sim_trace_from_file_advanced

from kipl_ml.data import assets
from kipl_ml.data.utils import parse_trace_to_tensor_dict
from kipl_ml.defences.base import DEFENCE_TYPE_KW, _Def
from kipl_ml.logging.logger import get_logger
from kipl_ml.logging.utils import log_multiline
from kipl_ml.network.network import NetworkContext, NetworkContextIntDict
from kipl_ml.tools.rng_samplers import MachineRng
from kipl_ml.trace.enums import Feats
from kipl_ml.trace.params import EVENTS_MULTIPLIER, MAX_TRACE_LENGTH

logger = get_logger(__name__)


class _MachineDeck:
    """Deterministic shuffled view over concrete deck entries."""

    def __init__(
        self,
        deck_path: os.PathLike,
        limits: dict,
        n_machines: int,
        scale: float,
        as_is: bool = False,
        seed: int | None = 42,
    ):
        with open(deck_path) as f:
            lines = f.readlines()
        self._defenses = [json.loads(line) for line in lines[1:]]
        self._as_is = as_is

        if not self._defenses:
            raise ValueError(f"No machine entries found in: {deck_path}")
        if n_machines > len(self._defenses):
            raise ValueError(
                f"Requested {n_machines} machines from deck with "
                f"{len(self._defenses)} entries: {deck_path}"
            )

        def _ordered(a, b):
            return (min(a, b), max(a, b))

        self._client_pf = _ordered(*limits["client"]["padding_frac"])
        self._client_bf = _ordered(*limits["client"]["blocking_frac"])
        self._server_pf = _ordered(*limits["server"]["padding_frac"])
        self._server_bf = _ordered(*limits["server"]["blocking_frac"])

        self._client_pb = limits["client"]["padding_budget"]
        self._client_db = limits["client"]["blocking_budget"]
        self._server_pb = limits["server"]["padding_budget"]
        self._server_db = limits["server"]["blocking_budget"]
        self._scale = scale

        rng = random.Random(seed)
        order = list(range(len(self._defenses)))
        rng.shuffle(order)
        self._order = order[:n_machines]

        self._n = n_machines
        self._seed = seed

    def __len__(self) -> int:
        return self._n

    @staticmethod
    def _sub_seed(seed: int | None, idx: int) -> int:
        """Deterministic per-(seed, idx) 32-bit hash (not Python's randomized hash)."""
        base = 0 if seed is None else seed
        h = (base * 0x9E3779B9 + idx * 0x85EBCA6B + 0xC4CEB9FE) & 0xFFFFFFFF
        return h

    def __getitem__(self, idx: int) -> dict:
        if idx < 0:
            idx += self._n
        if idx < 0 or idx >= self._n:
            raise IndexError(f"machine index {idx} out of range [0, {self._n})")

        defense = self._defenses[self._order[idx]]

        if self._as_is:
            return {
                "max_padding_frac_client": 0.0,
                "max_padding_frac_server": 0.0,
                "max_blocking_frac_client": 0.0,
                "max_blocking_frac_server": 0.0,
                "client_machines": list(defense["client"]),
                "server_machines": list(defense["server"]),
            }

        if self._seed is not None:
            sub_seed = self._sub_seed(self._seed, idx)
            rng = np.random.default_rng(sub_seed)
        else:
            rng = np.random.default_rng()

        # Sample fraction limits
        max_padding_frac_client = float(rng.uniform(*self._client_pf))
        max_padding_frac_server = float(rng.uniform(*self._server_pf))
        max_blocking_frac_client = float(rng.uniform(*self._client_bf))
        max_blocking_frac_server = float(rng.uniform(*self._server_bf))

        # Sample budgets (scaled) and apply via Rust sidecar
        cd_budget = rng.uniform(*self._client_pb) * self._scale
        ck_budget = rng.uniform(*self._client_db) * self._scale
        sd_budget = rng.uniform(*self._server_pb) * self._scale
        sk_budget = rng.uniform(*self._server_db) * self._scale

        client_machines, server_machines = apply_machine_budget(
            defense["client"], defense["server"],
            cd_budget, ck_budget, sd_budget, sk_budget,
        )

        return {
            "max_padding_frac_client": max_padding_frac_client,
            "max_padding_frac_server": max_padding_frac_server,
            "max_blocking_frac_client": max_blocking_frac_client,
            "max_blocking_frac_server": max_blocking_frac_server,
            "client_machines": list(client_machines),
            "server_machines": list(server_machines),
        }


class Maybenot(_Def):
    def __init__(
        self,
        deck_path: os.PathLike,
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
        as_is: bool = False,
        flavor: str | None = None,
    ):
        super().__init__(seed=seed, fixed_per_trace=fixed_per_trace)

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
        self.as_is = as_is
        self.flavor = flavor

        if not os.path.isfile(deck_path):
            raise ValueError(f"No deck found in: {deck_path}")

        self.machines = _MachineDeck(
            str(deck_path), self.limits, n_machines, scale, as_is=as_is, seed=seed
        )
        self.machine_rng = MachineRng(len(self.machines))
        self.deck_path = deck_path
        self.simul_kwargs = simul_kwargs or {}

    def report(self, to_log: bool = True) -> str:
        str_ = "Maybenot Defence:\n"
        str_ += f"\tFlavor: {self.flavor or '<none>'}\n"
        str_ += f"\tDeck: {self.deck_path}\n"
        str_ += f"\tN machines: {len(self.machines)}\n"
        str_ += f"\tScale: {self.scale}\n"
        str_ += f"\tAs-is: {self.as_is}\n"
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
        if idx is None:
            idx = self.machine_rng()

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
        self,
        trace_path: os.PathLike,
        machine_idx: int | None = None,
        trim_raw: int = 0,
        network_context: NetworkContextIntDict | None = None,
    ) -> dict[Feats, torch.Tensor]:
        if network_context is None:
            raise ValueError(f"{self.__class__.__name__} requires explicit network_context")
        network_kwargs = NetworkContext.to_rust_args(network_context)
        pad_bloc_fracs, (client_machines, server_machines) = self._get_machines(
            machine_idx
        )
        random_state = (
            self.simul_kwargs["random_state"]
            if "random_state" in self.simul_kwargs
            else self.simul_rng()
        )
        times, dirs, paddings = sim_trace_from_file_advanced(
            str(trace_path),
            client_machines,
            server_machines,
            network_kwargs=network_kwargs,
            **pad_bloc_fracs,
            max_trace_length=self.simul_kwargs.get(
                "max_trace_length", MAX_TRACE_LENGTH
            ),
            random_state=random_state,
            events_multiplier=self.simul_kwargs.get(
                "events_multiplier", EVENTS_MULTIPLIER
            ),
            trim_raw=trim_raw,
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
        d["as_is"] = str(self.as_is)
        d["flavor"] = self.flavor or ""
        d["deck"] = str(self.deck_path).rsplit("/", maxsplit=1)[-1]
        d["n_machines"] = str(len(self.machines))

        return d
