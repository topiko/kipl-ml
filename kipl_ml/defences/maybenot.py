from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Self

import numpy as np
import torch
import yaml
from mbnt import sim_trace_from_file_advanced

from kipl_ml.data import assets
from kipl_ml.data.utils import parse_trace_to_tensor_dict
from kipl_ml.defences.base import (
    DEFENCE_TYPE_KW,
    NetwkDelay,
    _Def,
    parse_netwk_delay_fun,
)
from kipl_ml.logging.logger import get_logger
from kipl_ml.logging.utils import key_val_fmt, log_multiline
from kipl_ml.trace.params import MAX_TRACE_LENGTH

logger = get_logger(__name__)


@dataclass
class DeckStats:
    n_machines: int
    n_client: int
    n_server: int
    deck_path: os.PathLike

    @classmethod
    def stats_path(cls, deck_path: os.PathLike) -> os.PathLike:
        dir_ = os.path.dirname(deck_path)
        deck = os.path.basename(deck_path).replace(".yaml", "")

        return Path(os.path.join(dir_, f"{deck}-stats.yaml"))

    @classmethod
    def load(cls, deck_path: os.PathLike) -> Self:
        try:
            with open(DeckStats.stats_path(deck_path), "r", encoding="utf-8") as fi:
                stats = yaml.safe_load(fi)

                logger.info(
                    f"Loaded deck stats from: {DeckStats.stats_path(deck_path)}"
                )
            stats = DeckStats(**stats)
        except FileNotFoundError:
            machines = load_machines(deck_path)

            n_machines = len(machines)
            n_client = len(machines[0]["client"])
            n_server = len(machines[0]["server"])

            stats = DeckStats(n_machines, n_client, n_server, deck_path)

        return stats

    def save(self):
        with open(DeckStats.stats_path(self.deck_path), "w", encoding="utf-8") as fi:
            yaml.safe_dump(asdict(self), fi)
            logger.info(f"Saved deck stats to: {DeckStats.stats_path(self.deck_path)}")

    def mlflow_log_params(self) -> dict[str, str]:
        return {
            "n_machines": str(self.n_machines),
            "client_machines": str(self.n_client),
            "server_machines": str(self.n_server),
            "deck_path": str(self.deck_path),
        }


class Deck:

    def __init__(self, stats: DeckStats, machine_idxs: list[int] | None = None):

        self.stats = stats
        # Note: machines are set when calling this:
        self.machine_idxs = machine_idxs or list(range(stats.n_machines))

    def get_machines(self, idx: int | None = None) -> tuple[list[str], list[str]]:

        idx = idx or np.random.choice(len(self.machine_idxs))
        try:
            return self.machines[idx]["client"], self.machines[idx]["server"]
        except IndexError as e:
            raise IndexError(
                f"You provided machine {idx}, however there is only \
                {len(self.machine_idxs)} machines available!"
            ) from e

    def report(self, to_log: bool = False) -> str:

        str_ = "Deck:\n"
        str_ += key_val_fmt("Deck path", self.stats.deck_path)
        str_ += key_val_fmt("N machines", self.stats.n_machines)
        if len(self.machine_idxs) < 20:
            str_ += key_val_fmt(
                "machine_idxs", "|".join([str(i) for i in self.machine_idxs])
            )
        str_ += key_val_fmt("N client machines", self.stats.n_client)
        str_ += key_val_fmt("N server machines", self.stats.n_server)

        if to_log:
            log_multiline(str_)

        return str_

    @property
    def machine_idxs(self) -> list[int]:
        return self._machine_idxs

    @machine_idxs.setter
    def machine_idxs(self, machine_idxs: list[int]):
        self._machine_idxs = machine_idxs
        self.stats.n_machines = len(self.machine_idxs)
        self.machines = load_machines(self.stats.deck_path, self.machine_idxs)

    def mlflow_log_params(self) -> dict[str, str]:
        d = self.stats.mlflow_log_params()
        d["n_machines"] = str(len(self.machine_idxs))
        return d


def load_machines(
    deck_path: os.PathLike, machine_idxs: list[int] | None = None
) -> list[dict[str, list[str]]]:
    logger.info("Loading machines from: %s", deck_path)
    with open(deck_path, "r", encoding="utf-8") as fi:
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
        deck: Deck,
        network_delay_millis: int | tuple[int, int] | NetwkDelay,
        padding_frac_client: str = "random",
        padding_frac_server: str = "random",
        blocking_frac_client: str = "no-blocking",
        blocking_frac_server: str = "no-blocking",
        max_padding_frac: float = 1.0,
        max_blocking_frac: float = 0.0,
        fixed_per_trace: bool = False,
    ):
        self.deck = deck
        self.network_delay_millis = parse_netwk_delay_fun(network_delay_millis)

        self.FIXED_PER_TRACE = fixed_per_trace

        if not all(f == "random" for f in (padding_frac_client, padding_frac_server)):
            raise ValueError("Only random padding is supported for now")

        if not all(
            f == "no-blocking" for f in (blocking_frac_client, blocking_frac_server)
        ):
            raise ValueError("Only no-blocking is supported for now")

        self.padding_frac_client = padding_frac_client
        self.padding_frac_server = padding_frac_server
        self.blocking_frac_client = blocking_frac_client
        self.blocking_frac_server = blocking_frac_server
        self.max_padding_frac = max_padding_frac
        self.max_blocking_frac = max_blocking_frac

    def report(self, to_log: bool = True) -> str:
        str_ = "Maybenot Defence:\n"
        str_ += "\t" + self.deck.report().replace("\n", "\n\t")
        str_ += f"Max padding frac: {self.max_padding_frac}\n"
        str_ += f"\tMax blocking frac: {self.max_blocking_frac}\n"
        str_ += f"\t{self.network_delay_millis}\n"

        if to_log:
            log_multiline(str_)
        return str_

    def _get_paddings(self) -> tuple[float, float]:
        def get_padding_frac(way: str) -> float:
            if way == "random":
                return np.random.uniform(0.0, self.max_padding_frac)
            raise NotImplementedError(f"Padding way {way} not implemented")

        client_padding = get_padding_frac(self.padding_frac_client)
        server_padding = get_padding_frac(self.padding_frac_server)

        return client_padding, server_padding

    def _get_blocking_fracs(self) -> tuple[float, float]:
        def get_blocking_frac(way: str) -> float:
            if way == "no-blocking":
                return 0.0
            raise NotImplementedError(f"Blocking way {way} not implemented")

        client_blocking = get_blocking_frac(self.blocking_frac_client)
        server_blocking = get_blocking_frac(self.blocking_frac_server)

        return client_blocking, server_blocking

    def _simulate(
        self, trace_path: os.PathLike, machine_idx: int | None = None
    ) -> dict[str, torch.Tensor]:

        max_padding_frac_client, max_padding_frac_server = self._get_paddings()

        max_blocking_frac_client, max_blocking_frac_server = self._get_blocking_fracs()

        client_machines, server_machines = self.deck.get_machines(machine_idx)
        times, dirs, paddings = sim_trace_from_file_advanced(
            str(trace_path),
            client_machines,
            server_machines,
            self.network_delay_millis(),
            max_padding_frac_client=max_padding_frac_client,
            max_padding_frac_server=max_padding_frac_server,
            max_blocking_frac_client=max_blocking_frac_client,
            max_blocking_frac_server=max_blocking_frac_server,
            max_trace_length=MAX_TRACE_LENGTH,
        )

        trace_d = parse_trace_to_tensor_dict(times, dirs, paddings, None)

        if trace_d[assets.TIMES].shape[0] == 0:
            raise ValueError("Empty trace!")

        return trace_d

    def _mlflow_log_params(self) -> dict[str, str]:

        n_machines = len(self.deck.machine_idxs)
        d = {}
        d[DEFENCE_TYPE_KW] = f"{self.__class__.__name__.lower()} w. {n_machines}"
        d["padding_frac_client"] = self.padding_frac_client
        d["padding_frac_server"] = self.padding_frac_server
        d["blocking_frac_client"] = self.blocking_frac_client
        d["blocking_frac_server"] = self.blocking_frac_server
        d["max_padding_frac"] = str(self.max_padding_frac)
        d["max_blocking_frac"] = str(self.max_blocking_frac)
        d.update(self.deck.mlflow_log_params())

        return d
