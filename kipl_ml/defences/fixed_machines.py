from __future__ import annotations

import os
import tempfile
from abc import abstractmethod
from pathlib import Path

import dotenv
import torch
from mbnt import sim_trace_from_file_advanced

from kipl_ml.data.utils import parse_trace_to_tensor_dict
from kipl_ml.defences.base import DEFENCE_TYPE_KW, _Def
from kipl_ml.defences.maybenot import Deck, DeckStats
from kipl_ml.logging.logger import get_logger
from kipl_ml.logging.utils import log_multiline
from kipl_ml.network.utils import NetwkDelay, NetwkPps
from kipl_ml.trace.params import EVENTS_MULTIPLIER, MAX_TRACE_LENGTH

dotenv.load_dotenv()


logger = get_logger(__name__)


class _FixedMachine(_Def):

    def __init__(
        self,
        network_delay_millis: tuple[int, int],
        network_pps: tuple[int, int],
        machination_kwargs: dict[str, int | float],
        seed: int | None = 42,
        fixed_per_trace: bool = False,
    ):
        if (MACHINATION := os.getenv("MACHINATION")) is None:
            raise ValueError("machination not found in environment")

        self.FIXED_PER_TRACE = fixed_per_trace
        self._rust_machination = MACHINATION
        tmpfile_ = tempfile.mktemp(prefix=self.__class__.__name__, suffix=".defence")

        self._machination(tmpfile_, **machination_kwargs)
        deck_stats = DeckStats.load(Path(tmpfile_))
        self.deck = Deck(deck_stats)

        os.remove(tmpfile_)

        self.network_delay_millis = NetwkDelay(*network_delay_millis, seed=seed)
        self.network_pps = NetwkPps(*network_pps, seed=seed)

    def report(self, to_log: bool = False) -> str:
        str_ = self.__class__.__name__ + "\n"
        str_ += "\t" + self.deck.report(to_log=False)
        str_ += f"\t{self.network_delay_millis}\n"
        str_ += f"\t{self.network_pps}\n"

        if to_log:
            log_multiline(str_)

        return str_

    def _simulate(
        self, trace_path: os.PathLike, machine_idx: int | None = None
    ) -> dict[str, torch.Tensor]:

        client_machines, server_machines = self.deck.get_machines(machine_idx)

        times, dirs, paddings = sim_trace_from_file_advanced(
            str(trace_path),
            client_machines,
            server_machines,
            self.network_delay_millis(),
            self.network_pps(),
            max_padding_frac_client=0,
            max_padding_frac_server=0,
            max_blocking_frac_client=0,
            max_blocking_frac_server=0,
            max_trace_length=MAX_TRACE_LENGTH,
            events_multiplier=EVENTS_MULTIPLIER,
        )

        trace_d = parse_trace_to_tensor_dict(times, dirs, paddings, None)

        return trace_d

    @abstractmethod
    def _machination(self, tmpfile_: str, **kwargs) -> None:
        raise NotImplementedError

    def _mlflow_log_params(self) -> dict[str, str]:
        d = {k: str(v) for k, v in self.machination_kwargs.items()}
        d[DEFENCE_TYPE_KW] = self.__class__.__name__.lower()

        return d
