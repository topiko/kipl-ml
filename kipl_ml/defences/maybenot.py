from __future__ import annotations

import os

import torch
from mbnt import sim_trace_from_file_advanced

from kipl_ml.data import assets
from kipl_ml.data.utils import parse_trace_to_tensor_dict
from kipl_ml.defences.base import DEFENCE_TYPE_KW, _Def
from kipl_ml.logging.logger import get_logger
from kipl_ml.logging.utils import log_multiline
from kipl_ml.trace.params import EVENTS_MULTIPLIER, MAX_TRACE_LENGTH

logger = get_logger(__name__)


class Maybenot(_Def):

    def __init__(
        self,
        deck_path: os.PathLike,
        network_delay_millis: tuple[int, int],
        network_pps: tuple[int, int],
        seed: int | None = 42,
        fixed_per_trace: bool = False,
    ):

        super().__init__(
            network_delay_millis=network_delay_millis,
            network_pps=network_pps,
            seed=seed,
            fixed_per_trace=fixed_per_trace,
        )

    def report(self, to_log: bool = True) -> str:
        str_ = "Maybenot Defence:\n"
        str_ += "\t" + self.deck.report().replace("\n", "\n\t")
        str_ += f"\t{self.network_delay_millis}\n"
        str_ += f"\t{self.network_pps}\n"

        if to_log:
            log_multiline(str_)
        return str_

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
            self.network_pps(),
            max_padding_frac_client=max_padding_frac_client,
            max_padding_frac_server=max_padding_frac_server,
            max_blocking_frac_client=max_blocking_frac_client,
            max_blocking_frac_server=max_blocking_frac_server,
            max_trace_length=MAX_TRACE_LENGTH,
            events_multiplier=EVENTS_MULTIPLIER,
        )

        trace_d = parse_trace_to_tensor_dict(times, dirs, paddings, None)

        if trace_d[assets.TIMES].shape[0] == 0:
            raise ValueError("Empty trace!")

        return trace_d

    def _mlflow_log_params(self) -> dict[str, str]:

        d = {}
        d[DEFENCE_TYPE_KW] = f"{self.__class__.__name__.lower()}"

        return d
