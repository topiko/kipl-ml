from __future__ import annotations

import os

import dotenv
import mlflow
import torch
from torch import nn

from kipl_ml.data.utils import get_std_trace_dict
from kipl_ml.defences.base import DEFENCE_TYPE_KW, _Def
from kipl_ml.logging.logger import get_logger
from kipl_ml.logging.utils import log_multiline
from kipl_ml.rl.action import send_exec
from kipl_ml.rl.observation import get_window_feature_dict
from kipl_ml.trace.enums import Feats

dotenv.load_dotenv()

logger = get_logger(__name__)


class _NNDef(_Def):
    def __init__(
        self,
        network_delay_millis: tuple[int, int],
        network_pps: tuple[int, int],
        obs_model: nn.Module | str,
        seed: int | None = 42,
        fixed_per_trace: bool = False,
        simul_kwargs: dict | None = None,
    ):
        if not network_delay_millis != (0, 0) or not network_pps != (0, 0):
            logger.warning(
                "NNdefs do not use the network simulator -> params. ignored."
            )
        if fixed_per_trace:
            logger.warning(
                "NNdefs do not use fixed_per_trace parameter -> param. ignored."
            )

        super().__init__(
            network_delay_millis=network_delay_millis,
            network_pps=network_pps,
            seed=seed,
            fixed_per_trace=fixed_per_trace,
        )

        self._model_id = None
        if isinstance(obs_model, str):
            self._model_id = obs_model
            self.defense_model = mlflow.pytorch.load_model(
                f"models:/{obs_model}", map_location="cpu"
            )

        self.simul_kwargs = simul_kwargs or {}

    def report(self, to_log: bool = False) -> str:
        str_ = self.__class__.__name__ + "\n"
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

    def _run_model(
        self, trace_d: dict[Feats, torch.Tensor]
    ) -> dict[Feats, torch.Tensor]:
        raise NotImplementedError

    def _simulate(
        self, trace_path: os.PathLike, machine_idx: int | None = None
    ) -> dict[Feats, torch.Tensor]:
        if machine_idx is not None:
            raise NotImplementedError(
                f"{self.__class__.__name__} does not support machine_idx argument."
            )
        trace_d = get_std_trace_dict(trace_path)

        trace_d.pop(Feats.SIZES)

        trace_d = self._run_model(trace_d)

        return trace_d

    def _mlflow_log_params(self) -> dict[str, str]:
        d = {}
        d[DEFENCE_TYPE_KW] = self.__class__.__name__.lower()
        d["model-id"] = str(self._model_id)

        return d


class RNNDef(_NNDef):
    def _run_model(
        self, trace_d: dict[Feats, torch.Tensor]
    ) -> dict[Feats, torch.Tensor]:
        # Implement RNN specific logic
        h = None

        trace_d = {k: v.unsqueeze(0).float() for k, v in trace_d.items()}

        fd = get_window_feature_dict(
            trace_d,
            self.defense_model.time_step,
            100.0,
            features=self.defense_model.features,
        )

        seq_lens = torch.Tensor([fd[self.defense_model.features[0]].shape[1]]).long()

        with torch.no_grad():
            act_times, actions = self.defense_model.act(
                fd, h, h_detach_period=100, seq_lens=seq_lens
            )[:2]

        trace_d = send_exec(trace_d, act_times, actions)

        trace_d = {k: v.squeeze(0) for k, v in trace_d.items()}

        trace_d[Feats.SIZES] = torch.ones_like(trace_d[Feats.TIMES])

        return trace_d
