from __future__ import annotations

import os
import random
from collections.abc import Sequence
from typing import Any

import dotenv
import mlflow
import torch
from torch import nn

from kipl_ml.data.utils import get_std_trace_dict
from kipl_ml.defences.base import DEFENCE_TYPE_KW, _Def
from kipl_ml.logging.logger import get_logger
from kipl_ml.logging.utils import log_multiline
from kipl_ml.rl.simulate import policy_rollout_single_pass, policy_rollout_streaming
from kipl_ml.trace.enums import Feats

dotenv.load_dotenv()

logger = get_logger(__name__)


def _load_model(model: str) -> nn.Module:
    model = mlflow.pytorch.load_model(f"models:/{model}", map_location="cpu")
    model.eval()
    return model


class _NNDef(_Def):
    def __init__(
        self,
        network_delay_millis: tuple[int, int],
        network_pps: tuple[int, int],
        obs_model: Sequence[str] | nn.Module | str,
        seed: int | None = 42,
        fixed_per_trace: bool = False,
        simul_kwargs: dict | None = None,
        state_dicts: Sequence[nn.Module.state_dict] | None = None,
        mlflow_keys: dict[str, Any] | None = None,
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

        self.mlflow_keys = mlflow_keys
        self._model_ids: list[str | None] = [None]
        self.defence_model_state_dicts = None

        if isinstance(obs_model, str):
            self._model_ids = [obs_model]
            self.defense_model = _load_model(obs_model)
        elif isinstance(obs_model, nn.Module):
            self.defense_model = obs_model
            if state_dicts is not None:
                self.defence_model_state_dicts = state_dicts
        elif isinstance(obs_model, list):
            self.defense_model = _load_model(obs_model[0])
            self.defence_model_state_dicts = [
                _load_model(obs).state_dict for obs in obs_model
            ]
            self._model_ids = obs_model

        self.defense_model.eval()

        self.simul_kwargs = simul_kwargs or {}

    def report(self, to_log: bool = False) -> str:
        str_ = self.__class__.__name__ + "\n"
        str_ += f"\t{self.network_delay_millis}\n"
        str_ += f"\t{self.network_pps}\n"
        str_ += f"\tFixed per trace: {self.FIXED_PER_TRACE}\n"

        dm = self.defense_model
        str_ += f"\t\t{dm.__class__.__name__}\n"
        str_ += f"\t\t\tTime step: {dm.time_step}\n"
        str_ += f"\t\t\tMax silence: {dm.max_silence_s}\n"
        str_ += "\t\t\tModel ids\n"
        for id_ in self._model_ids:
            str_ += f"\t\t\t\t{id_}\n"

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
        d["model-id"] = str(self._model_ids)
        if self.mlflow_keys is not None:
            d.update(self.mlflow_keys)

        return d


class RNNDef(_NNDef):
    def __init__(self, *args, n_packets: int = 5000, **kwargs):
        super().__init__(*args, **kwargs)
        self._n_packets = n_packets

    def _run_model(
        self, trace_d: dict[Feats, torch.Tensor]
    ) -> dict[Feats, torch.Tensor]:
        # Implement RNN specific logic
        h = None

        trace_d = {
            k: v[: self._n_packets].unsqueeze(0).float() for k, v in trace_d.items()
        }

        if self.defence_model_state_dicts is not None:
            st_d = random.choice(self.defence_model_state_dicts)
            self.defense_model.load_state_dict(st_d)

        defense_model = self.defense_model

        if getattr(defense_model, "enable_delay", False):
            _, _, _, _, _, _, _, trace_d = policy_rollout_streaming(
                defense_model,
                trace_d,
                sample=True,
                extend_end_s=0,
                max_packets=self._n_packets,
            )

            trace_d = {k: v.squeeze(0) for k, v in trace_d.items()}
            trace_d[Feats.SIZES] = torch.ones_like(trace_d[Feats.TIMES])
            return trace_d

        # Non-delay policy can be run in one pass.
        _, _, _, _, _, _, _, trace_d = policy_rollout_single_pass(
            defense_model,
            trace_d,
            detach_period=100,
            sample=True,
            extend_end_s=0,
        )
        trace_d = {k: v[:, : self._n_packets] for k, v in trace_d.items()}

        trace_d = {k: v.squeeze(0) for k, v in trace_d.items()}

        trace_d[Feats.SIZES] = torch.ones_like(trace_d[Feats.TIMES])

        return trace_d
