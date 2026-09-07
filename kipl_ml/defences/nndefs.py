from __future__ import annotations

import math
import os
from collections.abc import Mapping, Sequence
from typing import Any, cast

import dotenv
import mlflow
import numpy as np
import torch
from torch import nn

from kipl_ml.defences.base import DEFENCE_TYPE_KW, _Def
from kipl_ml.defences.models.brick_selection_agent import BrickSelectionAgent
from kipl_ml.logging.logger import get_logger
from kipl_ml.logging.utils import log_multiline
from kipl_ml.network.network import NetworkContextIntDict
from kipl_ml.rl.brick_simulate import (
    BrickSpecCollection,
    brick_policy_rollout,
)
from kipl_ml.rl.simulate import policy_obfuscate_trace
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
        obs_model: Sequence[str] | nn.Module | str,
        seed: int | None = 42,
        fixed_per_trace: bool = False,
        simul_kwargs: dict | None = None,
        state_dicts: Sequence[Mapping[str, torch.Tensor]] | None = None,
        mlflow_keys: dict[str, Any] | None = None,
    ):
        if fixed_per_trace:
            logger.warning(
                "NNdefs do not use fixed_per_trace parameter -> param. ignored."
            )

        super().__init__(seed=seed, fixed_per_trace=fixed_per_trace)

        self.mlflow_keys = mlflow_keys
        self._model_ids: Sequence[str | None] = [None]
        self.defence_model_state_dicts: list[Mapping[str, torch.Tensor]] | None = None

        if isinstance(obs_model, str):
            self._model_ids = [obs_model]
            self.defense_model = _load_model(obs_model)
        elif isinstance(obs_model, nn.Module):
            self.defense_model = obs_model
            if state_dicts is not None:
                self.defence_model_state_dicts = list(state_dicts)
        elif isinstance(obs_model, list):
            self.defense_model = _load_model(obs_model[0])
            self.defence_model_state_dicts = [
                _load_model(obs).state_dict() for obs in obs_model
            ]
            self._model_ids = [
                obs if isinstance(obs, str) else None for obs in obs_model
            ]

        self.defense_model.eval()

        self.simul_kwargs = simul_kwargs or {}

    def _load_random_state_dict(self, rng: np.random.Generator) -> None:
        if self.defence_model_state_dicts is None:
            return
        idx = int(rng.integers(len(self.defence_model_state_dicts)))
        st_d = self.defence_model_state_dicts[idx]
        self.defense_model.load_state_dict(st_d)

    def report(self, to_log: bool = False) -> str:
        str_ = self.__class__.__name__ + "\n"
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
        self,
        trace_path: os.PathLike,
        network_context: NetworkContextIntDict | None,
    ) -> dict[Feats, torch.Tensor]:
        raise NotImplementedError

    def _simulate(
        self,
        trace_path: os.PathLike,
        machine_idx: int | None = None,
        trim_raw: int = 0,
        network_context: NetworkContextIntDict | None = None,
    ) -> dict[Feats, torch.Tensor]:
        if machine_idx is not None:
            raise NotImplementedError(
                f"{self.__class__.__name__} does not support machine_idx argument."
            )

        trace_d = self._run_model(trace_path, network_context)

        return trace_d

    def _mlflow_log_params(self) -> dict[str, str]:
        d = {}
        d[DEFENCE_TYPE_KW] = self.__class__.__name__.lower()
        d["model-id"] = str(self._model_ids)
        if self.mlflow_keys is not None:
            d.update(self.mlflow_keys)

        return d


class RNNDef(_NNDef):
    def __init__(
        self,
        *args,
        n_packets: int | None = 5000,
        max_dur_s: float | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._n_packets = n_packets
        self._max_dur_s = max_dur_s or math.inf
        self.rng = np.random.default_rng()

    def _run_model(
        self,
        trace_path: os.PathLike,
        network_context: NetworkContextIntDict | None,
    ) -> dict[Feats, torch.Tensor]:
        # Implement RNN specific logic

        self._load_random_state_dict(self.rng)

        add_tail_s = 0.0

        if network_context is None:
            raise ValueError(
                f"{self.__class__.__name__} requires explicit network_context"
            )

        with torch.inference_mode():
            trace_d = cast(
                dict[Feats, torch.Tensor],
                policy_obfuscate_trace(
                    self.defense_model,
                    [str(trace_path)],
                    device=cast(torch.DeviceObjType, torch.device("cpu")),
                    sample=True,
                    max_packets=self._n_packets,
                    max_duration_s=self._max_dur_s,
                    add_tail_s=add_tail_s,
                    network_context=network_context,
                    seed=0 if self.seed is None else self.seed,
                ),
            )

        trace_d = {k: v.squeeze(0) for k, v in trace_d.items()}
        trace_d[Feats.SIZES] = torch.ones_like(trace_d[Feats.TIMES])

        return trace_d


class BrickSelectionDef(_NNDef):
    def __init__(
        self,
        *args,
        client_bricks: BrickSpecCollection,
        server_bricks: BrickSpecCollection,
        n_packets: int | None = 5000,
        max_dur_s: float | None = None,
        sample: bool = True,
        relative: bool = True,
        max_steps: int = 10_000,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if not isinstance(self.defense_model, BrickSelectionAgent):
            raise TypeError(
                f"BrickSelectionDef requires BrickSelectionAgent, "
                f"got {type(self.defense_model)}"
            )
        if max_steps <= 0:
            raise ValueError("max_steps must be > 0")

        self.client_bricks = list(client_bricks)
        self.server_bricks = list(server_bricks)
        self._n_packets = n_packets
        self._max_dur_s = max_dur_s or math.inf
        self.sample = sample
        self.relative = relative
        self.max_steps = int(max_steps)
        self.rng = np.random.default_rng()

    def report(self, to_log: bool = False) -> str:
        str_ = self.__class__.__name__ + "\n"
        str_ += f"\tFixed per trace: {self.FIXED_PER_TRACE}\n"

        dm = self.defense_model
        str_ += f"\t\t{dm.__class__.__name__}\n"
        str_ += f"\t\t\tTime step: {dm.time_step}\n"
        str_ += f"\t\t\tClient bricks: {dm.n_client_bricks}\n"
        str_ += f"\t\t\tServer bricks: {dm.n_server_bricks}\n"
        str_ += f"\t\t\tMax steps: {self.max_steps}\n"
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

    def _simulate(
        self,
        trace_path: os.PathLike,
        machine_idx: int | None = None,
        trim_raw: int = 0,
        network_context: NetworkContextIntDict | None = None,
    ) -> dict[Feats, torch.Tensor]:
        if machine_idx is not None:
            raise NotImplementedError(
                f"{self.__class__.__name__} does not support machine_idx argument."
            )
        if self._n_packets is None:
            raise ValueError("BrickSelectionDef requires explicit n_packets")
        if network_context is None:
            raise ValueError(
                f"{self.__class__.__name__} requires explicit network_context"
            )

        self._load_random_state_dict(self.rng)
        raw_trace = self.load_data(
            trace_path,
            trim_raw=trim_raw,
            network_context=network_context,
        )
        required_real_packets = int((raw_trace[Feats.DIRS] != 0).sum().item())
        policy = cast(BrickSelectionAgent, self.defense_model)

        with torch.inference_mode():
            *_, trace_d = brick_policy_rollout(
                policy=policy,
                trace_paths=[str(trace_path)],
                device=torch.device("cpu"),
                client_bricks=self.client_bricks,
                server_bricks=self.server_bricks,
                network_context=network_context,
                max_packets=self._n_packets,
                max_duration_s=self._max_dur_s,
                required_real_packets=required_real_packets,
                trim_raw=trim_raw,
                seed=0 if self.seed is None else self.seed,
                sample=self.sample,
                relative=self.relative,
                max_steps=self.max_steps,
            )

        trace_d = {k: v.squeeze(0) for k, v in trace_d.items()}
        trace_d[Feats.SIZES] = torch.ones_like(trace_d[Feats.TIMES])
        return trace_d
