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
from kipl_ml.rl.action import TraceExecState, send_exec
from kipl_ml.rl.enums import Actions
from kipl_ml.rl.observation import WindowFeatureStreamer, get_window_feature_dict
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

        # If delay is enabled on the policy, future observation windows depend on
        # the selected actions. Use a stepwise rollout to ensure windows reflect
        # delays (single-pass precomputation is invalid).
        if getattr(defense_model, "enable_delay", False):
            h = None
            base_n = int((trace_d[Feats.DIRS] != 0).sum().item())
            pad_n = 0
            streamer = WindowFeatureStreamer(
                trace_d,
                dt=float(defense_model.time_step),
                max_silence_s=float(defense_model.max_silence_s),
                features=list(defense_model.features),
                extend_end_s=0,
            )

            X_base = {
                Feats.TIMES: trace_d[Feats.TIMES].clone(),
                Feats.DIRS: trace_d[Feats.DIRS].clone(),
                Feats.PADDING: trace_d.get(Feats.PADDING, torch.zeros_like(trace_d[Feats.TIMES])).clone(),
            }
            exec_state = TraceExecState(X_base)

            defense_model.eval()
            with torch.no_grad():
                while True:
                    fd_t = streamer.step()
                    if not fd_t[Feats.TIMES].isfinite().any():
                        break

                    if hasattr(defense_model, "act_step"):
                        act_times, actions, *_rest, h = defense_model.act_step(
                            fd_t, h, sample=True
                        )
                    else:
                        act_times, actions, *_rest, h = defense_model.act(
                            fd_t,
                            h,
                            h_detach_period=None,
                            seq_lens=torch.ones((1,), device=fd_t[Feats.TIMES].device).long(),
                            sample=True,
                        )

                    exec_state.step(
                        trace_idx=torch.zeros((1,), device=act_times.device, dtype=torch.long),
                        times=act_times,
                        actions=actions,
                    )

                    # Stop once we've produced enough packets.
                    pad_n += int(
                        actions[Actions.SEND_COUNT_UP].item()
                        + actions[Actions.SEND_COUNT_DOWN].item()
                    )
                    if base_n + pad_n >= self._n_packets:
                        break

                    if Actions.DELAY in actions and (actions[Actions.DELAY] > 0).any():
                        streamer.apply_delay(actions[Actions.DELAY])

            trace_d = exec_state.finalize()
            # Cap output length.
            trace_d = {k: v[:, : self._n_packets] for k, v in trace_d.items()}
            trace_d = {k: v.squeeze(0) for k, v in trace_d.items()}
            trace_d[Feats.SIZES] = torch.ones_like(trace_d[Feats.TIMES])
            return trace_d

        fd = get_window_feature_dict(
            trace_d,
            defense_model.time_step,
            defense_model.max_silence_s,
            features=defense_model.features,
        )

        seq_lens = fd.pop(Feats.SEQ_LENS)

        defense_model.eval()
        with torch.no_grad():
            act_times, actions = defense_model.act(
                fd, h, h_detach_period=100, seq_lens=seq_lens
            )[:2]

        trace_d = send_exec(trace_d, act_times, actions)

        # Cap output length (disc features often use n_packets=None).
        trace_d = {k: v[:, : self._n_packets] for k, v in trace_d.items()}

        trace_d = {k: v.squeeze(0) for k, v in trace_d.items()}

        trace_d[Feats.SIZES] = torch.ones_like(trace_d[Feats.TIMES])

        return trace_d
