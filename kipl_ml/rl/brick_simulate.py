from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Protocol, runtime_checkable

import numpy as np
import torch

from kipl_ml.data.utils import DOWNLOAD, UPLOAD
from kipl_ml.data.wf_dataset import dict_to_device
from kipl_ml.network.network import NetworkContext
from kipl_ml.rl.enums import EntropyKeys, StepAction, StepActions
from kipl_ml.rl.lego import LegoBrickSpec
from kipl_ml.rl.simulate import (
    BrickBatchController,
    NumpyTrace,
    _batch_packet_level_features,
)
from kipl_ml.trace.enums import Feats


@runtime_checkable
class BrickPolicy(Protocol):
    time_step: float

    @property
    def features(self) -> Sequence[Feats]: ...

    def act_step(
        self,
        x: dict[Feats, torch.Tensor],
        current_client_bricks: torch.Tensor,
        current_server_bricks: torch.Tensor,
        sample: bool = True,
    ) -> tuple[
        torch.Tensor,
        StepActions,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        dict[str, torch.Tensor],
        None,
    ]: ...


class BrickController(Protocol):
    batch_size: int

    def current_client_bricks(
        self, active_mask: np.ndarray | None = None
    ) -> np.ndarray: ...

    def current_server_bricks(
        self, active_mask: np.ndarray | None = None
    ) -> np.ndarray: ...

    def step_current(self, active_mask: np.ndarray) -> list[NumpyTrace]: ...

    def select(
        self, step_actions: list[StepAction], active_mask: np.ndarray
    ) -> None: ...

    def is_done(self) -> list[bool]: ...


BrickRollout = tuple[
    dict[Feats, torch.Tensor],
    torch.Tensor,
    list[StepActions],
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    dict[str, torch.Tensor],
    dict[Feats, torch.Tensor],
]

BrickSpecCollection = Sequence[LegoBrickSpec]
STANDARD_BRICK_ROLLOUT_FEATURES = (
    Feats.TIME_BINS,
    Feats.Dt_BINS,
    Feats.UP_COUNT,
    Feats.DOWN_COUNT,
)


def brick_policy_rollout(  # noqa: C901
    policy: BrickPolicy,
    trace_paths: Sequence[str],
    device: Any,
    *,
    client_bricks: BrickSpecCollection | None,
    server_bricks: BrickSpecCollection | None,
    network_context: Mapping[str, object] | None,
    max_packets: int,
    max_duration_s: float,
    required_real_packets: int | torch.Tensor | None,
    trim_raw: int,
    seed: int,
    sample: bool,
    relative: bool,
    max_steps: int,
    controller: BrickController | None = None,
) -> BrickRollout:
    if max_steps <= 0:
        raise ValueError("max_steps must be > 0")
    if Feats.TIME_BINS not in policy.features or Feats.Dt_BINS not in policy.features:
        raise ValueError("brick rollout requires TIME_BINS and Dt_BINS policy features")

    if controller is None:
        if client_bricks is None or server_bricks is None:
            raise ValueError("client_bricks and server_bricks are required")
        if network_context is None:
            raise ValueError("network_context is required")
        network_kwargs = NetworkContext.to_rust_args_batch(dict(network_context))
        controller = BrickBatchController.new(
            trace_paths=trace_paths,
            window_duration_ns=int(round(float(policy.time_step) * 1e9)),
            client_bricks=client_bricks,
            server_bricks=server_bricks,
            network_kwargs=network_kwargs,
            max_trace_length=max_packets,
            seed=seed,
            trim_raw=trim_raw,
            relative=relative,
        )

    bs = controller.batch_size
    if len(trace_paths) != bs:
        raise ValueError("trace_paths length must equal controller.batch_size")

    required_real_packets_np = _required_real_packets(required_real_packets, bs)
    terminated = np.zeros(bs, dtype=bool)
    next_bins = np.zeros(bs, dtype=np.int64)
    n_packets = np.zeros(bs, dtype=np.int64)
    n_real_packets = np.zeros(bs, dtype=np.int64)
    X_obs_l: list[dict[Feats, list[torch.Tensor]]] = [
        {Feats.TIMES: [], Feats.DIRS: [], Feats.DECOY: []} for _ in range(bs)
    ]
    fd_features = tuple(
        dict.fromkeys((*STANDARD_BRICK_ROLLOUT_FEATURES, *policy.features))
    )
    fd_steps: dict[Feats, list[torch.Tensor]] = {f: [] for f in fd_features}
    actions_l: list[StepActions] = [[] for _ in range(bs)]
    act_time_bins_l: list[torch.Tensor] = []
    log_ps_l: list[torch.Tensor] = []
    sel_probs_l: list[torch.Tensor] = []
    values_actor_l: list[torch.Tensor] = []
    ent_sel_l: list[torch.Tensor] = []
    ent_cond_l: list[torch.Tensor] = []

    for _step in range(max_steps):
        simulator_done = np.asarray(controller.is_done(), dtype=bool)
        active = ~terminated & ~simulator_done
        if not active.any():
            break

        windows = controller.step_current(active)
        simulator_done = np.asarray(controller.is_done(), dtype=bool)
        active_for_policy = np.zeros(bs, dtype=bool)
        fd_w: dict[Feats, list[float]] = {f: [] for f in fd_features}

        for idx, window in enumerate(windows):
            if not active[idx]:
                continue
            times = torch.tensor(window[0] / 1e9).float()
            dirs = torch.tensor(window[1]).float()
            decoys = torch.tensor(window[2]).float()
            if times.numel() > 0:
                X_obs_l[idx][Feats.TIMES].append(times)
                X_obs_l[idx][Feats.DIRS].append(dirs)
                X_obs_l[idx][Feats.DECOY].append(decoys)

            up_count = float(((dirs == UPLOAD) & (decoys == 0)).sum().item())
            down_count = float(((dirs == DOWNLOAD) & (decoys == 0)).sum().item())
            n_packets[idx] += times.numel()
            n_real_packets[idx] += int(up_count + down_count)

            current_bin = int(next_bins[idx])
            next_bins[idx] += 1
            long_enough = n_packets[idx] >= max_packets or (
                next_bins[idx] * float(policy.time_step) > max_duration_s
            )
            reached_required_real = _reached_required_real(
                n_real_packets,
                required_real_packets_np,
            )
            if simulator_done[idx] or (long_enough and reached_required_real[idx]):
                terminated[idx] = True
                continue

            active_for_policy[idx] = True
            for feature in fd_features:
                fd_w[feature].append(
                    _feature_value(feature, current_bin, 1, up_count, down_count)
                )

        if not active_for_policy.any():
            continue

        fd_w_tensor_all = {
            k: torch.tensor(v).reshape(-1, 1).float() for k, v in fd_w.items()
        }
        fd_w_tensor_all = dict_to_device(fd_w_tensor_all, device=device)
        fd_w_tensor = {f: fd_w_tensor_all[f] for f in policy.features}
        current_client = torch.as_tensor(
            controller.current_client_bricks(active_for_policy),
            device=device,
        )
        current_server = torch.as_tensor(
            controller.current_server_bricks(active_for_policy),
            device=device,
        )
        (
            act_time_bins_a,
            actions_a,
            log_ps_a,
            sel_probs_a,
            values_a,
            ent_a,
            _h,
        ) = policy.act_step(
            fd_w_tensor,
            current_client,
            current_server,
            sample=sample,
        )

        active_t = torch.as_tensor(active_for_policy, device=act_time_bins_a.device)
        for feature in fd_features:
            fd_steps[feature].append(
                _densify(fd_w_tensor_all[feature], active_t, bs).cpu()
            )
        act_time_bins_l.append(_densify(act_time_bins_a, active_t, bs).detach().cpu())
        log_ps_l.append(_densify(log_ps_a, active_t, bs))
        sel_probs_l.append(_densify(sel_probs_a, active_t, bs))
        values_actor_l.append(_densify(values_a, active_t, bs))
        ent_sel_l.append(_densify(ent_a[EntropyKeys.SELECTION_ENTROPY], active_t, bs))
        ent_cond_l.append(_densify(ent_a[EntropyKeys.COND_ENTROPY], active_t, bs))
        for local_idx, batch_idx in enumerate(np.nonzero(active_for_policy)[0]):
            actions_l[int(batch_idx)].append(actions_a[local_idx])
        controller.select(actions_a, active_for_policy)
    else:
        raise RuntimeError(f"brick rollout reached max_steps={max_steps}")

    X_obs = _batch_packet_level_features(X_obs_l, device)
    if not act_time_bins_l:
        fd = {f: torch.zeros((bs, 1), device=device) for f in fd_features}
        fd[Feats.TIME_BINS] = torch.full((bs, 1), -1.0, device=device)
        fd[Feats.SEQ_LENS] = torch.zeros(bs, dtype=torch.long, device=device)
        zeros = torch.zeros((bs, 1), device=device)
        return (
            fd,
            torch.full((bs, 1), -1, dtype=torch.long, device=device),
            actions_l,
            zeros,
            torch.zeros((bs, 1, 1), device=device),
            zeros,
            {
                "selection_entropy": zeros,
                "conditional_entropy": zeros,
            },
            X_obs,
        )

    act_time_bins = torch.cat(act_time_bins_l, dim=1)
    act_time_bins[act_time_bins == 0] = -1
    fd = {f: torch.cat(vs, dim=1) for f, vs in fd_steps.items()}
    fd[Feats.TIME_BINS] = fd[Feats.TIME_BINS].masked_fill(act_time_bins < 0, -1)
    fd[Feats.SEQ_LENS] = (act_time_bins > 0).sum(dim=1).long()
    fd = dict_to_device(fd, device=device)

    return (
        fd,
        act_time_bins.to(device),
        actions_l,
        torch.cat(log_ps_l, dim=1),
        torch.cat(sel_probs_l, dim=1),
        torch.cat(values_actor_l, dim=1),
        {
            "selection_entropy": torch.cat(ent_sel_l, dim=1),
            "conditional_entropy": torch.cat(ent_cond_l, dim=1),
        },
        X_obs,
    )


def _required_real_packets(
    required_real_packets: int | torch.Tensor | None, bs: int
) -> np.ndarray | None:
    if required_real_packets is None:
        return None
    if isinstance(required_real_packets, int):
        return np.full(bs, required_real_packets, dtype=np.int64)
    if required_real_packets.shape != (bs,):
        raise ValueError(
            f"required_real_packets must have shape ({bs},), "
            f"got {tuple(required_real_packets.shape)}"
        )
    return required_real_packets.detach().cpu().numpy().astype(np.int64, copy=False)


def _reached_required_real(
    n_real_packets: np.ndarray, required_real_packets: np.ndarray | None
) -> np.ndarray:
    if required_real_packets is None:
        return np.ones_like(n_real_packets, dtype=bool)
    return n_real_packets >= required_real_packets


def _feature_value(
    feature: Feats,
    current_bin: int,
    n_steps: int,
    up_count: float,
    down_count: float,
) -> float:
    if feature == Feats.TIME_BINS:
        return float(current_bin)
    if feature == Feats.Dt_BINS:
        return float(n_steps)
    if feature == Feats.UP_COUNT:
        return up_count
    if feature == Feats.DOWN_COUNT:
        return down_count
    if feature == Feats.SILENCE_FLAG:
        return float(up_count == 0.0 and down_count == 0.0)
    raise ValueError(f"unsupported brick policy feature: {feature}")


def _densify(x: torch.Tensor, active: torch.Tensor, bs: int) -> torch.Tensor:
    full = torch.zeros((bs,) + x.shape[1:], device=x.device, dtype=x.dtype)
    full[active] = x
    return full
