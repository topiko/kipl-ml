from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from typing import Any, Protocol, runtime_checkable

import numpy as np
import torch

from kipl_ml.data.utils import DOWNLOAD, UPLOAD
from kipl_ml.data.wf_dataset import dict_to_device
from kipl_ml.network.network import NetworkContext
from kipl_ml.rl.enums import EntropyKeys, StepAction, StepActions
from kipl_ml.rl.lego import LegoBrickSpec
from kipl_ml.rl.rollout_end import RolloutEnd
from kipl_ml.rl.simulate import (
    BrickBatchController,
    NumpyTrace,
    _batch_packet_level_features,
)
from kipl_ml.trace.enums import Feats


@runtime_checkable
class BrickPolicy(Protocol):
    time_step_s: float

    @property
    def features(self) -> Sequence[Feats]: ...

    def act_step(
        self,
        x: dict[Feats, torch.Tensor],
        current_client_bricks: torch.Tensor,
        current_server_bricks: torch.Tensor,
        sample: bool = True,
        state: torch.Tensor | None = None,
    ) -> tuple[
        torch.Tensor,
        StepActions,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        dict[str, torch.Tensor],
        torch.Tensor | None,
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

    def stop_reasons(self) -> list[str | None]: ...


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
    Feats.UP_DECOY_COUNT,
    Feats.DOWN_DECOY_COUNT,
    Feats.SILENCE_FLAG,
)
DEFAULT_BRICK_MAX_SIM_STEPS = 1_000_000


def brick_policy_rollout(  # noqa: C901
    policy: BrickPolicy,
    trace_paths: Sequence[str],
    device: Any,
    *,
    client_bricks: BrickSpecCollection | None,
    server_bricks: BrickSpecCollection | None,
    network_context: Mapping[str, object] | None,
    max_packets: int | None,
    max_duration_s: float,
    trim_raw: int,
    seed: int,
    sample: bool,
    relative: bool,
    max_steps: int,
    controller: BrickController | None = None,
    endpoint: RolloutEnd | None = None,
    max_sim_steps: int = DEFAULT_BRICK_MAX_SIM_STEPS,
    detach_period: int | None = None,
) -> BrickRollout:
    """Shared replay/actor horizon; packet limits apply after complete windows."""
    if max_steps <= 0:
        raise ValueError("max_steps must be > 0")
    if max_sim_steps <= 0:
        raise ValueError("max_sim_steps must be > 0")
    if max_packets is not None and max_packets <= 0:
        raise ValueError("max_packets must be positive or None")
    if max_duration_s <= 0:
        raise ValueError("max_duration_s must be > 0")
    if controller is None:
        if client_bricks is None or server_bricks is None:
            raise ValueError("client_bricks and server_bricks are required")
        if network_context is None:
            raise ValueError("network_context is required")
        network_kwargs = NetworkContext.to_rust_args_batch(dict(network_context))
        controller = BrickBatchController.new(
            trace_paths=trace_paths,
            window_duration_ns=int(round(float(policy.time_step_s) * 1e9)),
            client_bricks=client_bricks,
            server_bricks=server_bricks,
            network_kwargs=network_kwargs,
            max_trace_length=0,
            seed=seed,
            trim_raw=trim_raw,
            relative=relative,
            stop_after_all_normal_packets=True,
            max_sim_steps=max_sim_steps,
        )

    bs = controller.batch_size
    if len(trace_paths) != bs:
        raise ValueError("trace_paths length must equal controller.batch_size")

    terminated = np.zeros(bs, dtype=bool)
    next_bins = np.zeros(bs, dtype=np.int64)
    n_packets = np.zeros(bs, dtype=np.int64)
    packet_cutoffs = 0
    X_obs_l: list[dict[Feats, list[torch.Tensor]]] = [
        {Feats.TIMES: [], Feats.DIRS: [], Feats.DECOY: []} for _ in range(bs)
    ]
    fd_features = tuple(
        dict.fromkeys((*STANDARD_BRICK_ROLLOUT_FEATURES, *policy.features))
    )
    fd_steps: dict[Feats, list[torch.Tensor]] = {f: [] for f in fd_features}
    brick_state_steps: dict[Feats, list[torch.Tensor]] = {
        Feats.CURRENT_CLIENT_BRICK: [],
        Feats.CURRENT_SERVER_BRICK: [],
    }
    actions_l: list[StepActions] = [[] for _ in range(bs)]
    act_time_bins_l: list[torch.Tensor] = []
    log_ps_l: list[torch.Tensor] = []
    sel_probs_l: list[torch.Tensor] = []
    values_actor_l: list[torch.Tensor] = []
    entropy_keys = {
        "selection_entropy": EntropyKeys.SELECTION_ENTROPY,
        "conditional_entropy": EntropyKeys.COND_ENTROPY,
        "client_entropy": EntropyKeys.CLIENT_ENTROPY,
        "server_entropy": EntropyKeys.SERVER_ENTROPY,
    }
    entropy_steps: dict[str, list[torch.Tensor]] = {key: [] for key in entropy_keys}
    recurrent = getattr(policy, "is_recurrent", False)
    policy_state: torch.Tensor | None = None

    def apply_policy(
        fd_w_tensor_all: dict[Feats, torch.Tensor],
        active_for_policy: np.ndarray,
    ) -> None:
        nonlocal policy_state
        fd_w_tensor_all = dict_to_device(fd_w_tensor_all, device=device)
        policy_input_features = tuple(
            dict.fromkeys((Feats.TIME_BINS, Feats.Dt_BINS, *policy.features))
        )
        fd_w_tensor = {f: fd_w_tensor_all[f] for f in policy_input_features}
        current_client = torch.as_tensor(
            controller.current_client_bricks(active_for_policy),
            device=device,
        )
        current_server = torch.as_tensor(
            controller.current_server_bricks(active_for_policy),
            device=device,
        )
        active_t = torch.as_tensor(active_for_policy, device=device)
        if (policy_state is not None and detach_period and detach_period > 0
                and len(act_time_bins_l) % detach_period == 0):
            policy_state = policy_state.detach()
        state_kwargs = (
            {"state": None if policy_state is None else policy_state[active_t]}
            if recurrent else {}
        )
        (
            act_time_bins_a,
            actions_a,
            log_ps_a,
            sel_probs_a,
            values_a,
            ent_a,
            next_state,
        ) = policy.act_step(
            fd_w_tensor,
            current_client,
            current_server,
            sample=sample,
            **state_kwargs,
        )

        if next_state is not None:
            # Out-of-place scatter preserves autograd and original trace indices
            # as the active batch shrinks. Finished histories are no longer needed.
            policy_state = _densify(next_state, active_t, bs)

        active_t = torch.as_tensor(active_for_policy, device=act_time_bins_a.device)
        # V(s_t) must see the brick active before sampling a_t, not the newly
        # selected brick. Record the same current state passed to the actor.
        for feature, current in (
            (Feats.CURRENT_CLIENT_BRICK, current_client),
            (Feats.CURRENT_SERVER_BRICK, current_server),
        ):
            brick_state_steps[feature].append(
                _densify(current.unsqueeze(1), active_t, bs, fill_value=-1)
                .detach()
                .cpu()
            )
        for feature in fd_features:
            fd_steps[feature].append(
                _densify(fd_w_tensor_all[feature], active_t, bs).cpu()
            )
        act_time_bins_l.append(
            _densify(act_time_bins_a, active_t, bs, fill_value=-1).detach().cpu()
        )
        log_ps_l.append(_densify(log_ps_a, active_t, bs))
        sel_probs_l.append(_densify(sel_probs_a, active_t, bs))
        values_actor_l.append(_densify(values_a, active_t, bs))
        for name, key in entropy_keys.items():
            entropy_steps[name].append(_densify(ent_a[key], active_t, bs))
        for local_idx, batch_idx in enumerate(np.nonzero(active_for_policy)[0]):
            actions_l[int(batch_idx)].append(actions_a[local_idx])
        controller.select(actions_a, active_for_policy)

    initial_active = ~np.asarray(controller.is_done(), dtype=bool)
    if initial_active.any():
        initial_values = {
            feature: [
                _feature_value(feature, 0, 0, 0.0, 0.0, 0.0, 0.0)
            ]
            * int(initial_active.sum())
            for feature in fd_features
        }
        apply_policy(_feature_tensors(initial_values), initial_active)

    for _step in range(max_steps):
        simulator_done = np.asarray(controller.is_done(), dtype=bool)
        active = ~terminated & ~simulator_done
        if not active.any():
            break

        windows = controller.step_current(active)
        simulator_done = np.asarray(controller.is_done(), dtype=bool)
        reasons = controller.stop_reasons()
        if any(active[idx] and reason == "max_sim_steps"
               for idx, reason in enumerate(reasons)):
            raise RuntimeError("brick simulator iteration budget exhausted")
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

            # TODO (deferred): Share window-statistics primitives with TAM and
            # simulate.py through an online observation builder. Keep simulator
            # state/window handling in this adapter; preserve normal/decoy separation,
            # completed-window timing, initial observations, and pre-action brick IDs.
            up_count = float(((dirs == UPLOAD) & (decoys == 0)).sum().item())
            down_count = float(((dirs == DOWNLOAD) & (decoys == 0)).sum().item())
            up_decoy_count = float(((dirs == UPLOAD) & (decoys != 0)).sum().item())
            down_decoy_count = float(
                ((dirs == DOWNLOAD) & (decoys != 0)).sum().item()
            )
            n_packets[idx] += times.numel()

            current_bin = int(next_bins[idx])
            next_bins[idx] += 1
            cutoff = (
                "max_duration" if next_bins[idx] * float(policy.time_step_s)
                >= max_duration_s else
                "max_packets" if max_packets is not None
                and n_packets[idx] >= max_packets else None
            )
            if cutoff == "max_packets" and not simulator_done[idx]:
                packet_cutoffs += 1
            if endpoint is not None:
                end_features = _feature_tensors({
                    feature: [_feature_value(
                        feature, current_bin, 1, up_count, down_count,
                        up_decoy_count, down_decoy_count,
                    )] for feature in fd_features
                })
                end_features[Feats.CURRENT_CLIENT_BRICK] = torch.tensor(
                    [[controller.current_client_bricks()[idx]]],
                )
                end_features[Feats.CURRENT_SERVER_BRICK] = torch.tensor(
                    [[controller.current_server_bricks()[idx]]],
                )
                if endpoint.record(
                    idx, end_features, reasons[idx], cutoff, float(policy.time_step_s),
                ):
                    terminated[idx] = True
                    if endpoint.truncated[idx]:
                        end_features = dict_to_device(end_features, device=device)
                        with torch.no_grad():
                            result = policy.act_step(
                                end_features,
                                end_features[Feats.CURRENT_CLIENT_BRICK].flatten(),
                                end_features[Feats.CURRENT_SERVER_BRICK].flatten(),
                                sample=False,
                                **({"state": policy_state[idx:idx + 1]}
                                   if policy_state is not None else {}),
                            )
                        endpoint.values[idx] = result[4].reshape(())
                    continue
            if endpoint is None and (
                simulator_done[idx] or cutoff is not None
            ):
                terminated[idx] = True
                continue

            active_for_policy[idx] = True
            for feature in fd_features:
                fd_w[feature].append(
                    _feature_value(
                        feature,
                        current_bin,
                        1,
                        up_count,
                        down_count,
                        up_decoy_count,
                        down_decoy_count,
                    )
                )

        if not active_for_policy.any():
            break

        apply_policy(_feature_tensors(fd_w), active_for_policy)
    else:
        raise RuntimeError(f"brick rollout reached max_steps={max_steps}")

    if packet_cutoffs:
        warnings.warn(
            f"Brick packet safety cutoff max_packets={max_packets} ended "
            f"{packet_cutoffs}/{bs} traces before normal completion or the time "
            "horizon. Set trace.n_packets=null to disable this cutoff.",
            RuntimeWarning, stacklevel=2,
        )

    X_obs = _batch_packet_level_features(X_obs_l, device)
    if not act_time_bins_l:
        fd = {f: torch.zeros((bs, 1), device=device) for f in fd_features}
        fd.update(
            {
                feature: torch.full((bs, 1), -1, dtype=torch.long, device=device)
                for feature in brick_state_steps
            }
        )
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
            dict.fromkeys(entropy_keys, zeros),
            X_obs,
        )

    act_time_bins = torch.cat(act_time_bins_l, dim=1)
    fd = {f: torch.cat(vs, dim=1) for f, vs in fd_steps.items()}
    fd.update({f: torch.cat(vs, dim=1) for f, vs in brick_state_steps.items()})
    fd[Feats.TIME_BINS] = fd[Feats.TIME_BINS].masked_fill(act_time_bins < 0, -1)
    fd[Feats.SEQ_LENS] = (act_time_bins >= 0).sum(dim=1).long()
    fd = dict_to_device(fd, device=device)

    return (
        fd,
        act_time_bins.to(device),
        actions_l,
        torch.cat(log_ps_l, dim=1),
        torch.cat(sel_probs_l, dim=1),
        torch.cat(values_actor_l, dim=1),
        {key: torch.cat(steps, dim=1) for key, steps in entropy_steps.items()},
        X_obs,
    )


def _feature_value(
    feature: Feats,
    current_bin: int,
    n_steps: int,
    up_count: float,
    down_count: float,
    up_decoy_count: float,
    down_decoy_count: float,
) -> float:
    if feature == Feats.TIME_BINS:
        return float(current_bin)
    if feature == Feats.Dt_BINS:
        return float(n_steps)
    if feature == Feats.UP_COUNT:
        return up_count
    if feature == Feats.DOWN_COUNT:
        return down_count
    if feature == Feats.UP_DECOY_COUNT:
        return up_decoy_count
    if feature == Feats.DOWN_DECOY_COUNT:
        return down_decoy_count
    if feature == Feats.SILENCE_FLAG:
        return float(up_count == 0.0 and down_count == 0.0)
    raise ValueError(f"unsupported brick policy feature: {feature}")


def _feature_tensors(
    values: dict[Feats, list[float]],
) -> dict[Feats, torch.Tensor]:
    return {
        feature: torch.tensor(
            feature_values,
            dtype=(
                torch.long
                if feature in (Feats.TIME_BINS, Feats.Dt_BINS)
                else torch.float32
            ),
        ).reshape(-1, 1)
        for feature, feature_values in values.items()
    }


def _densify(
    x: torch.Tensor,
    active: torch.Tensor,
    bs: int,
    fill_value: int | float = 0,
) -> torch.Tensor:
    full = torch.full(
        (bs,) + x.shape[1:],
        fill_value=fill_value,
        device=x.device,
        dtype=x.dtype,
    )
    full[active] = x
    return full
