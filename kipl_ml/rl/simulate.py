from __future__ import annotations

from time import perf_counter
from typing import Any, cast

import mbnt
import numpy as np
import torch

from kipl_ml.data.utils import DOWNLOAD, UPLOAD
from kipl_ml.data.wf_dataset import dict_to_device
from kipl_ml.logging.logger import get_logger
from kipl_ml.models.trgen import AGENT1, _hidden_w_mask
from kipl_ml.rl.enums import (
    Actions,
    EntropyKeys,
    NoAction,
    StepAction,
    StepActions,
)
from kipl_ml.trace.enums import Feats

logger = get_logger(__name__)

_StreamingRollout = tuple[
    dict[Feats, torch.Tensor],
    torch.Tensor,
    list[list[StepActions]],
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    dict[str, torch.Tensor],
    dict[Feats, torch.Tensor],
]


def map_step_actions(
    step_actions: list[StepAction], active_mask: np.array
) -> tuple[list[tuple[int, mbnt.RlAction]], list[tuple[int, mbnt.RlAction]]]:
    client_actions: list[tuple[int, mbnt.RlAction]] = []
    server_actions: list[tuple[int, mbnt.RlAction]] = []

    if len(step_actions) != active_mask.sum():
        raise ValueError("step_actions != active_mask.sum()??")

    active_indices = np.nonzero(active_mask)[0]
    for batch_idx, step_action in zip(active_indices, step_actions, strict=True):
        for ack_ in (Actions.SEND_UP, Actions.SEND_DOWN):
            if ack_ in step_action:
                act = step_action[ack_]
                ack_tuple = (
                    batch_idx,
                    mbnt.RlAction.Decoy(
                        count=int(act.count),
                        after_steps=int(act.after_steps),
                        bypass=bool(act.bypass),
                        replace=bool(act.replace),
                    ),
                )
                if ack_ == Actions.SEND_UP:
                    client_actions.append(ack_tuple)
                elif ack_ == Actions.SEND_DOWN:
                    server_actions.append(ack_tuple)
                else:
                    raise ValueError("Invalid ack")

        for ack_ in (Actions.DELAY_UP, Actions.DELAY_DOWN):
            if ack_ in step_action:
                act = step_action[ack_]
                ack_tuple = (
                    batch_idx,
                    mbnt.RlAction.Delay(
                        steps=int(act.steps),
                        bypass=bool(act.bypass),
                        replace=bool(act.replace),
                    ),
                )
                if ack_ == Actions.DELAY_UP:
                    client_actions.append(ack_tuple)
                elif ack_ == Actions.DELAY_DOWN:
                    server_actions.append(ack_tuple)
                else:
                    raise ValueError("Invalid ack")

    return client_actions, server_actions


def _batch_packet_level_features(
    fd_packet_level: list[dict[Feats, list[torch.Tensor]]], device: torch.DeviceObjType
) -> dict[Feats, torch.Tensor]:
    """Materialize per-trace packet-level histories into a padded batch."""

    if not fd_packet_level:
        raise ValueError("fd_packet_level must not be empty")

    bs = len(fd_packet_level)
    feats = list(fd_packet_level[0].keys())
    per_trace: dict[Feats, list[torch.Tensor]] = {f: [] for f in feats}

    max_len = 0
    time_sort_idx: list[torch.Tensor] = [torch.tensor([], dtype=torch.long)] * bs
    for trace_i in range(bs):
        for f in feats:
            if fd_packet_level[trace_i][f]:
                t = torch.cat(fd_packet_level[trace_i][f], dim=0)
            else:
                raise ValueError(f"Trace {trace_i} has no data for feature {f}")

            if f == Feats.TIMES:
                time_sort_idx[trace_i] = torch.argsort(t)
            per_trace[f].append(t)
            max_len = max(max_len, int(t.numel()))

    if max_len == 0:
        max_len = 1

    out: dict[Feats, torch.Tensor] = {}
    for f in feats:
        sample = per_trace[f][0]
        pad_val = -1 if f == Feats.TIMES else 0
        batched = torch.full(
            (bs, max_len), pad_val, device=sample.device, dtype=sample.dtype
        )
        for i in range(bs):
            t = per_trace[f][i][time_sort_idx[i]]

            if t.numel() == 0:
                continue

            batched[i, : t.numel()] = t

        out[f] = batched

    return dict_to_device(out, device)


def policy_rollout(
    obs: Any,
    trace_paths: list[str],
    device: torch.DeviceObjType,
    detach_period: int | None = None,
    sample: bool = True,
    add_tail_s: float | torch.Tensor | None = None,
    max_packets: int | None = None,
    trim_raw: int = 0,
    network_delay_millis: int | None = None,
    network_packets_per_second: int | None = None,
) -> _StreamingRollout:
    """Run policy stepwise on streamed windows and execute actions.

    If max_packets is set, stops once base_packets + requested_decoy >= max_packets.
    """

    res = cast(
        _StreamingRollout,
        _policy_rollout_impl(
            obs,
            trace_paths,
            device=device,
            detach_period=detach_period,
            sample=sample,
            add_tail_s=add_tail_s,
            max_packets=max_packets,
            trim_raw=trim_raw,
            network_delay_millis=network_delay_millis,
            network_packets_per_second=network_packets_per_second,
            record_policy=True,
        ),
    )
    return res


def policy_obfuscate_trace(
    obs: Any,
    trace_paths: list[str],
    device: torch.DeviceObjType,
    detach_period: int | None = None,
    sample: bool = True,
    add_tail_s: float | torch.Tensor | None = None,
    rtt_bins: int = 0,
    max_packets: int | None = None,
    trim_raw: int = 0,
    network_delay_millis: int = 10,
    network_packets_per_second: int = 0,
) -> dict[Feats, torch.Tensor]:
    """Obfuscate a trace stepwise using the rollout simulator.

    This is a lighter-weight variant of policy_rollout() for inference
    use-cases (e.g. NN defences): it avoids storing per-step policy outputs.

    If max_packets is set, stops once base_packets + requested_decoy >= max_packets.
    """

    X_obs = cast(
        dict[Feats, torch.Tensor],
        _policy_rollout_impl(
            obs,
            trace_paths,
            device,
            detach_period=detach_period,
            sample=sample,
            add_tail_s=add_tail_s,
            max_packets=max_packets,
            trim_raw=trim_raw,
            network_delay_millis=network_delay_millis,
            network_packets_per_second=network_packets_per_second,
            record_policy=False,
        ),
    )
    return X_obs


def _policy_rollout_impl(
    obs: AGENT1,
    trace_paths: list[str],
    device: torch.DeviceObjType,
    detach_period: int | None,
    sample: bool,
    add_tail_s: float | torch.Tensor | None,
    max_packets: int | None,
    trim_raw: int,
    network_delay_millis: int,
    network_packets_per_second: int,
    record_policy: bool,
) -> dict[Feats, torch.Tensor] | _StreamingRollout:
    """Internal streaming rollout implementation.

    When record_policy=False, returns only X_obs. Otherwise returns the full
    (fd, act_time_bins, actions, log_ps, sel_probs, values_actor, entropies, X_obs)
    tuple.
    """

    num_machines = 4096
    max_trace_length = 60_000
    seed = 0
    max_silence_bins = int(round(obs.max_silence_s / obs.time_step))

    simul_batch = mbnt.Batch.new(
        trace_paths=trace_paths,
        window_duration_ns=int(round(obs.time_step * 1e9)),
        num_machines=num_machines,
        network_delay_millis=network_delay_millis,
        network_packets_per_second=network_packets_per_second,
        max_trace_length=max_trace_length,
        seed=seed,
        trim_raw=trim_raw,
        relative=True,
    )
    bs = len(trace_paths)

    if bs > 1 and max_packets is not None:
        raise NotImplementedError("max_packets is not supported for batch size > 1")
    max_packets = max_packets or 0

    # The streamer does per-trace stepping with Python control flow. If X lives on
    # CUDA, keep the streamer on CPU to avoid per-step GPU syncs.
    log_ps_l: list[torch.Tensor] = []
    sel_probs_l: list[torch.Tensor] = []
    values_actor_l: list[torch.Tensor] = []
    ent_sel_l: list[torch.Tensor] = []
    ent_cond_l: list[torch.Tensor] = []
    act_time_bins_l: list[torch.Tensor] = []
    actions_l: list[StepActions] = [[] for _ in range(bs)]
    fd_steps: dict[Feats, list[torch.Tensor]] = {f: [] for f in obs.features}

    actions_a: StepActions = [NoAction(time=0) for _ in range(bs)]
    active = np.ones(bs, dtype=bool)
    has_started = np.zeros_like(active, dtype=bool)
    next_bins = np.zeros(bs, dtype=np.int64)
    X_obs_l: list[dict[Feats, list[torch.Tensor]]] = [
        {Feats.TIMES: [], Feats.DIRS: [], Feats.DECOY: []} for _ in range(bs)
    ]

    def _densify(x: torch.Tensor, active: torch.Tensor) -> torch.Tensor:
        full = torch.zeros((bs,) + x.shape[1:], device=x.device, dtype=x.dtype)
        full[active] = x
        return full

    t_stepping_ = 0.0
    t_acting_1_ = 0.0
    t_acting_2_ = 0.0
    t_storing_X_obs_ = 0.0
    t_storing_policy_ = 0.0

    hobs = None
    step_count = 0
    n_packets = 0
    while True:
        # The streamer sleeps internally until it emits or hits its cap.
        t0 = perf_counter()

        client_actions, server_actions = map_step_actions(actions_a, active)
        trace_windows, stepped_bins = simul_batch.step_until_emit(
            client_actions, server_actions, max_silence_bins
        )

        next_active = ~np.asarray(simul_batch.is_done(), dtype=bool)
        t1 = perf_counter()
        t_stepping_ += t1 - t0

        prev_idxs = np.nonzero(active)[0]
        next_idxs = np.nonzero(next_active)[0]

        if (n_next_active := next_active.sum()) == 0:
            break

        i = 0
        fd_w = {f: torch.zeros((n_next_active, 1)) for f in obs.features}
        for idx, window, n_steps in zip(range(bs), trace_windows, stepped_bins):
            times = torch.tensor(window[0] / 1e9).float()  # ns -> s
            dirs = torch.tensor(window[1]).float()  # dirs
            decoys = torch.tensor(window[2]).float()  # decoy

            if (len(times) > 0) and (not has_started[idx]):
                has_started[idx] = True

            if idx not in prev_idxs or (not has_started[idx]):
                # This trace is done now -> discard
                continue

            # Store the observation.
            X_obs_l[idx][Feats.TIMES].append(times)
            X_obs_l[idx][Feats.DIRS].append(dirs)
            X_obs_l[idx][Feats.DECOY].append(decoys)

            # The actions_a is a list of the prev action len(actions_a) == len(prev_idxs)
            # We need to map e.g., prev_idxs = [0, 2, 12] and idx == 12 -> actions_a[2]
            aidx = np.argwhere(prev_idxs == idx)[0, 0]

            if idx not in next_idxs:
                continue

            current_bin = int(next_bins[idx])
            actions_l[idx].append(actions_a[aidx])
            fd_w[Feats.UP_COUNT][i, 0] = ((dirs == UPLOAD) & (decoys == 0)).sum()
            fd_w[Feats.DOWN_COUNT][i, 0] = ((dirs == DOWNLOAD) & (decoys == 0)).sum()
            fd_w[Feats.Dt_BINS][i, 0] = n_steps
            fd_w[Feats.TIME_BINS][i, 0] = current_bin
            fd_w[Feats.SILENCE_FLAG][i, 0] = float(
                (fd_w[Feats.UP_COUNT][i, 0] == 0)
                and (fd_w[Feats.DOWN_COUNT][i, 0] == 0)
            )

            next_bins[idx] += int(n_steps)

            i += 1

        # Update which are active
        active = next_active

        h_active = _hidden_w_mask(hobs, active)
        t2 = perf_counter()
        t_acting_1_ += t2 - t1

        fd_w = dict_to_device(fd_w, device=device)
        act_time_bins_a, actions_a, log_ps_a, sel_probs_a, values_a, ent_a, h_active = (
            obs.act_step(fd_w, h_active, sample=sample)
        )
        t3 = perf_counter()
        t_acting_2_ += t3 - t2

        hobs = _hidden_w_mask(hobs, active, h_active)

        if detach_period is not None and step_count % detach_period == 0:
            hobs = tuple(h_.detach() for h_ in hobs)

        t4 = perf_counter()
        t_storing_X_obs_ += t4 - t3

        if max_packets:
            # NOTE: n_packets only applies when bs = 1
            n_packets += times.numel()
            if n_packets >= max_packets:
                break

        if record_policy:
            for f in obs.features:
                fd_ = torch.zeros((bs, 1))
                fd_[active, 0] = fd_w[f].flatten().to("cpu")
                fd_steps[f].append(fd_)

            act_time_bins_l.append(_densify(act_time_bins_a, active).detach().cpu())
            log_ps_l.append(_densify(log_ps_a, active))
            sel_probs_l.append(_densify(sel_probs_a, active))
            values_actor_l.append(_densify(values_a, active))
            ent_sel_l.append(_densify(ent_a[EntropyKeys.SELECTION_ENTROPY], active))
            ent_cond_l.append(_densify(ent_a[EntropyKeys.COND_ENTROPY], active))

        t5 = perf_counter()
        t_storing_policy_ += t5 - t4

        step_count += 1

    # print("Batch simulated, timings:")
    # print(f"{t_stepping_=}")
    # print(f"{t_acting_1_=}")
    # print(f"{t_acting_2_=}")
    # print(f"{t_storing_X_obs_=}")
    # print(f"{t_storing_policy_=}")

    X_obs = _batch_packet_level_features(X_obs_l, device)

    if not record_policy:
        return X_obs

    assert fd_steps is not None

    act_time_bins = torch.cat(act_time_bins_l, dim=1)
    # 0 is here used for padding, action time _can never be 0_..
    act_time_bins[act_time_bins == 0] = -1

    log_ps = torch.cat(log_ps_l, dim=1)
    sel_probs = torch.cat(sel_probs_l, dim=1)
    values_actor = torch.cat(values_actor_l, dim=1)
    entropies = {
        "selection_entropy": torch.cat(ent_sel_l, dim=1),
        "conditional_entropy": torch.cat(ent_cond_l, dim=1),
    }

    fd = {f: torch.cat(vs, dim=1) for f, vs in fd_steps.items()}
    fd[Feats.SEQ_LENS] = (act_time_bins > 0).sum(dim=1).long()

    fd = dict_to_device(fd, device)
    act_time_bins = act_time_bins.to(device)

    return (
        fd,
        act_time_bins,
        actions_l,
        log_ps,
        sel_probs,
        values_actor,
        entropies,
        X_obs,
    )
