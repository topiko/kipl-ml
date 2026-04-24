from __future__ import annotations

from time import perf_counter
from typing import Any, cast

import torch

from kipl_ml.data.wf_dataset import dict_to_device
from kipl_ml.logging.logger import get_logger
from kipl_ml.models.trgen import _hidden_w_mask
from kipl_ml.rl.enums import EntropyKeys, NoAction, StepActions
from kipl_ml.rl.streaming import WindowFeatureStreamer
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


def policy_rollout_streaming(
    obs: Any,
    X: dict[Feats, torch.Tensor],
    detach_period: int | None = None,
    sample: bool = True,
    add_tail_s: float | torch.Tensor | None = None,
    rtt_bins: int = 0,
    max_packets: int | None = None,
) -> _StreamingRollout:
    """Run policy stepwise on streamed windows and execute actions.

    If max_packets is set, stops once base_packets + requested_decoy >= max_packets.
    """

    res = cast(
        _StreamingRollout,
        _policy_rollout_streaming_impl(
            obs,
            X,
            detach_period=detach_period,
            sample=sample,
            add_tail_s=add_tail_s,
            rtt_bins=rtt_bins,
            max_packets=max_packets,
            record_policy=True,
        ),
    )
    return res


def policy_obfuscate_trace_streaming(
    obs: Any,
    X: dict[Feats, torch.Tensor],
    detach_period: int | None = None,
    sample: bool = True,
    add_tail_s: float | torch.Tensor | None = None,
    rtt_bins: int = 0,
    max_packets: int | None = None,
) -> dict[Feats, torch.Tensor]:
    """Obfuscate a trace stepwise using WindowFeatureStreamer.

    This is a lighter-weight variant of policy_rollout_streaming() for inference
    use-cases (e.g. NN defences): it avoids storing per-step policy outputs.

    If max_packets is set, stops once base_packets + requested_decoy >= max_packets.
    """

    X_obs = cast(
        dict[Feats, torch.Tensor],
        _policy_rollout_streaming_impl(
            obs,
            X,
            detach_period=detach_period,
            sample=sample,
            add_tail_s=add_tail_s,
            rtt_bins=rtt_bins,
            max_packets=max_packets,
            record_policy=False,
        ),
    )
    return X_obs


def _policy_rollout_streaming_impl(
    obs: Any,
    X: dict[Feats, torch.Tensor],
    detach_period: int | None,
    sample: bool,
    add_tail_s: float | torch.Tensor | None,
    rtt_bins: int,
    max_packets: int | None,
    record_policy: bool,
) -> dict[Feats, torch.Tensor] | _StreamingRollout:
    """Internal streaming rollout implementation.

    When record_policy=False, returns only X_obs. Otherwise returns the full
    (fd, act_time_bins, actions, log_ps, sel_probs, values_actor, entropies, X_obs)
    tuple.
    """

    if Feats.TIMES not in X or Feats.DIRS not in X:
        raise ValueError("X must contain TIMES and DIRS")
    Xb = {k: v.clone() for k, v in X.items()}
    if Feats.DECOY not in Xb:
        Xb[Feats.DECOY] = torch.zeros_like(Xb[Feats.TIMES])

    device = Xb[Feats.TIMES].device
    bs = int(Xb[Feats.TIMES].shape[0])

    if bs > 1 and max_packets is not None:
        raise NotImplementedError("max_packets is not supported for batch size > 1")
    max_packets = max_packets or 0

    # The streamer does per-trace stepping with Python control flow. If X lives on
    # CUDA, keep the streamer on CPU to avoid per-step GPU syncs.
    stream_device = torch.device("cpu")
    Xs = {
        Feats.TIMES: Xb[Feats.TIMES].detach().to(stream_device),
        Feats.DIRS: Xb[Feats.DIRS].detach().to(stream_device),
        Feats.DECOY: Xb[Feats.DECOY].detach().to(stream_device),
    }

    streamer = WindowFeatureStreamer(
        Xs,
        dt=obs.time_step,
        max_silence_s=obs.max_silence_s,
        features=obs.features,
        rtt_bins=rtt_bins,
        add_tail_s=add_tail_s,
    )

    log_ps_l: list[torch.Tensor] = []
    sel_probs_l: list[torch.Tensor] = []
    values_actor_l: list[torch.Tensor] = []
    ent_sel_l: list[torch.Tensor] = []
    ent_cond_l: list[torch.Tensor] = []
    act_time_bins_l: list[torch.Tensor] = []
    actions_l: list[StepActions] = [[] for _ in range(bs)]
    fd_steps: dict[Feats, list[torch.Tensor]] = {f: [] for f in obs.features}

    actions_a: StepActions = [NoAction(time=0) for _ in range(bs)]
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
    while True:
        # The streamer sleeps internally until it emits or hits its cap.
        t0 = perf_counter()
        fd_t, X_obs_, active = streamer.step(actions_a)
        t1 = perf_counter()
        t_stepping_ += t1 - t0

        if active.sum() == 0:
            break

        fd_active = {f: fd_t[f][active].to(device) for f in obs.features}
        h_active = _hidden_w_mask(hobs, active)
        t2 = perf_counter()
        t_acting_1_ += t2 - t1

        act_time_bins_a, actions_a, log_ps_a, sel_probs_a, values_a, ent_a, h_active = (
            obs.act_step(fd_active, h_active, sample=sample)
        )
        t3 = perf_counter()
        t_acting_2_ += t3 - t2

        hobs = _hidden_w_mask(hobs, active, h_active)

        if detach_period is not None and step_count % detach_period == 0:
            hobs = tuple(h_.detach() for h_ in hobs)

        active_idxs = torch.nonzero(active, as_tuple=False).flatten().tolist()
        for i, aidx in enumerate(active_idxs):
            for k in (Feats.TIMES, Feats.DIRS, Feats.DECOY):
                t_ = X_obs_[k][i]
                if t_.numel() == 0:
                    continue
                X_obs_l[aidx][k].append(X_obs_[k][i])
            actions_l[aidx].append(actions_a[i])
        t4 = perf_counter()
        t_storing_X_obs_ += t4 - t3

        if max_packets:
            if len(X_obs_l[aidx][Feats.TIMES]) >= max_packets:
                break

        if record_policy:
            for f in obs.features:
                fd_steps[f].append(fd_t[f].to("cpu"))

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
