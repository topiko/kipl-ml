from __future__ import annotations

from time import perf_counter
from typing import Any, Literal, cast, overload

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
    list[StepActions],
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
    *,
    sample: bool = True,
    cut_off_time_s: float | torch.Tensor | None = None,
    max_packets: int | None = None,
    stream_workers: int = 1,
) -> tuple[
    dict[Feats, torch.Tensor],
    torch.Tensor,
    list[StepActions],
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    dict[str, torch.Tensor],
    dict[Feats, torch.Tensor],
]:
    """Run policy stepwise on streamed windows and execute actions.

    If max_packets is set, stops once base_packets + requested_decoy >= max_packets.
    """

    t0 = perf_counter()
    res = cast(
        _StreamingRollout,
        _policy_rollout_streaming_impl(
            obs,
            X,
            sample=sample,
            cut_off_time_s=cut_off_time_s,
            max_packets=max_packets,
            stream_workers=stream_workers,
            record_policy=True,
        ),
    )
    logger.info(
        "stream rollout wall time (workers=%d): %.3fs",
        int(stream_workers),
        perf_counter() - t0,
    )
    return res


def policy_obfuscate_trace_streaming(
    obs: Any,
    X: dict[Feats, torch.Tensor],
    *,
    sample: bool = True,
    cut_off_time_s: float | torch.Tensor | None = None,
    max_packets: int | None = None,
    stream_workers: int = 1,
) -> dict[Feats, torch.Tensor]:
    """Obfuscate a trace stepwise using WindowFeatureStreamer.

    This is a lighter-weight variant of policy_rollout_streaming() for inference
    use-cases (e.g. NN defences): it avoids storing per-step policy outputs.

    If max_packets is set, stops once base_packets + requested_decoy >= max_packets.
    """

    t0 = perf_counter()
    X_obs = cast(
        dict[Feats, torch.Tensor],
        _policy_rollout_streaming_impl(
            obs,
            X,
            sample=sample,
            cut_off_time_s=cut_off_time_s,
            max_packets=max_packets,
            stream_workers=stream_workers,
            record_policy=False,
        ),
    )
    logger.info(
        "stream obfuscation wall time (workers=%d): %.3fs",
        int(stream_workers),
        perf_counter() - t0,
    )
    return X_obs


@overload
def _policy_rollout_streaming_impl(
    obs: Any,
    X: dict[Feats, torch.Tensor],
    *,
    sample: bool,
    cut_off_time_s: float | torch.Tensor | None,
    max_packets: int | None,
    stream_workers: int,
    record_policy: Literal[True],
) -> tuple[
    dict[Feats, torch.Tensor],
    torch.Tensor,
    list[StepActions],
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    dict[str, torch.Tensor],
    dict[Feats, torch.Tensor],
]: ...


@overload
def _policy_rollout_streaming_impl(
    obs: Any,
    X: dict[Feats, torch.Tensor],
    *,
    sample: bool,
    cut_off_time_s: float | torch.Tensor | None,
    max_packets: int | None,
    stream_workers: int,
    record_policy: Literal[False],
) -> dict[Feats, torch.Tensor]: ...


def _policy_rollout_streaming_impl(
    obs: Any,
    X: dict[Feats, torch.Tensor],
    *,
    sample: bool,
    cut_off_time_s: float | torch.Tensor | None,
    max_packets: int | None,
    stream_workers: int,
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
        cut_off_time_s=cut_off_time_s,
        step_workers=stream_workers,
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
    fd_packet_level: list[dict[Feats, list[torch.Tensor]]] = [
        {Feats.TIMES: [], Feats.DIRS: [], Feats.DECOY: []} for _ in range(bs)
    ]

    def _densify(x: torch.Tensor) -> torch.Tensor:
        full = torch.zeros((bs,) + x.shape[1:], device=x.device, dtype=x.dtype)
        full[active] = x
        return full

    hobs = None
    while True:
        # The streamer sleeps internally until it emits or hits its cap.
        fd_t, fd_packet_level_, active = streamer.step(actions_a)

        if active.sum() == 0:
            break

        active_idxs = torch.nonzero(active, as_tuple=False).flatten().tolist()
        for i, aidx in enumerate(active_idxs):
            for k in (Feats.TIMES, Feats.DIRS, Feats.DECOY):
                t_ = fd_packet_level_[k][i]
                if t_.numel() == 0:
                    continue
                fd_packet_level[aidx][k].append(fd_packet_level_[k][i])

            actions_l[aidx].append(actions_a[i])

        if record_policy:
            for f in obs.features:
                fd_steps[f].append(fd_t[f].to("cpu"))

        fd_active = {f: fd_t[f][active].to(device) for f in obs.features}
        h_active = _hidden_w_mask(hobs, active)
        act_time_bins_a, actions_a, log_ps_a, sel_probs_a, values_a, ent_a, h_active = (
            obs.act_step(fd_active, h_active, sample=sample)
        )

        hobs = _hidden_w_mask(hobs, active, h_active)

        act_time_bins_l.append(_densify(act_time_bins_a).detach().cpu())
        log_ps_l.append(_densify(log_ps_a))
        sel_probs_l.append(_densify(sel_probs_a))
        values_actor_l.append(_densify(values_a))
        ent_sel_l.append(_densify(ent_a[EntropyKeys.SELECTION_ENTROPY]))
        ent_cond_l.append(_densify(ent_a[EntropyKeys.COND_ENTROPY]))

    X_obs = _batch_packet_level_features(fd_packet_level, device)

    if record_policy is False:
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
