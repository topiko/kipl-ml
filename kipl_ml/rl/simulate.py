from __future__ import annotations

from typing import Any, Literal, cast, overload

import torch

from kipl_ml.models.trgen import _hidden_w_mask
from kipl_ml.rl.enums import Actions, NoAction, StepActions
from kipl_ml.rl.observation import WindowFeatureStreamer
from kipl_ml.trace.enums import Feats

_StreamingRollout = tuple[
    dict[Feats, torch.Tensor],
    torch.Tensor,
    dict[Actions, torch.Tensor],
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    dict[str, torch.Tensor],
    dict[Feats, torch.Tensor],
]


def _batch_packet_level_features(
    fd_packet_level: list[dict[Feats, list[torch.Tensor]]],
) -> dict[Feats, torch.Tensor]:
    """Materialize per-trace packet-level histories into a padded batch."""

    if not fd_packet_level:
        raise ValueError("fd_packet_level must not be empty")

    bs = len(fd_packet_level)
    feats = list(fd_packet_level[0].keys())
    per_trace: dict[Feats, list[torch.Tensor]] = {f: [] for f in feats}

    max_len = 0
    for trace_i in range(bs):
        for f in feats:
            if fd_packet_level[trace_i][f]:
                t = torch.cat(fd_packet_level[trace_i][f], dim=0)
            else:
                t = torch.zeros((0,), dtype=torch.long)
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
            t = per_trace[f][i]
            if t.numel() == 0:
                continue
            batched[i, : t.numel()] = t
        out[f] = batched

    return out


def policy_rollout_streaming(
    obs: Any,
    X: dict[Feats, torch.Tensor],
    *,
    sample: bool = True,
    cut_off_time_s: float | torch.Tensor | None = None,
    max_packets: int | None = None,
) -> tuple[
    dict[Feats, torch.Tensor],
    torch.Tensor,
    dict[Actions, torch.Tensor],
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    dict[str, torch.Tensor],
    dict[Feats, torch.Tensor],
]:
    """Run policy stepwise on streamed windows and execute actions.

    If max_packets is set, stops once base_packets + requested_padding >= max_packets.
    """

    res = cast(
        _StreamingRollout,
        _policy_rollout_streaming_impl(
            obs,
            X,
            sample=sample,
            cut_off_time_s=cut_off_time_s,
            max_packets=max_packets,
            record_policy=True,
        ),
    )
    return res


def policy_obfuscate_trace_streaming(
    obs: Any,
    X: dict[Feats, torch.Tensor],
    *,
    sample: bool = True,
    cut_off_time_s: float | torch.Tensor | None = None,
    max_packets: int | None = None,
) -> dict[Feats, torch.Tensor]:
    """Obfuscate a trace stepwise using WindowFeatureStreamer.

    This is a lighter-weight variant of policy_rollout_streaming() for inference
    use-cases (e.g. NN defences): it avoids storing per-step policy outputs.

    If max_packets is set, stops once base_packets + requested_padding >= max_packets.
    """

    X_obs = cast(
        dict[Feats, torch.Tensor],
        _policy_rollout_streaming_impl(
            obs,
            X,
            sample=sample,
            cut_off_time_s=cut_off_time_s,
            max_packets=max_packets,
            record_policy=False,
        ),
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
    record_policy: Literal[True],
) -> tuple[
    dict[Feats, torch.Tensor],
    torch.Tensor,
    dict[Actions, torch.Tensor],
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
    record_policy: Literal[False],
) -> dict[Feats, torch.Tensor]: ...


def _policy_rollout_streaming_impl(
    obs: Any,
    X: dict[Feats, torch.Tensor],
    *,
    sample: bool,
    cut_off_time_s: float | torch.Tensor | None,
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
    if Feats.PADDING not in Xb:
        Xb[Feats.PADDING] = torch.zeros_like(Xb[Feats.TIMES])

    obs_ = cast(Any, obs)

    device = Xb[Feats.TIMES].device
    bs = int(Xb[Feats.TIMES].shape[0])

    # The streamer does per-trace stepping with Python control flow. If X lives on
    # CUDA, keep the streamer on CPU to avoid per-step GPU syncs.
    stream_device = torch.device("cpu") if device.type == "cuda" else device
    Xs = {
        Feats.TIMES: Xb[Feats.TIMES].detach().to(stream_device),
        Feats.DIRS: Xb[Feats.DIRS].detach().to(stream_device),
        Feats.PADDING: Xb[Feats.PADDING].detach().to(stream_device),
    }

    streamer = WindowFeatureStreamer(
        Xs,
        dt=obs_.time_step,
        max_silence_s=obs_.max_silence_s,
        features=obs_.features,
        cut_off_time_s=cut_off_time_s,
    )

    log_ps_l: list[torch.Tensor] = []
    sel_probs_l: list[torch.Tensor] = []
    values_actor_l: list[torch.Tensor] = []
    ent_sel_l: list[torch.Tensor] = []
    ent_cond_l: list[torch.Tensor] = []
    act_time_bins_l: list[torch.Tensor] = []
    actions_l: dict[Actions, list[torch.Tensor]] | None = None
    fd_steps: dict[Feats, list[torch.Tensor]] = {f: [] for f in obs_.features}

    actions_a: StepActions = [NoAction(time=0) for _ in range(bs)]
    fd_packet_level: list[dict[Feats, list[torch.Tensor]]] = [
        {Feats.TIMES: [], Feats.DIRS: [], Feats.PADDING: []} for _ in range(bs)
    ]

    hobs = None
    while True:
        # The streamer sleeps internally until it emits or hits its cap.
        fd_t, fd_packet_level_, active = streamer.step(actions_a)

        if active.sum() == 0:
            break

        active_idxs = torch.nonzero(active, as_tuple=False).flatten().tolist()
        for i, aidx in enumerate(active_idxs):
            for k in (Feats.TIMES, Feats.DIRS, Feats.PADDING):
                fd_packet_level[aidx][k].append(fd_packet_level_[k][i])

        if record_policy:
            for f in obs_.features:
                fd_steps[f].append(fd_t[f].to("cpu"))

        h_active = _hidden_w_mask(hobs, active)
        act_time_bins_a, actions_a, log_ps_a, sel_probs_a, values_a, ent_a, h_active = (
            obs_.act_step(fd_t, h_active, sample=sample)
        )

        hobs = _hidden_w_mask(hobs, active, h_active)

        # STOP HERE

    X_obs = _batch_packet_level_features(fd_packet_level)

    if record_policy is False:
        return X_obs

    if actions_l is None:
        raise ValueError("No actions produced")

    assert fd_steps is not None

    act_time_bins = torch.cat(act_time_bins_l, dim=1)
    log_ps = torch.cat(log_ps_l, dim=1)
    sel_probs = torch.cat(sel_probs_l, dim=1)
    values_actor = torch.cat(values_actor_l, dim=1)
    entropies = {
        "selection_entropy": torch.cat(ent_sel_l, dim=1),
        "conditional_entropy": torch.cat(ent_cond_l, dim=1),
    }
    actions = {k: torch.cat(vs, dim=1) for k, vs in actions_l.items()}

    fd = {f: torch.cat(vs, dim=1) for f, vs in fd_steps.items()}
    fd[Feats.SEQ_LENS] = (act_time_bins >= 0).sum(dim=1).long()

    return fd, act_time_bins, actions, log_ps, sel_probs, values_actor, entropies, X_obs
