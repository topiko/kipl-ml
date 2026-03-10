from __future__ import annotations

from typing import Any, Literal, cast, overload

import torch

from kipl_ml.data.utils import UPLOAD
from kipl_ml.rl.action import TraceExecState, send_exec
from kipl_ml.rl.enums import Actions
from kipl_ml.rl.observation import WindowFeatureStreamer, get_window_feature_dict
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


def _ensure_trace_dict(
    X: dict[Feats, torch.Tensor],
) -> dict[Feats, torch.Tensor]:
    if Feats.TIMES not in X or Feats.DIRS not in X:
        raise ValueError("X must contain TIMES and DIRS")

    X2 = {k: v.clone() for k, v in X.items()}
    if Feats.PADDING not in X2:
        X2[Feats.PADDING] = torch.zeros_like(X2[Feats.TIMES])
    return X2


def _apply_extend_end_inplace(
    X: dict[Feats, torch.Tensor], extend_end_s: float
) -> None:
    if extend_end_s <= 0:
        return

    bs, L = X[Feats.DIRS].shape
    device = X[Feats.DIRS].device
    mask = X[Feats.DIRS] == 0
    seq_lens = (~mask).sum(dim=1)
    col_idx = seq_lens[seq_lens != L]
    row_idx = torch.arange(bs, device=device)[seq_lens != L]

    # Insert a final UP packet so the silence extension becomes visible to the
    # window generator.
    X[Feats.DIRS][row_idx, col_idx] = UPLOAD
    X[Feats.TIMES][mask] += extend_end_s


def _hidden_w_mask(
    h: tuple[torch.Tensor, ...] | None,
    mask: torch.Tensor,
    hmasked: tuple[torch.Tensor, ...] | None = None,
) -> tuple[torch.Tensor, ...] | None:
    if h is None:
        if not mask.all():
            raise ValueError("Hidden is none, but only a subset is selected")
        return hmasked

    if hmasked is None:
        return tuple(h_[:, mask] for h_ in h)

    for i, h_ in enumerate(h):
        h_[:, mask] = hmasked[i]
    return h


def policy_rollout_single_pass(
    obs: Any,
    X: dict[Feats, torch.Tensor],
    *,
    detach_period: int = 20,
    sample: bool = True,
    extend_end_s: float = 2.0,
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
    """Run policy in one pass on precomputed windows and execute actions."""

    obs_ = cast(Any, obs)

    Xb = _ensure_trace_dict(X)
    _apply_extend_end_inplace(Xb, extend_end_s)

    fd = get_window_feature_dict(
        Xb,
        float(cast(Any, obs_.time_step)),
        float(cast(Any, obs_.max_silence_s)),
        features=list(cast(Any, obs_.features)),
        extend_end_s=0,
    )
    action_seq_lens = fd.pop(Feats.SEQ_LENS)

    act_times, actions, log_ps, sel_probs, values_actor, entropies, h = obs_.act(
        fd,
        None,
        h_detach_period=detach_period,
        seq_lens=action_seq_lens,
        sample=sample,
    )
    Xobs = send_exec(Xb, act_times, actions)

    fd[Feats.SEQ_LENS] = action_seq_lens
    return fd, act_times, actions, log_ps, sel_probs, values_actor, entropies, Xobs


def policy_rollout_streaming(
    obs: Any,
    X: dict[Feats, torch.Tensor],
    *,
    sample: bool = True,
    extend_end_s: float = 2.0,
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
            extend_end_s=extend_end_s,
            max_packets=max_packets,
            record_policy=True,
        ),
    )
    return res


def policy_obfuscate_trace_single_pass(
    obs: Any,
    X: dict[Feats, torch.Tensor],
    *,
    sample: bool = True,
    extend_end_s: float = 2.0,
) -> dict[Feats, torch.Tensor]:
    """Obfuscate a trace in one pass; return only the executed trace."""

    _, _, _, _, _, _, _, Xobs = policy_rollout_single_pass(
        obs,
        X,
        detach_period=100,
        sample=sample,
        extend_end_s=extend_end_s,
    )
    return Xobs


def policy_obfuscate_trace_streaming(
    obs: Any,
    X: dict[Feats, torch.Tensor],
    *,
    sample: bool = True,
    extend_end_s: float = 2.0,
    max_packets: int | None = None,
) -> dict[Feats, torch.Tensor]:
    """Obfuscate a trace stepwise using WindowFeatureStreamer.

    This is a lighter-weight variant of policy_rollout_streaming() for inference
    use-cases (e.g. NN defences): it avoids storing per-step policy outputs.

    If max_packets is set, stops once base_packets + requested_padding >= max_packets.
    """

    Xobs = cast(
        dict[Feats, torch.Tensor],
        _policy_rollout_streaming_impl(
            obs,
            X,
            sample=sample,
            extend_end_s=extend_end_s,
            max_packets=max_packets,
            record_policy=False,
        ),
    )
    return Xobs


@overload
def _policy_rollout_streaming_impl(
    obs: Any,
    X: dict[Feats, torch.Tensor],
    *,
    sample: bool,
    extend_end_s: float,
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
    extend_end_s: float,
    max_packets: int | None,
    record_policy: Literal[False],
) -> dict[Feats, torch.Tensor]: ...


def _policy_rollout_streaming_impl(
    obs: Any,
    X: dict[Feats, torch.Tensor],
    *,
    sample: bool,
    extend_end_s: float,
    max_packets: int | None,
    record_policy: bool,
) -> dict[Feats, torch.Tensor] | _StreamingRollout:
    """Internal streaming rollout implementation.

    When record_policy=False, returns only Xobs. Otherwise returns the full
    (fd, act_times, actions, log_ps, sel_probs, values_actor, entropies, Xobs)
    tuple.
    """

    Xb = _ensure_trace_dict(X)
    _apply_extend_end_inplace(Xb, extend_end_s)

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

    streamer_features = list(obs_.features)
    if Feats.WINDOW_BINS not in streamer_features:
        streamer_features.append(Feats.WINDOW_BINS)

    streamer = WindowFeatureStreamer(
        Xs,
        dt=float(obs_.time_step),
        max_silence_s=float(obs_.max_silence_s),
        features=streamer_features,
        extend_end_s=0,
    )

    exec_state = TraceExecState(
        {
            Feats.TIMES: Xb[Feats.TIMES].clone(),
            Feats.DIRS: Xb[Feats.DIRS].clone(),
            Feats.PADDING: Xb[Feats.PADDING].clone(),
        }
    )

    base_n = (Xb[Feats.DIRS] != 0).sum(dim=1).long()
    pad_n = torch.zeros((bs,), device=device, dtype=torch.long)

    hobs = None

    log_ps_l: list[torch.Tensor] = []
    sel_probs_l: list[torch.Tensor] = []
    values_actor_l: list[torch.Tensor] = []
    ent_sel_l: list[torch.Tensor] = []
    ent_cond_l: list[torch.Tensor] = []
    act_times_l: list[torch.Tensor] = []
    actions_l: dict[Actions, list[torch.Tensor]] | None = None
    fd_steps: dict[Feats, list[torch.Tensor]] | None = (
        {f: [] for f in obs_.features} if record_policy else None
    )

    while True:
        fd_t_full = streamer.step()

        active_cpu = fd_t_full[Feats.TIMES].isfinite().squeeze(1)
        if int(active_cpu.sum().item()) == 0:
            break

        active = active_cpu.to(device=device)

        if record_policy:
            assert fd_steps is not None
            for f in obs_.features:
                fd_steps[f].append(fd_t_full[f].to(device))

        fd_t_active = {
            k: fd_t_full[k][active_cpu].nan_to_num(nan=0.0).to(device)
            for k in obs_.features
        }

        h_active = _hidden_w_mask(hobs, active)
        act_times_a, actions_a, log_ps_a, sel_probs_a, values_a, ent_a, h_active = (
            obs_.act_step(fd_t_active, h_active, sample=sample)
        )

        hobs = _hidden_w_mask(hobs, active, h_active)
        active_idx_cpu = torch.where(active_cpu)[0]
        active_idx = active_idx_cpu.to(device)

        exec_state.step(trace_idx=active_idx, times=act_times_a, actions=actions_a)

        if Actions.DELAY in actions_a and (actions_a[Actions.DELAY] > 0).any():
            # Apply delay in window-bin space to avoid float transfers.
            shift_full = torch.zeros((bs,), device=stream_device, dtype=torch.long)
            delay_mask_a = (actions_a[Actions.DELAY] > 0).squeeze(1).detach().to("cpu")
            shift_full[active_idx_cpu] = delay_mask_a.to(torch.long)
            # Use the policy's action times (right edge: TIMES + Dt) to determine
            # delay start. This matches TraceExecState/send_exec semantics.
            dt_s = float(obs_.time_step)
            start_bins_full = torch.zeros((bs,), device=stream_device, dtype=torch.long)
            start_bins_a = torch.round(act_times_a.squeeze(1).detach().to("cpu") / dt_s).to(
                torch.long
            )
            start_bins_full[active_idx_cpu] = start_bins_a
            streamer.apply_delay_bins(start_bins_full, shift_full)

        if max_packets is not None:
            active_n = int(active_cpu.sum().item())
            inc = torch.zeros((active_n,), device=device, dtype=torch.long)
            if Actions.SEND_COUNT_UP in actions_a:
                inc = inc + actions_a[Actions.SEND_COUNT_UP].squeeze(1).to(torch.long)
            if Actions.SEND_COUNT_DOWN in actions_a:
                inc = inc + actions_a[Actions.SEND_COUNT_DOWN].squeeze(1).to(torch.long)
            pad_n[active_idx] = pad_n[active_idx] + inc
            if ((base_n + pad_n) >= int(max_packets)).all():
                break

        if record_policy:
            # Scatter back to full batch.
            act_times_t = torch.full((bs, 1), torch.nan, device=device)
            act_times_t[active] = act_times_a

            log_ps_t = torch.zeros((bs, 1), device=device)
            log_ps_t[active] = log_ps_a

            values_t = torch.zeros((bs, 1), device=device)
            values_t[active] = values_a

            sel_probs_t = torch.zeros((bs, 1, sel_probs_a.shape[-1]), device=device)
            sel_probs_t[active] = sel_probs_a

            ent_sel_t = torch.zeros((bs, 1), device=device)
            ent_sel_t[active] = ent_a["selection_entropy"]

            ent_cond_t = torch.zeros((bs, 1), device=device)
            ent_cond_t[active] = ent_a["conditional_entropy"]

            if actions_l is None:
                actions_l = {k: [] for k in actions_a.keys()}

            for k in actions_l.keys():
                a_full = torch.zeros((bs, 1), device=device, dtype=actions_a[k].dtype)
                a_full[active] = actions_a[k]
                actions_l[k].append(a_full)

            act_times_l.append(act_times_t)
            log_ps_l.append(log_ps_t)
            sel_probs_l.append(sel_probs_t)
            values_actor_l.append(values_t)
            ent_sel_l.append(ent_sel_t)
            ent_cond_l.append(ent_cond_t)

    Xobs = exec_state.finalize()
    if max_packets is not None:
        Xobs = {k: v[:, : int(max_packets)] for k, v in Xobs.items()}

    if record_policy is False:
        return Xobs

    if actions_l is None:
        raise ValueError("No actions produced")

    assert fd_steps is not None

    act_times = torch.cat(act_times_l, dim=1)
    log_ps = torch.cat(log_ps_l, dim=1)
    sel_probs = torch.cat(sel_probs_l, dim=1)
    values_actor = torch.cat(values_actor_l, dim=1)
    entropies = {
        "selection_entropy": torch.cat(ent_sel_l, dim=1),
        "conditional_entropy": torch.cat(ent_cond_l, dim=1),
    }
    actions = {k: torch.cat(vs, dim=1) for k, vs in actions_l.items()}

    fd = {f: torch.cat(vs, dim=1) for f, vs in fd_steps.items()}
    fd[Feats.SEQ_LENS] = act_times.isfinite().sum(dim=1).long()

    return fd, act_times, actions, log_ps, sel_probs, values_actor, entropies, Xobs
