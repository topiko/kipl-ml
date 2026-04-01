from __future__ import annotations

from typing import Any, Literal, cast, overload

import torch

from kipl_ml.models.trgen import _hidden_w_mask
from kipl_ml.rl.action import TraceExecState, execute_actions_from_sequence
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


def policy_rollout_single_pass(
    obs: Any,
    X: dict[Feats, torch.Tensor],
    *,
    detach_period: int = 20,
    sample: bool = True,
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

    if Feats.TIMES not in X or Feats.DIRS not in X:
        raise ValueError("X must contain TIMES and DIRS")
    Xb = {k: v.clone() for k, v in X.items()}
    if Feats.PADDING not in Xb:
        Xb[Feats.PADDING] = torch.zeros_like(Xb[Feats.TIMES])

    fd = get_window_feature_dict(
        Xb,
        float(cast(Any, obs_.time_step)),
        float(cast(Any, obs_.max_silence_s)),
        features=list(cast(Any, obs_.features)),
    )
    action_seq_lens = fd.pop(Feats.SEQ_LENS)

    act_time_bins, actions, log_ps, sel_probs, values_actor, entropies, h = obs_.act(
        fd,
        None,
        h_detach_period=detach_period,
        seq_lens=action_seq_lens,
        sample=sample,
    )

    X_obs = execute_actions_from_sequence(
        Xb, act_time_bins, actions, time_step_s=float(obs_.time_step)
    )

    fd[Feats.SEQ_LENS] = action_seq_lens
    return fd, act_time_bins, actions, log_ps, sel_probs, values_actor, entropies, X_obs


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


def policy_obfuscate_trace_single_pass(
    obs: Any,
    X: dict[Feats, torch.Tensor],
    *,
    sample: bool = True,
) -> dict[Feats, torch.Tensor]:
    """Obfuscate a trace in one pass; return only the executed trace."""

    _, _, _, _, _, _, _, X_obs = policy_rollout_single_pass(
        obs,
        X,
        detach_period=100,
        sample=sample,
    )
    return X_obs


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

    exec_state = TraceExecState(
        {
            Feats.TIMES: Xb[Feats.TIMES].clone(),
            Feats.DIRS: Xb[Feats.DIRS].clone(),
            Feats.PADDING: Xb[Feats.PADDING].clone(),
        },
        time_step_s=obs_.time_step,
    )

    base_n = (Xb[Feats.DIRS] != 0).sum(dim=1).long()
    pad_n = torch.zeros((bs,), device=device, dtype=torch.long)

    hobs = None

    log_ps_l: list[torch.Tensor] = []
    sel_probs_l: list[torch.Tensor] = []
    values_actor_l: list[torch.Tensor] = []
    ent_sel_l: list[torch.Tensor] = []
    ent_cond_l: list[torch.Tensor] = []
    act_time_bins_l: list[torch.Tensor] = []
    actions_l: dict[Actions, list[torch.Tensor]] | None = None
    fd_steps: dict[Feats, list[torch.Tensor]] | None = (
        {f: [] for f in obs_.features} if record_policy else None
    )

    while True:
        fd_t_full = streamer.step()

        # TIME_BINS are int bins; -1 means invalid.
        # active_cpu is on stream_device (CPU when X is on GPU to avoid syncs).
        active_cpu = (fd_t_full[Feats.TIME_BINS] >= 0).squeeze(1)
        if int(active_cpu.sum().item()) == 0:
            break

        # active_gpu is on device (GPU when X is on GPU).
        active_gpu = active_cpu.to(device)

        if record_policy:
            assert fd_steps is not None
            for f in obs_.features:
                fd_steps[f].append(fd_t_full[f].to(device))

        # Replace -1 with 0 for invalid entries (agent expects 0 for padding)
        fd_t_active = {
            k: torch.where(
                fd_t_full[k][active_cpu] >= 0,
                fd_t_full[k][active_cpu],
                torch.zeros_like(fd_t_full[k][active_cpu]),
            ).to(device)
            for k in obs_.features
        }

        h_active = _hidden_w_mask(hobs, active_gpu)
        act_time_bins_a, actions_a, log_ps_a, sel_probs_a, values_a, ent_a, h_active = (
            obs_.act_step(fd_t_active, h_active, sample=sample)
        )

        hobs = _hidden_w_mask(hobs, active_gpu, h_active)

        exec_state.step(
            trace_idx=torch.where(active_gpu)[0],
            times=act_time_bins_a,
            actions=actions_a,
        )

        if (
            Actions.DELAY_BINS in actions_a
            and (actions_a[Actions.DELAY_BINS] > 0).any()
        ):
            delay_idx = torch.where(active_cpu)[0]
            shift_full = torch.zeros((bs,), device=stream_device, dtype=torch.long)
            shift_full[delay_idx] = (
                actions_a[Actions.DELAY_BINS].squeeze(1).to(stream_device).long()
            )

            start_bins_full = torch.zeros((bs,), device=stream_device, dtype=torch.long)
            start_bins_full[delay_idx] = fd_t_full[Feats.TIME_BINS][active_cpu].squeeze(
                1
            ) + fd_t_full[Feats.Dt_BINS][active_cpu].squeeze(1)
            streamer.apply_delay_bins(start_bins_full, shift_full)

        if record_policy:
            # Scatter back to full batch. -1 = inactive.
            act_time_bins_t = torch.full((bs, 1), -1, device=device, dtype=torch.long)
            act_time_bins_t[active_gpu] = act_time_bins_a

            log_ps_t = torch.zeros((bs, 1), device=device)
            log_ps_t[active_gpu] = log_ps_a

            values_t = torch.zeros((bs, 1), device=device)
            values_t[active_gpu] = values_a

            sel_probs_t = torch.zeros((bs, 1, sel_probs_a.shape[-1]), device=device)
            sel_probs_t[active_gpu] = sel_probs_a

            ent_sel_t = torch.zeros((bs, 1), device=device)
            ent_sel_t[active_gpu] = ent_a["selection_entropy"]

            ent_cond_t = torch.zeros((bs, 1), device=device)
            ent_cond_t[active_gpu] = ent_a["conditional_entropy"]

            if actions_l is None:
                actions_l = {k: [] for k in actions_a.keys()}

            for k in actions_l.keys():
                a_full = torch.zeros((bs, 1), device=device, dtype=actions_a[k].dtype)
                a_full[active_gpu] = actions_a[k]
                actions_l[k].append(a_full)

            act_time_bins_l.append(act_time_bins_t)
            log_ps_l.append(log_ps_t)
            sel_probs_l.append(sel_probs_t)
            values_actor_l.append(values_t)
            ent_sel_l.append(ent_sel_t)
            ent_cond_l.append(ent_cond_t)

        if max_packets is not None:
            dev_idx = torch.where(active_gpu)[0]
            active_n = dev_idx.shape[0]
            inc = torch.zeros((active_n,), device=device, dtype=torch.long)
            if Actions.SEND_COUNT_UP in actions_a:
                inc = inc + actions_a[Actions.SEND_COUNT_UP].squeeze(1)
            if Actions.SEND_COUNT_DOWN in actions_a:
                inc = inc + actions_a[Actions.SEND_COUNT_DOWN].squeeze(1)
            pad_n[dev_idx] = pad_n[dev_idx] + inc
            if ((base_n + pad_n) >= int(max_packets)).all():
                break

    X_obs = exec_state.finalize()
    if max_packets is not None:
        X_obs = {k: v[:, : int(max_packets)] for k, v in X_obs.items()}

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
