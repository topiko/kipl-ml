from __future__ import annotations

from collections.abc import Callable

import torch
from torch import nn

from kipl_ml.data.utils import UPLOAD
from kipl_ml.rl.action import TraceExecState, send_exec
from kipl_ml.rl.enums import Actions
from kipl_ml.rl.observation import WindowFeatureStreamer, get_window_feature_dict
from kipl_ml.trace.enums import Feats


def _ensure_trace_dict(
    X: dict[Feats, torch.Tensor],
) -> dict[Feats, torch.Tensor]:
    if Feats.TIMES not in X or Feats.DIRS not in X:
        raise ValueError("X must contain TIMES and DIRS")

    X2 = {k: v.clone() for k, v in X.items()}
    if Feats.PADDING not in X2:
        X2[Feats.PADDING] = torch.zeros_like(X2[Feats.TIMES])
    return X2


def _apply_extend_end_inplace(X: dict[Feats, torch.Tensor], extend_end_s: float) -> None:
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
    obs: nn.Module,
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

    Xb = _ensure_trace_dict(X)
    _apply_extend_end_inplace(Xb, extend_end_s)

    fd = get_window_feature_dict(
        Xb,
        float(obs.time_step),
        float(obs.max_silence_s),
        features=list(obs.features),
        extend_end_s=0,
    )
    action_seq_lens = fd.pop(Feats.SEQ_LENS)

    act_times, actions, log_ps, sel_probs, values_actor, entropies, h = obs.act(
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
    obs: nn.Module,
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

    Xb = _ensure_trace_dict(X)
    _apply_extend_end_inplace(Xb, extend_end_s)

    device = Xb[Feats.TIMES].device
    bs = int(Xb[Feats.TIMES].shape[0])

    streamer = WindowFeatureStreamer(
        Xb,
        dt=float(obs.time_step),
        max_silence_s=float(obs.max_silence_s),
        features=list(obs.features),
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
    act_step_fn: Callable | None = getattr(obs, "act_step", None)

    log_ps_l: list[torch.Tensor] = []
    sel_probs_l: list[torch.Tensor] = []
    values_actor_l: list[torch.Tensor] = []
    ent_sel_l: list[torch.Tensor] = []
    ent_cond_l: list[torch.Tensor] = []
    act_times_l: list[torch.Tensor] = []
    actions_l: dict[Actions, list[torch.Tensor]] | None = None
    fd_steps: dict[Feats, list[torch.Tensor]] = {f: [] for f in obs.features}

    while True:
        fd_t_full = streamer.step()
        for f in obs.features:
            fd_steps[f].append(fd_t_full[f])

        active = fd_t_full[Feats.TIMES].isfinite().squeeze(1)
        if active.sum() == 0:
            break

        fd_t_active = {
            k: torch.where(
                fd_t_full[k][active].isfinite(),
                fd_t_full[k][active],
                torch.zeros((int(active.sum().item()), 1), device=device),
            )
            for k in obs.features
        }

        h_active = _hidden_w_mask(hobs, active)
        if act_step_fn is not None:
            act_times_a, actions_a, log_ps_a, sel_probs_a, values_a, ent_a, h_active = (
                act_step_fn(fd_t_active, h_active, sample=sample)
            )
        else:
            act_times_a, actions_a, log_ps_a, sel_probs_a, values_a, ent_a, h_active = (
                obs.act(
                    fd_t_active,
                    h_active,
                    h_detach_period=None,
                    seq_lens=torch.ones((int(active.sum().item()),), device=device).long(),
                    sample=sample,
                )
            )

        hobs = _hidden_w_mask(hobs, active, h_active)

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

        actions_t: dict[Actions, torch.Tensor] = {}
        for k in actions_l.keys():
            a_full = torch.zeros((bs, 1), device=device, dtype=actions_a[k].dtype)
            a_full[active] = actions_a[k]
            actions_t[k] = a_full
            actions_l[k].append(a_full)

        act_times_l.append(act_times_t)
        log_ps_l.append(log_ps_t)
        sel_probs_l.append(sel_probs_t)
        values_actor_l.append(values_t)
        ent_sel_l.append(ent_sel_t)
        ent_cond_l.append(ent_cond_t)

        # Record action execution.
        exec_state.step(trace_idx=torch.where(active)[0], times=act_times_a, actions=actions_a)

        # Apply delay to future windows.
        if Actions.DELAY in actions_a and (actions_a[Actions.DELAY] > 0).any():
            delay_full = torch.zeros((bs, 1), device=device)
            delay_full[active] = actions_a[Actions.DELAY]
            start_full = torch.full((bs, 1), torch.nan, device=device)
            start_full[active] = act_times_a
            streamer.apply_delay(start_full, delay_full)

        # Optional early stop.
        if max_packets is not None:
            pad_n = pad_n + (
                actions_t.get(Actions.SEND_COUNT_UP, torch.zeros_like(act_times_t)).squeeze(1).to(torch.long)
                + actions_t.get(Actions.SEND_COUNT_DOWN, torch.zeros_like(act_times_t)).squeeze(1).to(torch.long)
            )
            if ((base_n + pad_n) >= int(max_packets)).all():
                break

    if actions_l is None:
        raise ValueError("No actions produced")

    fd = {f: torch.cat(vs, dim=1) for f, vs in fd_steps.items()}
    action_seq_lens = fd[Feats.TIMES].isfinite().sum(dim=1).long()
    fd[Feats.SEQ_LENS] = action_seq_lens

    act_times = torch.cat(act_times_l, dim=1)
    log_ps = torch.cat(log_ps_l, dim=1)
    sel_probs = torch.cat(sel_probs_l, dim=1)
    values_actor = torch.cat(values_actor_l, dim=1)
    entropies = {
        "selection_entropy": torch.cat(ent_sel_l, dim=1),
        "conditional_entropy": torch.cat(ent_cond_l, dim=1),
    }
    actions = {k: torch.cat(vs, dim=1) for k, vs in actions_l.items()}
    Xobs = exec_state.finalize()
    if max_packets is not None:
        Xobs = {k: v[:, : int(max_packets)] for k, v in Xobs.items()}

    return fd, act_times, actions, log_ps, sel_probs, values_actor, entropies, Xobs
