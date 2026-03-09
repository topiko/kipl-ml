import torch
from torch import nn

from kipl_ml.data.utils import UPLOAD
from kipl_ml.logging.logger import get_logger
from kipl_ml.models.trgen import _hidden_w_mask
from kipl_ml.rl.action import TraceExecState, send_exec
from kipl_ml.rl.enums import Actions
from kipl_ml.rl.observation import WindowFeatureStreamer, get_window_feature_dict
from kipl_ml.trace.enums import Feats
from kipl_ml.trace.features import FeatureTrs

logger = get_logger(__name__)


def get_rewards(
    action_times: torch.Tensor,
    actions: dict[Actions, torch.Tensor],
    Xobs: dict[Feats, torch.Tensor],
    X: dict[Feats, torch.Tensor],
    y: torch.Tensor,
    disc_logits: torch.Tensor,
    packet_seq_lens: torch.Tensor,
    disc_seq_lens: torch.Tensor,
    feat_mode: str,
    reward_scales: dict[str, float],
) -> dict[str, torch.Tensor]:
    # Shapes:
    # - action_times: (B, T)
    # - disc_logits: (B, N, C)
    # - seq_lens: (B,)
    N = disc_logits.shape[1]
    bs, T = action_times.shape

    rewards: dict[str, torch.Tensor] = {
        k.replace("_scale", ""): torch.zeros_like(action_times) for k in reward_scales
    }

    # (B, N)
    times = Xobs[Feats.TIMES][:, :N]
    padding = Xobs[Feats.PADDING][:, :N].bool()

    # (B, N, C)
    # probs = nn.functional.softmax(disc_logits, dim=-1)
    # (B, N)
    # target_probs = probs.gather(2, y[:, None, None].expand(-1, N, 1)).squeeze(-1)

    # (B, N, 1)
    target_idx = y[:, None, None].expand(-1, N, 1)

    # (B, N)
    target_logits = disc_logits.gather(2, target_idx).squeeze(-1)
    # (B, N, C)
    other_logits = disc_logits.scatter(2, target_idx, float("-inf"))
    # (B, N)
    rest_lse = torch.logsumexp(other_logits, dim=-1)

    # (B, N)
    m = target_logits - rest_lse

    boundaries = action_times.nan_to_num(nan=float("inf"))

    if feat_mode == "dir":
        # Assign each packet time to an action interval [t_i, t_{i+1}).
        # We treat NaNs in action_times as +inf, which makes the last finite action
        # cover the rest of the trace.
        # (B, N) in [0..T], then shift to [ -1 .. T-1 ]
        idx = torch.searchsorted(boundaries, times, right=True) - 1
        valid_idx = (idx >= 0) & (idx < T)

        # Ignore padded packets beyond seq_lens.
        pkt_valid = (
            torch.arange(N, device=times.device)[None, :]
            < packet_seq_lens.to(times.device)[:, None]
        )
        valid = valid_idx & pkt_valid

        idx_clamped = idx.clamp(0, T - 1)

        # Padding penalty: count padding packets per action interval.
        # =============================================
        pad_w = (padding & valid).to(times.dtype)
        npad = torch.zeros(
            (bs, T), device=times.device, dtype=times.dtype
        ).scatter_add_(1, idx_clamped, pad_w)
        rewards["padding"] -= npad * reward_scales["padding_scale"]

        # Classifier reward: mean over normal packets per interval.
        # =============================================
        normal_w = ((~padding) & valid).to(times.dtype)
        normal_cnt = torch.zeros(
            (bs, T), device=times.device, dtype=times.dtype
        ).scatter_add_(1, idx_clamped, normal_w)

        r_pkt = torch.clamp(-m, min=-10, max=10.0)
        normal_sum = torch.zeros(
            (bs, T), device=times.device, dtype=times.dtype
        ).scatter_add_(1, idx_clamped, r_pkt * normal_w)
        mean_p = torch.where(normal_cnt > 0, normal_sum / normal_cnt, 0.0)
        rewards["clf"] += mean_p * reward_scales["clf_scale"]

    elif feat_mode == "tam":
        # (bs, T)
        disc_times = X[Feats.TAM_TIMES][:, 1:].contiguous()
        m = m[:, 1:]

        idxs = torch.searchsorted(boundaries, disc_times, right=True) - 1
        idxs = idxs.clamp(0, T - 1)

        # padding
        # =============================================
        rewards["padding"] -= (
            actions[Actions.SEND_COUNT_DOWN] * reward_scales["padding_scale"]
        )
        rewards["padding"] -= (
            actions[Actions.SEND_COUNT_UP] * reward_scales["padding_scale"]
        )

        # clf
        # =============================================
        # (bs, N)
        disc_seq_len_mask = (
            torch.arange(N, device=disc_logits.device)[None, :]
            < disc_seq_lens[:, None].to(disc_logits.device)
        )[:, 1:].float()

        # (bs, T)
        sum_ = torch.zeros(
            (bs, T), device=times.device, dtype=times.dtype
        ).scatter_add_(1, idxs, torch.ones_like(m) * disc_seq_len_mask)

        # (bs, N)
        r_pkt = torch.clamp(-m, min=-10, max=10.0)
        # (bs, T)
        mp = torch.zeros((bs, T), device=times.device, dtype=times.dtype).scatter_add_(
            1, idxs, r_pkt * disc_seq_len_mask
        )

        max_t_idxs = action_times.nan_to_num(nan=float("-inf")).max(dim=1).indices
        if (sum_.gather(1, max_t_idxs[:, None] - 1) < 1).any():
            raise ValueError(
                "There are action intervals w. no disc. clf score. "
                + f"Max. obs time: {action_times.nan_to_num(nan=0).max():.02f}. "
                + "Increase the TAM max_load_time_s (to increase the seq. "
                + "lens the disc sees) or dcrease the trace_len (to limit "
                + "the max len of action seq. lens)"
            )

        mean_p = torch.where(sum_ > 0, mp / sum_, 0.0)
        rewards["clf"] += mean_p * reward_scales["clf_scale"]
        # =============================================

    # change in prob reward
    mask = mean_p != 0
    rewards["d_clf"] += torch.where(
        mask & mask.roll(1, dims=1),
        (
            mean_p.diff(dim=1, prepend=mean_p[:, :1].clone())
            / action_times.diff(
                dim=1,
                prepend=torch.ones((bs, 1), device=action_times.device) * float("-inf"),
            )
        ).clamp(max=0)
        * reward_scales["d_clf_scale"],
        0,
    )
    return rewards


def compute_values(
    *,
    critic: nn.Module,
    critic_detach_period: int | None,
    fd: dict[Feats, torch.Tensor],
    action_seq_lens: torch.Tensor,
    y: torch.Tensor,
) -> torch.Tensor:
    fd_critic = fd
    if hasattr(critic, "features") and Feats.LABEL in critic.features:
        fd_critic = dict(fd)
        fd_critic[Feats.LABEL] = y.unsqueeze(1).repeat(1, int(fd[Feats.TIMES].shape[1]))

    return critic(
        fd_critic,
        None,
        h_detach_period=critic_detach_period,
        seq_lens=action_seq_lens,
    )[0][Feats.STATE_VALUE]


def compute_rewards_league(
    *,
    disc: nn.Module,
    disc_league: list[tuple[int, nn.Module.state_dict]],
    disc_features: FeatureTrs | None,
    reward_scales: dict[str, float],
    Xobs: dict[Feats, torch.Tensor],
    y: torch.Tensor,
    act_times: torch.Tensor,
    actions: dict[Actions, torch.Tensor],
) -> dict[str, torch.Tensor]:
    device = y.device
    current_disc_state = {k: v.detach().clone() for k, v in disc.state_dict().items()}

    packet_seq_lens_gpu = (Xobs[Feats.DIRS] != 0).sum(dim=1).long()

    if disc_features is not None:
        X_ = disc_features.transform_batch(Xobs)
    else:
        X_ = Xobs

    disc_seq_lens = disc.seq_len_fun(X_).to("cpu")
    X_ = {k: v[:, : disc_seq_lens.max()] for k, v in X_.items()}

    rewards_l = []
    for _, state_d in disc_league:
        if state_d is None:
            disc.load_state_dict(current_disc_state)
        else:
            disc.load_state_dict({k: v.to(device) for k, v in state_d.items()})

        hdisc = None
        disc.eval()
        with torch.no_grad():
            logits, hdisc = disc.pack_and_forward(X_, hdisc, disc_seq_lens)

        rewards_ = get_rewards(
            act_times,
            actions,
            Xobs,
            X_,
            y,
            logits,
            packet_seq_lens_gpu,
            disc_seq_lens,
            feat_mode=disc.feat_mode,
            reward_scales=reward_scales,
        )
        rewards_l.append(rewards_)

    rewards = {k: torch.stack([r[k] for r in rewards_l], dim=0) for k in rewards_l[0].keys()}
    disc.load_state_dict(current_disc_state)
    return rewards


def _rollout_single_pass(
    obs: nn.Module,
    critic: nn.Module | None,
    disc: nn.Module,
    X: dict[Feats, torch.Tensor],
    y: torch.Tensor,
    disc_league: list[tuple[int, nn.Module.state_dict]],
    disc_features: FeatureTrs | None = None,
    detach_period: int = 20,
    critic_detach_period: int | None = None,
    reward_scales: dict[str, float] | None = None,
    sample: bool = True,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    dict[str, torch.Tensor] | None,
    dict[str, torch.Tensor],
    torch.Tensor,
    dict[Actions, torch.Tensor],
    dict[Feats, torch.Tensor],
    dict[Feats, torch.Tensor],
]:
    hobs = None
    device = y.device
    critic_detach_period = critic_detach_period or detach_period

    if X[Feats.TIMES].isnan().any():
        raise ValueError("NaN in times feature")

    fd = get_window_feature_dict(
        X, obs.time_step, obs.max_silence_s, features=obs.features, extend_end_s=2
    )
    if X[Feats.TIMES].isnan().any():
        raise ValueError("NaN in times feature")

    # We need the seq. lens in forward.
    action_seq_lens = fd.pop(Feats.SEQ_LENS)
    L = action_seq_lens.max().item()

    act_times, actions, log_ps, sel_probs, values_actor, entropies, h = obs.act(
        fd,
        hobs,
        h_detach_period=detach_period,
        seq_lens=action_seq_lens,
        sample=sample,
    )

    if h is not None:
        h_norm = h[0].norm(2, dim=-1).max().item()
        c_norm = h[1].norm(2, dim=-1).max().item()
        if (c_norm > 10_000) or (h_norm > 14):
            logger.warning("Ill agent h state")
        if h_norm > 100 or c_norm > 400:
            h_var = h[0].var().item()
            c_var = h[1].var().item()
            if (c_var * 1e2 < c_norm) or (h_var * 1e2 < h_norm):
                logger.warning("Ill agent h state:")
                logger.warning("Huge hidden/cell: %.05f, %.05f", h_norm, c_norm)
                logger.warning("Vars hidden/cell: %.05f, %.05f", h_var, c_var)

    Xobs = send_exec(X, act_times, actions)

    values = values_actor
    if critic is not None:
        values = compute_values(
            critic=critic,
            critic_detach_period=critic_detach_period,
            fd=fd,
            action_seq_lens=action_seq_lens,
            y=y,
        )

    rewards = None
    if reward_scales is not None:
        rewards = compute_rewards_league(
            disc=disc,
            disc_league=disc_league,
            disc_features=disc_features,
            reward_scales=reward_scales,
            Xobs=Xobs,
            y=y,
            act_times=act_times,
            actions=actions,
        )

    return (
        log_ps,
        sel_probs,
        values,
        rewards,
        entropies,
        act_times,
        actions,
        Xobs,
        fd,
    )


def rollout(
    obs: nn.Module,
    critic: nn.Module | None,
    disc: nn.Module,
    X: dict[Feats, torch.Tensor],
    y: torch.Tensor,
    disc_league: list[tuple[int, nn.Module.state_dict]],
    disc_features: FeatureTrs | None = None,
    detach_period: int = 20,
    critic_detach_period: int | None = None,
    reward_scales: dict[str, float] | None = None,
    sample: bool = True,
):
    """Rollout entrypoint.

    Uses streaming rollout when delay is enabled on the agent.
    """

    if getattr(obs, "enable_delay", False):
        return _rollout_streaming(
            obs=obs,
            critic=critic,
            disc=disc,
            X=X,
            y=y,
            disc_league=disc_league,
            disc_features=disc_features,
            detach_period=detach_period,
            critic_detach_period=critic_detach_period,
            reward_scales=reward_scales,
            sample=sample,
        )

    return _rollout_single_pass(
        obs=obs,
        critic=critic,
        disc=disc,
        X=X,
        y=y,
        disc_league=disc_league,
        disc_features=disc_features,
        detach_period=detach_period,
        critic_detach_period=critic_detach_period,
        reward_scales=reward_scales,
        sample=sample,
    )




def _rollout_streaming(
    obs: nn.Module,
    critic: nn.Module | None,
    disc: nn.Module,
    X: dict[Feats, torch.Tensor],
    y: torch.Tensor,
    disc_league: list[tuple[int, nn.Module.state_dict]],
    disc_features: FeatureTrs | None = None,
    detach_period: int = 20,
    critic_detach_period: int | None = None,
    reward_scales: dict[str, float] | None = None,
    sample: bool = True,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    dict[str, torch.Tensor] | None,
    dict[str, torch.Tensor],
    torch.Tensor,
    dict[Actions, torch.Tensor],
    dict[Feats, torch.Tensor],
    dict[Feats, torch.Tensor],
]:
    """Discrete-time rollout where observation windows are generated on the fly."""

    device = y.device
    critic_detach_period = critic_detach_period or detach_period

    if X[Feats.TIMES].isnan().any():
        raise ValueError("NaN in times feature")

    # Apply the same end-extension as get_window_feature_dict(extend_end_s=2).
    X_base = {k: v.clone() for k, v in X.items()}
    bs0, L0 = X_base[Feats.DIRS].shape
    mask0 = X_base[Feats.DIRS] == 0
    seq_lens0 = (~mask0).sum(dim=1)
    col_idx0 = seq_lens0[seq_lens0 != L0]
    row_idx0 = torch.arange(bs0, device=seq_lens0.device)[seq_lens0 != L0]
    X_base[Feats.DIRS][row_idx0, col_idx0] = UPLOAD
    X_base[Feats.TIMES][mask0] += 2

    exec_state = TraceExecState(
        {
            Feats.TIMES: X_base[Feats.TIMES].clone(),
            Feats.DIRS: X_base[Feats.DIRS].clone(),
            Feats.PADDING: X_base.get(Feats.PADDING, torch.zeros_like(X_base[Feats.TIMES])).clone(),
        }
    )
    bs = y.shape[0]

    streamer = WindowFeatureStreamer(
        X_base,
        dt=obs.time_step,
        max_silence_s=obs.max_silence_s,
        features=obs.features,
        extend_end_s=0,
    )

    hobs = None

    log_ps_l: list[torch.Tensor] = []
    sel_probs_l: list[torch.Tensor] = []
    values_actor_l: list[torch.Tensor] = []
    ent_sel_l: list[torch.Tensor] = []
    ent_cond_l: list[torch.Tensor] = []
    act_times_l: list[torch.Tensor] = []
    actions_l: dict[Actions, list[torch.Tensor]] | None = None

    fd_steps: dict[Feats, list[torch.Tensor]] = {f: [] for f in obs.features}

    act_step_fn = getattr(obs, "act_step", None)

    max_steps = int((X_base[Feats.TIMES].max().item() / obs.time_step)) + 10_000
    step_i = 0
    while True:
        if step_i > max_steps:
            raise RuntimeError(
                "Exceeded max_steps in streaming rollout (possible infinite delay loop)"
            )
        fd_t_full = streamer.step()  # (B, 1) per feature
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
                    seq_lens=torch.ones(
                        (int(active.sum().item()),), device=device
                    ).long(),
                    sample=sample,
                )
            )

        hobs = _hidden_w_mask(hobs, active, h_active)

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

        # Record this step's action for final trace reconstruction.
        exec_state.step(
            trace_idx=torch.where(active)[0],
            times=act_times_a,
            actions=actions_a,
        )

        # If delay is selected, it shifts future observation windows.
        if Actions.DELAY in actions_a:
            if (actions_a[Actions.DELAY] > 0).any():
                delay_full = torch.zeros((bs, 1), device=device)
                delay_full[active] = actions_a[Actions.DELAY]
                streamer.apply_delay(delay_full)

        step_i += 1

    if actions_l is None:
        raise ValueError("No actions produced")

    fd = {f: torch.cat(vs, dim=1) for f, vs in fd_steps.items()}
    action_seq_lens = fd[Feats.TIMES].isnan().logical_not().sum(dim=1)
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

    rewards = None
    values = values_actor

    if critic is not None:
        values = compute_values(
            critic=critic,
            critic_detach_period=critic_detach_period,
            fd=fd,
            action_seq_lens=action_seq_lens,
            y=y,
        )

    if reward_scales is not None:
        rewards = compute_rewards_league(
            disc=disc,
            disc_league=disc_league,
            disc_features=disc_features,
            reward_scales=reward_scales,
            Xobs=Xobs,
            y=y,
            act_times=act_times,
            actions=actions,
        )

    return (
        log_ps,
        sel_probs,
        values,
        rewards,
        entropies,
        act_times,
        actions,
        Xobs,
        fd,
    )
