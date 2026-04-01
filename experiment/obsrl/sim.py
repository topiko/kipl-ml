import torch
from torch import nn

from kipl_ml.rl.enums import Actions
from kipl_ml.rl.simulate import policy_rollout_single_pass, policy_rollout_streaming
from kipl_ml.rl.utils import fill_after_seq_end
from kipl_ml.trace.enums import Feats
from kipl_ml.trace.features import FeatureTrs
from kipl_ml.utils.time import _boundary_time_to_bin_idx, _time_to_bin_idx


def get_rewards(
    action_times: torch.Tensor,
    actions: dict[Actions, torch.Tensor],
    X_obs: dict[Feats, torch.Tensor],
    X_raw: dict[Feats, torch.Tensor] | None,
    obs_dt_s: float | None,
    X_disc: dict[Feats, torch.Tensor],
    y: torch.Tensor,
    disc_logits: torch.Tensor,
    packet_seq_lens: torch.Tensor,
    disc_seq_lens: torch.Tensor,
    reward_scales: dict[str, float],
    tam_dt_s: float | None = None,
) -> dict[str, torch.Tensor]:
    """Compute rewards for TAM-based discriminator.

    Args:
        action_times: (B, T) int bins
        actions: Dict of action tensors
        X_obs: Executed trace
        X_raw: Original trace (for delay penalty)
        obs_dt_s: Observation time step
        X_disc: Discriminator input features
        y: Target labels
        disc_logits: (B, N, C) discriminator logits
        packet_seq_lens: (B,) packet sequence lengths
        disc_seq_lens: (B,) discriminator sequence lengths
        reward_scales: Dict of reward scale factors
        tam_dt_s: TAM bin width in seconds (required)

    Returns:
        Dict of reward tensors (B, T)
    """
    if tam_dt_s is None or tam_dt_s <= 0:
        raise ValueError("tam_dt_s must be provided for TAM reward mapping")

    # Enforce that TAM bins and obs action times use the same binning scheme.
    # This is critical for correct reward computation: action_times are in
    # obs_dt_s bins, and TAM bins are in tam_dt_s bins. If they differ,
    # the searchsorted logic would map to wrong intervals.
    if obs_dt_s is not None and abs(obs_dt_s - tam_dt_s) > 1e-9:
        raise ValueError(
            f"obs_dt_s ({obs_dt_s}) must equal tam_dt_s ({tam_dt_s}) for reward computation. "
            f"Action times are binned with dt={obs_dt_s}, but TAM bins use dt={tam_dt_s}."
        )

    N = disc_logits.shape[1]
    bs, T = action_times.shape

    # Convert int bin action times to float seconds for reward computation.
    if obs_dt_s is not None and action_times.dtype in (torch.long, torch.int):
        action_times_f = action_times.float() * float(obs_dt_s)
        # Mark invalid bins (-1) as inf so searchsorted sorts them last.
        action_times_f = torch.where(
            action_times >= 0, action_times_f, torch.tensor(float("inf"))
        )
    else:
        action_times_f = action_times.float()

    rewards: dict[str, torch.Tensor] = {
        k.replace("_scale", ""): torch.zeros((bs, T), device=action_times.device)
        for k in reward_scales
    }

    # (B, N) - N is number of TAM bins, used for classifier reward
    times = X_obs[Feats.TIMES][:, :N]

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

    # Padding penalty: count actual padding packets per action interval.
    # Use ALL packets (packet_seq_lens), not just first N (which is TAM bins).
    # N is the number of TAM bins, not packets!
    n_packets = int(packet_seq_lens.max().item())
    times_all = X_obs[Feats.TIMES][:, :n_packets]
    padding_all = X_obs[Feats.PADDING][:, :n_packets].bool()

    # Make contiguous to avoid searchsorted warning.
    pkt_idx = (
        torch.searchsorted(
            action_times_f.contiguous(), times_all.contiguous(), right=True
        )
        - 1
    )
    pkt_valid_idx = (pkt_idx >= 0) & (pkt_idx < T)
    pkt_in_seq = (
        torch.arange(n_packets, device=times_all.device)[None, :]
        < packet_seq_lens.to(times_all.device)[:, None]
    )
    pkt_valid = pkt_valid_idx & pkt_in_seq
    pkt_idx_clamped = pkt_idx.clamp(0, T - 1)

    pad_w = (padding_all & pkt_valid).to(times_all.dtype)
    npad = torch.zeros(
        (bs, T), device=times_all.device, dtype=times_all.dtype
    ).scatter_add_(1, pkt_idx_clamped, pad_w)
    rewards["padding"] -= npad * reward_scales["padding_scale"]

    # TAM reward computation
    # (bs, N-1)
    disc_bins_full = X_disc.get(Feats.TAM_BINS, None)
    if disc_bins_full is None:
        # Backward-compat fallback: derive bins from times.
        disc_bins_full = _time_to_bin_idx(X_disc[Feats.TAM_TIMES], float(tam_dt_s))

    disc_bins = disc_bins_full[:, 1:].contiguous().to(torch.long)
    m = m[:, 1:]

    # (bs, T) int64, NaNs -> large bin so they sort last.
    boundaries_bins = torch.full(
        action_times_f.shape,
        int(1e18),
        device=action_times_f.device,
        dtype=torch.long,
    )
    m_fin = action_times_f.isfinite()
    if bool(m_fin.any().item()):
        boundaries_bins[m_fin] = _boundary_time_to_bin_idx(
            action_times_f[m_fin], float(tam_dt_s)
        )

    idxs = torch.searchsorted(boundaries_bins, disc_bins, right=True) - 1
    idxs = idxs.clamp(0, T - 1)

    # clf reward
    disc_seq_len_mask = (
        torch.arange(N, device=disc_logits.device)[None, :]
        < disc_seq_lens[:, None].to(disc_logits.device)
    )[:, 1:].float()

    # (bs, T)
    sum_ = torch.zeros((bs, T), device=times.device, dtype=times.dtype).scatter_add_(
        1, idxs, torch.ones_like(m) * disc_seq_len_mask
    )

    # (bs, N)

    clf_scale = reward_scales["clf_scale"]
    # We make the clf reward live between -1, 1...
    r_pkt = torch.clamp(-m, min=-1 / clf_scale, max=1 / clf_scale)
    # (bs, T)
    mp = torch.zeros((bs, T), device=times.device, dtype=times.dtype).scatter_add_(
        1, idxs, r_pkt * disc_seq_len_mask
    )

    mean_p = torch.where(sum_ > 0, mp / sum_, 0.0)
    rewards["clf"] += mean_p * clf_scale

    # Delay penalty: charge only for packets that actually get delayed.
    if (
        X_raw is not None
        and obs_dt_s is not None
        and obs_dt_s > 0
        and "delay_scale" in reward_scales
        and Actions.DELAY_BINS in actions
    ):
        delay = actions[Actions.DELAY_BINS]
        if delay.ndim == 3 and delay.shape[-1] == 1:
            delay = delay.squeeze(-1)
        delay_mask = (delay > 0) & (action_times >= 0)

        # (B, L) original packet bins; fill padding with +inf bin to preserve sort.
        dirs0 = X_raw[Feats.DIRS]
        m0 = dirs0 != 0
        t0 = X_raw[Feats.TIMES]
        t0_f = fill_after_seq_end(t0, m0, fill_val="max")
        pkt_bins = _time_to_bin_idx(t0_f, float(obs_dt_s))

        # (B, T) start bins for delay windows. action_times are already int bins.
        start_bins = action_times.to(torch.long)

        # Count occurrences per step via searchsorted on sorted pkt_bins.
        lo = torch.searchsorted(pkt_bins, start_bins, right=False)
        hi = torch.searchsorted(pkt_bins, start_bins, right=True)
        delayed_cnt = (hi - lo).float()

        rewards["delay"] -= (
            delayed_cnt * delay_mask.float() * reward_scales["delay_scale"]
        )

    # change in prob reward
    mask = mean_p != 0
    rewards["d_clf"] += torch.where(
        mask & mask.roll(1, dims=1),
        (
            mean_p.diff(dim=1, prepend=mean_p[:, :1].clone())
            / action_times_f.diff(
                dim=1,
                prepend=torch.ones((bs, 1), device=action_times_f.device)
                * float("-inf"),
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
        fd_critic[Feats.LABEL] = y.unsqueeze(1).repeat(
            1, int(fd[Feats.TIME_BINS].shape[1])
        )

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
    X_obs: dict[Feats, torch.Tensor],
    X_raw: dict[Feats, torch.Tensor] | None,
    obs_dt_s: float | None,
    y: torch.Tensor,
    act_times: torch.Tensor,
    actions: dict[Actions, torch.Tensor],
) -> dict[str, torch.Tensor]:
    device = y.device
    current_disc_state = {k: v.detach().clone() for k, v in disc.state_dict().items()}

    packet_seq_lens_gpu = (X_obs[Feats.DIRS] != 0).sum(dim=1).long()

    if disc_features is not None:
        X_disc = disc_features.transform_batch(X_obs)
    else:
        X_disc = X_obs

    disc_seq_lens = disc.seq_len_fun(X_disc).to("cpu")
    X_disc = {k: v[:, : disc_seq_lens.max()] for k, v in X_disc.items()}

    rewards_l = []
    for _, state_d in disc_league:
        if state_d is None:
            disc.load_state_dict(current_disc_state)
        else:
            disc.load_state_dict({k: v.to(device) for k, v in state_d.items()})

        hdisc = None
        disc.eval()
        with torch.no_grad():
            logits, hdisc = disc.pack_and_forward(X_disc, hdisc, disc_seq_lens)

        rewards_ = get_rewards(
            act_times,
            actions,
            X_obs,
            X_raw,
            obs_dt_s,
            X_disc,
            y,
            logits,
            packet_seq_lens_gpu,
            disc_seq_lens,
            reward_scales=reward_scales,
            tam_dt_s=float(disc.tam_dict.get("window_width_s", 0.0)),
        )
        rewards_l.append(rewards_)

    # (League, B, T)
    rewards = {
        k: torch.stack([r[k] for r in rewards_l], dim=0) for k in rewards_l[0].keys()
    }
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
    critic_detach_period = critic_detach_period or detach_period

    fd, act_times, actions, log_ps, sel_probs, values_actor, entropies, X_obs = (
        policy_rollout_single_pass(
            obs,
            X,
            detach_period=detach_period,
            sample=sample,
            extend_end_s=2.0,
        )
    )

    action_seq_lens = fd[Feats.SEQ_LENS]

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
            X_obs=X_obs,
            X_raw=X,
            obs_dt_s=float(obs.time_step),
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
        X_obs,
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

    critic_detach_period = critic_detach_period or detach_period
    fd, act_times, actions, log_ps, sel_probs, values_actor, entropies, X_obs = (
        policy_rollout_streaming(
            obs,
            X,
            sample=sample,
            extend_end_s=2.0,
            max_packets=None,
        )
    )

    action_seq_lens = fd[Feats.SEQ_LENS]

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
            X_obs=X_obs,
            X_raw=X,
            obs_dt_s=float(obs.time_step),
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
        X_obs,
        fd,
    )
