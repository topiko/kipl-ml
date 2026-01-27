import time

import torch
from torch import nn

from kipl_ml.logging.logger import get_logger
from kipl_ml.rl.action import send_exec
from kipl_ml.rl.enums import Actions
from kipl_ml.rl.observation import get_window_feature_dict
from kipl_ml.trace.enums import Feats
from kipl_ml.trace.features import FeatureTrs

logger = get_logger(__name__)


def get_rewards(
    action_times: torch.Tensor,
    X: dict[Feats, torch.Tensor],
    y: torch.Tensor,
    disc_logits: torch.Tensor,
    seq_lens: torch.Tensor,
    reward_scales: dict[str, float] = {"clf_scale": 1.0, "padding_scale": 1.0},
) -> dict[str, torch.Tensor]:
    N = disc_logits.shape[1]

    bs, T = action_times.shape

    # (B, T)
    rewards: dict[str, torch.Tensor] = {
        k.replace("_scale", ""): torch.zeros_like(action_times) for k in reward_scales
    }

    # (B, N)
    times = X[Feats.TIMES]
    padding = X[Feats.PADDING].bool()

    # (B, N, C)
    probs = nn.functional.softmax(disc_logits, dim=-1)

    # (B, N)
    target_probs = probs.gather(2, y.unsqueeze(1).expand(-1, N).unsqueeze(-1)).squeeze(
        -1
    )

    for i in range(T):
        # (B, 1)
        t0 = action_times[:, i].unsqueeze(1)

        # (B, 1)
        if i + 1 == T:
            t1 = torch.full_like(t0, torch.inf)
        else:
            t1 = action_times[:, i + 1].unsqueeze(1)

        t0 = t0.nan_to_num(nan=torch.inf)
        t1 = t1.nan_to_num(nan=torch.inf, posinf=torch.inf)

        # From the last action we take rewards all the way to end of times.
        mask = (times >= t0) & (times < t1)

        # (B, )
        npad = (padding & mask).sum(dim=1).float()
        rewards["padding"][:, i] -= npad * reward_scales["padding_scale"]

        # Count the "clf reward" only from the normal packets.
        p_lvl = 0.1
        float_mltp = (mask & ~padding).float()
        nnormal = float_mltp.sum(dim=1)
        mean_p = torch.where(
            nnormal > 0,
            ((p_lvl - target_probs) * (mask & ~padding).float()).sum(dim=1) / nnormal,
            0,
        )

        #
        normal_mask = nnormal > 0
        rewards["clf"][normal_mask, i] += (
            mean_p[normal_mask] * reward_scales["clf_scale"]
        )

    return rewards


def rollout(
    obs: nn.Module,
    critic: nn.Module,
    disc: nn.Module,
    X: dict[Feats, torch.Tensor],
    y: torch.Tensor,
    disc_league: list[tuple[int, nn.Module.state_dict]],
    disc_features: FeatureTrs | None = None,
    detach_period: int = 20,
    reward_scales: dict[str, float] | None = None,
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
    rewards = None
    device = y.device

    t0 = time.time()
    if X[Feats.TIMES].isnan().any():
        raise ValueError("NaN in times feature")

    fd = get_window_feature_dict(
        X, obs.time_step, obs.max_silence_s, features=obs.features
    )
    if X[Feats.TIMES].isnan().any():
        raise ValueError("NaN in times feature")

    t1 = time.time()

    # We need the seq. lens in forward.
    action_seq_lens = fd.pop(Feats.SEQ_LENS)
    L = action_seq_lens.max().item()

    act_times, actions, log_ps, sel_probs, _, entropies, h = obs.act(
        fd, hobs, h_detach_period=detach_period, seq_lens=action_seq_lens
    )

    if h is not None:
        h_norm = h[0].norm(2, dim=-1).max().item()
        c_norm = h[1].norm(2, dim=-1).max().item()
        if h_norm > 100 or c_norm > 400:
            print("Huge hidden/cell:", h_norm, c_norm)

    t2 = time.time()
    Xobs = send_exec(X, act_times, actions)

    t3 = time.time()

    if reward_scales is not None:
        current_disc_state = {
            k: v.detach().clone() for k, v in disc.state_dict().items()
        }
        rewards_l = []
        league_values_l = []
        seq_lens = (Xobs[Feats.DIRS] != 0).sum(dim=1).long().cpu()

        div_lens = action_seq_lens[:, None]
        # div_lens = torch.stack(
        #     [
        #         fpad(torch.arange(l, 0, -1), (0, L - l), value=1)
        #         for l in action_seq_lens
        #     ],
        #     dim=0,
        # ).to(device)

        if disc_features is not None:
            X_ = disc_features.transform_batch(Xobs)
        else:
            X_ = Xobs

        for disc_id, state_d in disc_league:
            disc.load_state_dict({k: v.to(device) for k, v in state_d.items()})

            hdisc = None
            disc.eval()
            with torch.no_grad():
                logits, hdisc = disc.pack_and_forward(X_, hdisc, seq_lens)

            rewards_ = get_rewards(
                act_times, Xobs, y, logits, seq_lens, reward_scales=reward_scales
            )

            # Rather use the "reward contribution" than the actual rewards to
            # remove the dep. on seq_len.
            rewards_ = {k: v / div_lens for k, v in rewards_.items()}

            rewards_l.append(rewards_)

            fd[Feats.DISC_ID] = torch.full_like(fd[critic.features[0]], disc_id)
            fd[Feats.LABEL] = y.unsqueeze(1).repeat(1, L)

            league_values_ = critic(
                fd, None, h_detach_period=detach_period, seq_lens=seq_lens
            )[0][Feats.STATE_VALUE]

            league_values_l.append(league_values_)

        # Rewards from different disc. checkpoints.
        rewards = {
            k: torch.stack([r[k] for r in rewards_l], dim=0)
            for k in rewards_l[0].keys()
        }

        # Make the values tensor
        league_values = torch.stack(league_values_l, dim=0)

        # Make sure correct state is restored.
        disc.load_state_dict(current_disc_state)

    t4 = time.time()

    timings = {
        "Extr": t1 - t0,
        "Act": t2 - t1,
        "Exec": t3 - t2,
        "Disc+rew": t4 - t3,
    }

    return (
        log_ps,
        sel_probs,
        league_values,
        rewards,
        entropies,
        act_times,
        actions,
        Xobs,
        fd,
    )
