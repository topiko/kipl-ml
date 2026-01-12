import time

import torch
from torch import nn

from kipl_ml.logging.logger import get_logger
from kipl_ml.rl.action import send_exec
from kipl_ml.rl.enums import Actions
from kipl_ml.rl.observation import get_window_feature_dict
from kipl_ml.trace.enums import Feats

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
    disc: nn.Module,
    X: dict[Feats, torch.Tensor],
    y: torch.Tensor,
    detach_period: int = 20,
    reward_scales: dict[str, float] | None = None,
    disc_state_dicts: list | None = None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    dict[str, torch.Tensor] | None,
    torch.Tensor,
    torch.Tensor,
    dict[Actions, torch.Tensor],
    dict[Feats, torch.Tensor],
    dict[Feats, torch.Tensor],
]:
    hdisc = None
    hobs = None

    rewards = None

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
    seq_lens = fd.pop(Feats.SEQ_LENS)

    act_times, actions, log_ps, values, entropies, h = obs.act(
        fd, hobs, h_detach_period=detach_period, seq_lens=seq_lens
    )

    if h is not None:
        h_norm = h[0].norm(2, dim=-1).max().item()
        c_norm = h[1].norm(2, dim=-1).max().item()
        if h_norm > 100 or c_norm > 10000:
            print("Huge hidden/cell:", h_norm, c_norm)

    t2 = time.time()
    Xobs = send_exec(X, act_times, actions)

    t3 = time.time()

    if (disc_state_dicts is None) or (len(disc_state_dicts) == 0):
        disc_state_dicts = [disc.state_dict()]
    else:
        disc_state_dicts.append(disc.state_dict())

    if reward_scales is not None:
        rewards_l = []
        seq_lens = (Xobs[Feats.DIRS] != 0).sum(dim=1).long().cpu()
        for state_d in disc_state_dicts:
            disc.load_state_dict(state_d)

            disc.eval()
            with torch.no_grad():
                logits, hdisc = disc.pack_and_forward(Xobs, hdisc, seq_lens)

            rewards_ = get_rewards(
                act_times, Xobs, y, logits, seq_lens, reward_scales=reward_scales
            )

            rewards_l.append(rewards_)

        # Average the rewards from different disc. checkpoints.
        rewards = {
            k: torch.stack([r[k] for r in rewards_l], dim=-1).mean(dim=-1)
            for k in rewards_l[0].keys()
        }

        # Ensure we are back to the last disc.
        disc.load_state_dict(disc_state_dicts[-1])

    t4 = time.time()

    timings = {
        "Extr": t1 - t0,
        "Act": t2 - t1,
        "Exec": t3 - t2,
        "Disc+rew": t4 - t3,
    }

    return (
        log_ps,
        values,
        rewards,
        entropies,
        act_times,
        actions,
        Xobs,
        fd,
    )
