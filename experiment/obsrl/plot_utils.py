from functools import partial

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import torch
from omegaconf import DictConfig
from torch import nn
from tqdm import tqdm

from experiment.obsrl.sim import rollout
from experiment.obsrl.utils import (
    get_advantages,
)
from kipl_ml.data.wf_dataset import WFDataset, dict_to_device
from kipl_ml.logging.logger import TQDM_W, get_logger
from kipl_ml.rl.enums import StepActions
from kipl_ml.tools.plottr import (
    plot_actions,
    plot_obs_features,
    plot_rewards,
    plot_tam,
)
from kipl_ml.trace.features import Feats, FeatureTrs

logger = get_logger(__name__)


def _plot_set(
    cfg: DictConfig,
    ds: WFDataset,
    obs: nn.Module,
    critic: nn.Module | None,
    obs_features: FeatureTrs | None,
    disc_orig: nn.Module,
    disc_trained: nn.Module,
    disc_features: FeatureTrs,
    active_disc_league: list[torch.nn.Module.state_dict],
    weights: torch.Tensor,
    e: int,
    reward_scales: dict[str, float],
    device: torch.DeviceObjType,
    ntraces: int = 3,
):
    rng = np.random.default_rng(seed=42)

    idxs = rng.choice(len(ds), size=ntraces, replace=False)

    disc_orig.eval()
    disc_trained.eval()
    obs.eval()
    if critic is not None:
        critic.eval()

    # Build one batch for rollout.
    # NOTE: We keep the original per-trace dicts for the "orig disc" plot.
    X_orig_l: list[dict[Feats, torch.Tensor]] = []
    y_l: list[torch.Tensor] = []
    X_rollin_l: list[dict[Feats, torch.Tensor]] = []

    for idx in idxs:
        idx = int(idx)
        X_i, y_i = ds[idx]
        X_orig_l.append(X_i)
        y_l.append(y_i)
        if obs_features is not None:
            X_rollin_l.append(obs_features(X_i))
        else:
            X_rollin_l.append(X_i)

    # Stack feature dict batch.
    keys = list(X_rollin_l[0].keys())
    X_batch = {k: torch.stack([x[k] for x in X_rollin_l], dim=0) for k in keys}
    y_batch = torch.stack(y_l, dim=0)

    X_batch = dict_to_device(X_batch, device)
    y_batch = y_batch.to(device)

    # One rollout for the whole set.
    with torch.no_grad():
        (
            _,
            sel_probs,
            values,
            league_rewards,
            entropies,
            times,
            actions,
            X_obs,
            fd,
        ) = rollout(
            obs,
            critic,
            disc_trained,
            X_batch,
            y_batch,
            disc_league=active_disc_league,
            disc_features=disc_features,
            reward_scales=reward_scales,
        )

    action_seq_lens = fd[Feats.SEQ_LENS]
    G, advantages = get_advantages(league_rewards, values, action_seq_lens, cfg)

    # Pre-compute discriminator probabilities in batch to avoid per-figure forwards.
    # Orig traces
    Xd_batch = disc_features.transform_batch(X_batch)
    Xd_batch = dict_to_device(Xd_batch, device)
    logits_d, _ = disc_orig(Xd_batch)
    probs_d = nn.functional.softmax(logits_d, dim=-1)

    # Obs traces
    X_obs_d = disc_features.transform_batch(X_obs)
    X_obs_d = dict_to_device(X_obs_d, device)
    logits_o, _ = disc_trained(X_obs_d)
    probs_o = nn.functional.softmax(logits_o, dim=-1)

    logger.info("Generating figs:")
    with tqdm(list(enumerate(idxs)), desc="Gen figs", ncols=TQDM_W) as pbar:
        for batch_i, ds_idx in pbar:
            _plot_single(
                cfg=cfg,
                disc_orig=disc_orig,
                disc_trained=disc_trained,
                disc_features=disc_features,
                e=e,
                device=device,
                ds_idx=int(ds_idx),
                batch_i=int(batch_i),
                X_orig=X_orig_l[batch_i],
                y_orig=y_l[batch_i],
                sel_probs=sel_probs,
                probs_orig=probs_d,
                probs_obs=probs_o,
                values=values,
                league_rewards=league_rewards,
                entropies=entropies,
                times=times,
                actions=actions,
                X_obs=X_obs,
                fd=fd,
                G=G,
                advantages=advantages,
                weights=weights,
                obs_dt_s=float(obs.time_step_s),
            )


def _plot_single(
    cfg: DictConfig,
    disc_orig: nn.Module,
    disc_trained: nn.Module,
    disc_features: FeatureTrs,
    e: int,
    device: torch.DeviceObjType,
    ds_idx: int,
    batch_i: int,
    X_orig: dict[Feats, torch.Tensor],
    y_orig: torch.Tensor,
    sel_probs: torch.Tensor,
    probs_orig: torch.Tensor,
    probs_obs: torch.Tensor,
    values: torch.Tensor,
    league_rewards: dict[str, torch.Tensor] | None,
    entropies: dict[str, torch.Tensor],
    times: torch.Tensor,
    actions: list[StepActions],
    X_obs: dict[Feats, torch.Tensor],
    fd: dict[Feats, torch.Tensor],
    G: torch.Tensor,
    advantages: torch.Tensor,
    weights: torch.Tensor,
    obs_dt_s: float | None = None,
    show: bool = False,
):
    fig, (ax, ax_fd, ax_a, ax_a_e, ax_o, ax_rew, ax_mean_rew, ax_ret, ax_adv) = (
        plt.subplots(9, 1, figsize=(20, 15.0), sharex=True)
    )

    # Make rows 1, 2, and 4 easier to visually compare.
    # Use sharey() for compatibility across matplotlib versions.
    ax_fd.sharey(ax)
    ax_o.sharey(ax)
    ax_a.sharey(ax)

    plot_fn_ = partial(plot_tam, window_width=disc_orig.tam_dict["window_width_s"])

    # Orig disc on trace:
    # ========================================
    X_d = disc_features(X_orig)
    X_d = dict_to_device(X_d, device)
    y = y_orig.to(device).unsqueeze(-1)
    plot_fn_(
        X_d,
        ax=ax,
        cl_probs=probs_orig,
        idx=batch_i,
        true_class=y.item(),
    )
    ax.set_title(f"True class: {y.item()}")
    # ========================================

    # Obsfuscator features and actions:
    # ========================================
    # Plot obs inputs
    # dt_s is needed to convert int bins back to seconds for plotting.
    plot_obs_features(fd, idx=batch_i, ax=ax_fd, dt_s=obs_dt_s)
    ax_fd.set_title("Obs. features")

    # Convert action times (int bins) to seconds for line plots.
    if obs_dt_s is None:
        raise ValueError(
            "obs_dt_s is None; plotting requires obs_dt_s to convert bins to seconds"
        )

    # Cut sequence at policy sequence length first, then drop invalid bins.
    seq_len_i = fd[Feats.SEQ_LENS][batch_i].item()
    times_np = (times[batch_i, :seq_len_i] * obs_dt_s).cpu().numpy()

    # Plot actions
    plot_actions(
        actions[batch_i],
        ax=ax_a,
        dt_s=obs_dt_s,
    )

    # Plot entropy
    ax_entropy = ax_a_e
    for entropy, entropy_values in entropies.items():
        values_i = entropy_values[batch_i, :seq_len_i]
        ax_entropy.plot(
            times_np,
            values_i.squeeze().cpu().numpy(),
            "--",
            lw=2,
            label=entropy,
        )
    ax_entropy.set_ylabel("Action entropy")
    ax_entropy.legend(frameon=False, loc=1)

    # Plot selection probs
    ax_probs = ax_a_e.twinx()
    ax_probs.axes.spines["right"].set_visible(True)
    ax_probs.set_ylabel("Selection probs.")

    ax_probs.plot(
        times_np,
        sel_probs[batch_i, :seq_len_i].squeeze().cpu().numpy(),
        "-",
        lw=1,
    )

    ax_a.set_title("Actions, entropies")

    # Trained disc on obsfuscated trace (from batched rollout):
    # ========================================
    # Plot obsfuscated
    X_obs_i = disc_features.transform_batch(X_obs)
    plot_fn_(
        X_obs_i,
        ax=ax_o,
        cl_probs=probs_obs,
        idx=batch_i,
        true_class=y.item(),
    )
    ax_o.set_title("Obs. trace, disc trained")

    # Back to obsfuscator aspects:
    # ========================================
    # Plot rewards
    if league_rewards is None:
        raise ValueError("league_rewards is None; plotting requires reward_scales")
    # The current disc rewards are at latest idx.
    rewards = {k: v[-1, batch_i, :seq_len_i] for k, v in league_rewards.items()}
    plot_rewards(times_np, rewards, idx=batch_i, ax=ax_rew)
    ax_rew.legend(frameon=False, loc=2)

    # Plot the mean of all league rewards for reference.
    # v.shape = (league, B, T) -> (league, T)
    nleague = len(weights)
    for i in range(nleague):
        plot_rewards(
            times_np,
            {
                k: v[i, batch_i, :seq_len_i].squeeze(0).cpu()
                for k, v in league_rewards.items()
            },
            idx=None,
            ax=ax_mean_rew,
            only_sum=True,
            ls="-",
            marker="",
            lw=0.5,
            alpha=0.5,
            color="black",
        )

    mean_disc_rewards = (
        weights[:, None] * sum(league_rewards.values())[:, batch_i, :seq_len_i]
    ).sum(dim=0)
    ax_mean_rew.plot(
        times_np,
        mean_disc_rewards.cpu().numpy(),
        "-|",
        color="black",
        lw=2.0,
        label="Mean disc rew.",
    )

    # Plot returns
    # (league, B, T) -> (B, T) -> (T, )
    G_mean = (weights[:, None, None] * G).sum(dim=0)[batch_i, :seq_len_i]
    # Mean
    ax_ret.plot(
        times_np,
        G_mean.squeeze().cpu().numpy(),
        "k-",
        label="Return",
        lw=2,
    )
    # League cloud
    ax_ret.plot(
        times_np,
        G[:, batch_i, :seq_len_i].permute(1, 0).cpu().numpy(),
        "k-",
        alpha=0.5,
        lw=0.2,
    )
    # Current active
    ax_ret.plot(
        times_np,
        G[-1, batch_i, :seq_len_i].squeeze().cpu().numpy(),
        "k-",
        alpha=1.0,
        lw=0.5,
    )

    # Values
    values_i = values[batch_i, :seq_len_i]
    ax_ret.plot(
        times_np,
        values_i.squeeze().cpu().numpy(),
        "--",
        label="Values estim.",
        color="black",
        lw=1,
    )
    ax_ret.axhline(color="black", lw=0.5)
    ax_ret.set_ylabel("Return", color="k")
    ax_ret.legend(frameon=False, loc=1)
    ax_ret.axes.spines["bottom"].set_visible(False)

    # Advantages (league, T)
    advantages_i = advantages[:, batch_i, :seq_len_i]
    advantages_mean = (weights[:, None, None] * advantages).sum(dim=0)[
        batch_i, :seq_len_i
    ]

    # Mean
    ax_adv.plot(
        times_np,
        advantages_mean.cpu().numpy(),
        label="advantage_w_mean",
        color="green",
        lw=2,
    )
    # Cloud
    ax_adv.plot(
        times_np,
        advantages_i.permute(1, 0).cpu().numpy(),
        lw=0.5,
        alpha=0.2,
        color="green",
    )
    ax_adv.plot(
        times_np,
        advantages_i[-1, :].squeeze().cpu().numpy(),
        lw=0.5,
        alpha=1.0,
        color="green",
    )

    ax_rew.set_title("Rewards, returns... ")
    ax_adv.set_ylabel("Advantages", color="k")
    ax_adv.legend(frameon=False, loc=3)
    ax_adv.axhline(color="black", lw=0.5)

    ax_adv.set_xlabel("Time [s]")

    mlflow.log_figure(fig, f"trace_{ds_idx}_clf_epoch={e:03d}.png")

    if show:
        plt.show()
    plt.close()
