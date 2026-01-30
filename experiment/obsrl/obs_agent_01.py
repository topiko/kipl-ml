import copy
import os

import dotenv
import hydra
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import torch
from omegaconf import DictConfig
from torch import nn
from torch.utils.data import SubsetRandomSampler
from tqdm import tqdm

from experiment.obsrl.sim import rollout
from experiment.obsrl.utils import one_batch_train_disc
from experiment.trace_gan.data_utils import dl_
from experiment.utils import defence_builder
from kipl_ml.data.utils import Datasets, assets
from kipl_ml.data.wf_dataset import WFDataset, dict_to_device, get_train_valid_test
from kipl_ml.defences.base import NoDefence
from kipl_ml.defences.nndefs import RNNDef
from kipl_ml.logging.logger import TQDM_W, get_logger
from kipl_ml.metrics.clf_metrics import Accuracy
from kipl_ml.model_eval.evaluate import evaluate_model
from kipl_ml.models.trgen import AGENT1, CRITIC01
from kipl_ml.rl.advantages import get_gae, get_returns
from kipl_ml.tools.mlflow_utils import get_mlflow_expr
from kipl_ml.tools.plottr import (
    plot_actions,
    plot_obs_features,
    plot_rewards,
    plot_trace,
)
from kipl_ml.trace.features import Feats, FeatureTrs

logger = get_logger(__name__)
dotenv.load_dotenv()

WORKING_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_DIR_PATH = os.path.join(WORKING_DIR, "config")

N_SPLITS = 5
TEST_XV = 0
TARGET = assets.PAGE_LABEL
DATASET = Datasets.BIGENOUGH


def keymap(key: str) -> str:
    if "loss" in key:
        key = f"losses / {key}"
    if "entropy" in key:
        key = f"entropies / {key}"

    return key


def _plot_set(
    cfg: DictConfig,
    ds: WFDataset,
    obs: nn.Module,
    critic: nn.Module,
    obs_features: FeatureTrs,
    disc_orig: nn.Module,
    disc_trained: nn.Module,
    disc_features: FeatureTrs,
    disc_league_idx: int,
    e: int,
    reward_scales: dict[str, float],
    device: torch.DeviceObjType,
    ntraces: int = 3,
    max_len: int = 10_000,
):
    rng = np.random.default_rng(seed=42)

    idxs = rng.choice(len(ds), size=ntraces, replace=False)

    logger.info("Generating figs:")
    for i, idx in enumerate(idxs):
        logger.info(f"\t{i + 1:02d}/{ntraces:02d}")
        _plot_single(
            cfg=cfg,
            ds=ds,
            obs=obs,
            critic=critic,
            obs_features=obs_features,
            disc_orig=disc_orig,
            disc_trained=disc_trained,
            disc_features=disc_features,
            disc_league_idx=disc_league_idx,
            e=e,
            reward_scales=reward_scales,
            device=device,
            idx=idx,
            max_len=max_len,
        )


@torch.no_grad()
def _plot_single(
    cfg: DictConfig,
    ds: WFDataset,
    obs: nn.Module,
    critic: nn.Module,
    obs_features: FeatureTrs,
    disc_orig: nn.Module,
    disc_trained: nn.Module,
    disc_features: FeatureTrs,
    disc_league_idx: int,
    e: int,
    reward_scales: dict[str, float],
    device: torch.DeviceObjType,
    idx: int,
    max_len: int = 10_000,
):
    fig, (ax, ax_o, ax_fd, ax_a, ax_b) = plt.subplots(
        5, 1, figsize=(20, 12.0), sharex=True
    )

    disc_orig.eval()
    disc_trained.eval()
    obs.eval()

    def _unsqueeze(X: dict[Feats, torch.Tensor]) -> dict[Feats, torch.Tensor]:
        return {k: v.unsqueeze(0) for k, v in X.items()}

    def X_to_probs(clf: nn.Module, X: dict[Feats, torch.Tensor]) -> torch.Tensor:
        with torch.no_grad():
            logits, _ = clf(_unsqueeze(X))
            probs = nn.functional.softmax(logits, dim=-1)
        return probs

    # Orig disc on trace:
    # ========================================
    X_d, y = ds[idx]
    X_d = disc_features(X_d)
    X_d = dict_to_device(X_d, device)
    y = y.to(device).unsqueeze(-1)

    plot_trace(
        X_d,
        ax=ax,
        cl_probs=X_to_probs(disc_orig, X_d),
        true_class=y.item(),
    )
    ax.set_title(f"True class: {y.item()}")
    # ========================================

    # Trained disc on obsfuscated trace:
    # ========================================
    # Use the obsfuscator features when entering rollout,
    # the disc feats insiderollout make sure disc gets right set of features.
    X, y = ds[idx]
    X = _unsqueeze(obs_features(X))
    X = dict_to_device(X, device)
    y = y.to(device).unsqueeze(-1)

    _, sel_probs, state_values, league_rewards, entropies, times, actions, Xobs, fd = (
        rollout(
            obs,
            critic,
            disc_trained,
            X,
            y,
            disc_league=[(disc_league_idx, disc_trained.state_dict())],
            disc_features=disc_features,
            reward_scales=reward_scales,
        )
    )

    G, advantages = get_advantages(
        league_rewards, state_values, get_action_seq_lens(fd), cfg
    )

    Xobs = {k: v.squeeze(0) for k, v in Xobs.items()}
    mask = Xobs[Feats.DIRS] != 0
    if mask.sum() > max_len:
        logger.warning("Long seqs. detected -> truncating to %d.", max_len)
        Xobs[Feats.TIMES] = Xobs[Feats.TIMES][:max_len]
        Xobs[Feats.DIRS] = Xobs[Feats.DIRS][:max_len]
        Xobs[Feats.PADDING] = Xobs[Feats.PADDING][:max_len]

    # Plot obsfuscated
    plot_trace(
        Xobs,
        ax=ax_o,
        cl_probs=X_to_probs(disc_trained, disc_features(Xobs)),
        true_class=y.item(),
    )
    ax_o.set_title("Obs. trace, disc trained")

    # Plot obs inputs
    plot_obs_features(fd, ax=ax_fd)
    ax_fd.set_title("Obs. features")

    # Plot actions
    plot_actions(times, actions, ax=ax_a)

    # Plot entropy
    ax_entropy = ax_a.twinx()
    ax_entropy.axes.spines["right"].set_visible(True)
    for entropy, values in entropies.items():
        ax_entropy.plot(
            times.squeeze().cpu().numpy(),
            values.squeeze().cpu().numpy(),
            "--",
            lw=2,
            label=entropy,
        )
    ax_entropy.set_ylabel("Action entropy")
    ax_entropy.legend(frameon=False, loc=1)

    # Plot selection probs
    ax_probs = ax_a.twinx()
    ax_probs.axes.spines["right"].set_visible(True)
    ax_probs.spines["right"].set_position(("outward", 40))  # offset by 40 points
    ax_probs.set_ylabel("Selection probs.")

    ax_probs.plot(
        times.squeeze().cpu().numpy(),
        sel_probs.squeeze(0).cpu().numpy(),
        "-",
        lw=1,
    )

    ax_a.set_title("Actions, entropies")

    # Plot rewards
    rewards = {k: v.squeeze(0) for k, v in league_rewards.items()}
    plot_rewards(times, rewards, ax=ax_b)

    ax_advantages = ax_b.twinx()
    ax_advantages.axes.spines["right"].set_visible(True)
    ax_advantages.plot(
        times.squeeze().cpu().numpy(),
        advantages.squeeze().cpu().numpy(),
        label="advantage",
    )
    ax_b.set_title(f"Rewards, returns, disc_id={disc_league_idx}")
    ax_advantages.set_ylabel("Advantages", color="k")

    # Plot returns
    ax_r = ax_b.twinx()
    ax_r.axes.spines["right"].set_visible(True)
    ax_r.spines["right"].set_position(("outward", 40))  # offset by 40 points

    ax_r.plot(
        times.squeeze().cpu().numpy(),
        G.squeeze().cpu().numpy(),
        "k-",
        label="Return",
        lw=1,
    )
    ax_r.plot(
        times.squeeze().cpu().numpy(),
        state_values.squeeze().cpu().numpy(),
        "--",
        label="Values estim.",
        color="black",
        lw=1,
    )

    ax_r.set_ylabel("Return", color="k")

    ax_r.legend(frameon=False, loc=1)
    ax_b.legend(frameon=False, loc=2)
    ax_advantages.legend(frameon=False, loc=3)
    ax_b.set_xlabel("Time [s]")

    fig.canvas.draw()

    mlflow.log_figure(fig, f"trace_{idx}_clf_epoch={e:03d}.png")

    plt.close()


def get_advantages(
    rewards: torch.Tensor | dict[str, torch.Tensor],
    values: torch.Tensor,
    seq_lens: torch.Tensor,
    cfg: DictConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    if isinstance(rewards, dict):
        rewards = sum(rewards.values())

    # In case of "league rewards"
    if rewards.ndim == 3:
        if cfg.advantages.standardize:
            raise NotImplementedError("You should not standardize here for league")

        G_l = []
        advantages_l = []
        for i in range(rewards.shape[0]):
            G_, advantages_ = get_advantages(rewards[i], values[i], seq_lens, cfg)
            G_l.append(G_)
            advantages_l.append(advantages_)

        return torch.stack(G_l, dim=0), torch.stack(advantages_l, dim=0)

    values_detached = values.detach()

    if cfg.advantages.type in {"mc", "mc_w_bootstrap"}:
        # NOTE: this is not pure MC when bootstrap != 0.
        if cfg.advantages.type == "mc":
            bootstrap = None
        elif cfg.advantages.type == "mc_w_bootstrap":
            # Time-limit truncation bootstrap: treat end-of-trace as non-terminal.
            bootstrap = values_detached.gather(1, seq_lens[:, None] - 1).squeeze(1)
        else:
            raise KeyError()

        G = get_returns(
            rewards,
            seq_lens,
            gamma=cfg.discounting,
            bootstrap=bootstrap,
        )
        advantages = G - values_detached
    elif cfg.advantages.type == "gae":
        advantages = get_gae(
            rewards,
            values_detached,
            seq_lens,
            lambda_=cfg.advantages.lambda_,
            gamma=cfg.discounting,
        )
        G = advantages + values_detached
    else:
        raise NotImplementedError(f"Invalid advantage type: {cfg.advantages.type}")

    # Normalize advantages so that point in time on the seq does not matter.
    # (1, T)
    t = torch.arange(advantages.shape[1], device=advantages.device)[None, :]
    # (bs, T)
    gamma = cfg.discounting
    seq_lens_ = seq_lens.to(device=advantages.device)
    to_seq_end = (seq_lens_[:, None] - t).clamp_min(1).to(advantages.dtype)
    if abs(gamma - 1.0) < 1e-8:
        Z = to_seq_end
    else:
        Z = (1 - gamma**to_seq_end) / (1 - gamma)

    advantages /= Z + 1e-8

    if cfg.advantages.standardize:
        mask = make_time_mask(
            seq_lens.to(values.device), values.shape[1], device=values.device
        )
        mean, std = masked_mean_std(advantages, mask)
        advantages = (advantages) / std
        # Optional: keep padding at 0
        advantages = advantages * mask.to(advantages.dtype)

    return G, advantages


def assert_finite(name, x):
    raise_ = False
    if isinstance(x, (float, int)):
        if not np.isfinite(x):
            raise_ = True

    elif not torch.isfinite(x).all():
        raise_ = True
    elif torch.isnan(x).any():
        raise_ = True

    if raise_:
        print(f"Non-finite in {name}")
        raise ValueError


def _append_to_league(
    league: list[tuple[int, dict]], state_dict: dict
) -> list[tuple[int, nn.Module.state_dict]]:
    league.append(
        (len(league), {k: v.cpu() for k, v in copy.deepcopy(state_dict).items()})
    )

    return league


def _get_obs_def_dl(
    disc: nn.Module,
    obs: nn.Module,
    ds: WFDataset,
    n_packets: int,
    bs: int = 64,
    obs_league: list[nn.Module.state_dict] | None = None,
) -> WFDataset:
    # Set features the fetures:
    ds.feature_trs = FeatureTrs(feature_names=disc.features, n_packets=n_packets)

    # Set the defense:
    ds.defence = RNNDef(
        (0, 0),
        (40_000, 40_000),
        obs.to("cpu"),
        n_packets=n_packets,
        state_dicts=obs_league,
    )

    if ds.defence_aug != 0:
        raise ValueError("If def aug != 0 - you are reusing traces from previous runs")

    return dl_(ds, bs=bs, collate_fn=None, shuffle=False, nworkers=None)


def _restore_obs_def_ds(
    ds: WFDataset,
    feature_trs: FeatureTrs | None,
    obs: nn.Module,
    device: torch.DeviceObjType,
):
    # Restore no defence
    ds.defence = NoDefence(network_delay_millis=(0, 0), network_pps=(40_000, 40_000))
    # Restore no features.
    ds.feature_trs = feature_trs

    obs.to(device)


def valid_metrics(
    disc: nn.Module,
    obs: nn.Module,
    ds_valid: WFDataset,
    n_packets: int,
    key: str = "valid:obs_vs._disc",
    device: torch.DeviceObjType = "cpu",
    obs_league: list[nn.Module.state_dict] | None = None,
) -> dict[str, float]:
    orig_features_trs = ds_valid.feature_trs
    dl_valid = _get_obs_def_dl(
        disc=disc,
        obs=obs,
        ds=ds_valid,
        n_packets=n_packets,
        bs=32,
        obs_league=obs_league,
    )

    d = evaluate_model(
        disc, dl_valid, metrics=[Accuracy()], key=key, loss_fn=nn.CrossEntropyLoss()
    )

    _restore_obs_def_ds(ds_valid, orig_features_trs, obs, device)

    return d


def league_rewards2rewards(
    league_rewards: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    return {k: v.mean(dim=0) for k, v in league_rewards.items()}


def get_league_scores(
    league: list[tuple[int, nn.Module]],
    ds: WFDataset,
    obs: nn.Module,
    critic: nn.Module,
    obs_features: FeatureTrs,
    disc: nn.Module,
    disc_features: FeatureTrs,
    reward_scales: dict[str, float],
    device: torch.DeviceObjType,
    subset_indices: torch.Tensor,
) -> torch.Tensor:
    orig_features = ds.feature_trs

    # Set the defence and features:
    ds.feature_trs = obs_features

    sampler = SubsetRandomSampler(subset_indices)
    dl = dl_(ds, bs=64, collate_fn=None, shuffle=False, nworkers=None, sampler=sampler)
    obs.eval()

    with torch.no_grad():
        with tqdm(
            dl,
            desc="league scoring",
            ncols=TQDM_W,
        ) as pbar:
            rewards_l = []
            for X, y in pbar:
                X = dict_to_device(X, device)
                y = y.to(device)
                league_values, league_rewards, _, _, _, _, fd = rollout(
                    obs=obs,
                    critic=critic,
                    disc=disc,
                    X=X,
                    y=y,
                    disc_features=disc_features,
                    disc_league=league,
                    detach_period=500,
                    reward_scales=reward_scales,
                )[2:]

                action_seq_lens = get_action_seq_lens(fd)

                time_mask = make_time_mask(
                    action_seq_lens, fd[Feats.TIMES].shape[1], device=device
                )
                # (nleague, nbatch, ntimesteps) -> (nleague, nbatch) -> (nleague, 1)
                rewards = {
                    k: torch.tensor(
                        [masked_mean_std(v[i], time_mask)[0] for i in range(v.shape[0])]
                    )
                    for k, v in league_rewards.items()
                }
                rewards_ = sum(rewards.values())
                rewards_l.append(rewards_)

        league_scores = -torch.stack(rewards_l, dim=0).mean(dim=0).to(device)

    ds.feature_trs = orig_features

    return league_scores


def make_time_mask(seq_lens: torch.Tensor, L: int, device=None) -> torch.Tensor:
    device = device or seq_lens.device
    return torch.arange(L, device=device)[None, :] < seq_lens[:, None]  # (B, L) bool


def masked_mean(x: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    if isinstance(x, dict):
        return masked_mean(sum(x.values()), mask, eps)
    return masked_mean_std(x, mask, eps)[0]


def masked_mean_std(x: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8):
    m = mask.to(dtype=x.dtype)
    denom = m.sum().clamp(min=1.0)
    mean = (x * m).sum() / denom
    var = ((x - mean) * m).pow(2).sum() / denom
    std = (var + eps).sqrt()
    return mean, std


def get_action_seq_lens(fd: dict[Feats, torch.Tensor]) -> torch.Tensor:
    return fd[Feats.TIMES].isnan().logical_not().sum(dim=1)


def _get_optim(
    nn: nn.Module, lr: float, lr_rnn: float | None = None
) -> torch.optim.Optimizer:
    lr_rnn = lr_rnn or lr * 0.1
    rnn_params = list(nn.rnn.parameters())
    other_params = [p for n, p in nn.named_parameters() if not n.startswith("rnn.")]
    return torch.optim.Adam(
        [
            {"params": rnn_params, "lr": lr_rnn},
            {"params": other_params, "lr": lr},
        ]
    )


def get_active_league(
    active_league_idx: np.ndarray | None,
    league: list[tuple[int, nn.Module.state_dict]],
    ds: WFDataset,
    obs: nn.Module,
    critic: nn.Module,
    obs_feats: FeatureTrs,
    disc: nn.Module,
    disc_feats: FeatureTrs,
    reward_scales: dict[str, float],
    device: torch.DeviceObjType,
    league_size: int,
    league_update_frac: float,
    prune: bool = False,
) -> tuple[
    list[tuple[int, nn.Module.state_dict]],
    np.ndarray,
    list[nn.Module.state_dict],
    torch.Tensor,
]:
    league_scores = get_league_scores(
        league=league,
        ds=ds,
        obs=obs,
        critic=critic,
        obs_features=obs_feats,
        disc=disc,
        disc_features=disc_feats,
        reward_scales=reward_scales,
        device=device,
        subset_indices=torch.randint(0, len(ds), (1000,)).numpy(),
    )

    if active_league_idx is None:
        active_league_idx = np.random.choice(
            len(league), min(len(league), league_size), replace=False
        )

    if prune:
        logger.info("Pruning disc league.")
        val = league_scores.min().item()
        mask = league_scores > val
        league_scores = league_scores[mask]
        league = [l_ for i, l_ in enumerate(league) if mask[i]]

    if league_size == 1:
        active_league_idx = np.array([len(league) - 1])

    elif len(league) > league_size:
        scores = np.array(league_scores.cpu().numpy())
        probs = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
        probs /= probs.sum()

        active_league_idx = np.random.choice(
            len(league), league_size, p=probs, replace=False
        )
    else:
        active_league_idx = np.arange(len(league))

        # =======================================

    active_league = [league[i] for i in active_league_idx]

    if len(active_league_idx) > 1:
        weights = league_scores[active_league_idx]
        weights -= weights.min()
        weights /= weights.max()
        weights = torch.clamp(weights, 0.1, 1.0)
    else:
        weights = torch.tensor([1.0], device=device)

    weights /= weights.sum()

    logger.info("League scores:")
    for i, s in enumerate(league_scores):
        str_ = "           "
        if i in active_league_idx:
            w = weights[active_league_idx == i][0].item()
            str_ = f"* [w={w:.03f}]"

        logger.info(f"\t{i:4d} == {league[i][0]:4d}{str_} : {s:.4f}")

    return league, active_league_idx, active_league, weights


@hydra.main(config_path=CONFIG_DIR_PATH, config_name="config", version_base=None)
def main(cfg: DictConfig):
    experiment_name = "obsrl"
    experiment_id = get_mlflow_expr(experiment_name=experiment_name)
    mlflow.set_experiment(experiment_id=experiment_id)

    discriminator_orig = mlflow.pytorch.load_model(
        mlflow.get_logged_model("m-a1bec780ec314b95b4f0caac4dec5f46").model_uri,
        map_location="cpu",
    )
    # This one already somewhat trained for obsfuscation.
    discriminator = mlflow.pytorch.load_model(
        mlflow.get_logged_model("m-a1bec780ec314b95b4f0caac4dec5f46").model_uri,
        map_location="cpu",
    )
    # discriminator = mlflow.pytorch.load_model(
    #     mlflow.get_logged_model("m-0300713d9fff44489468fa1d40eed723").model_uri,
    #     map_location="cpu",
    # )
    discriminator.predict_ks = cfg.predict_ks

    feature_names = [Feats.DIRS, Feats.TIMES]

    ds_train, ds_valid, _ = get_train_valid_test(
        dataset=DATASET,
        label=assets.PAGE_LABEL,
        n_splits=N_SPLITS,
        test_xv=TEST_XV,
        random_state=42,
        feature_trs=FeatureTrs(feature_names=feature_names, n_packets=cfg.trace_len),
        defence_aug_valid=0,
        n_min_packets=cfg.min_packets_in_trace,
        **defence_builder.get_defence(cfg),
    )
    # ds_train.meta_df = ds_train.meta_df.sample(frac=0.5)
    # logger.warning("Using only subset of training data!")

    # Obs feature trs
    obs_features = ds_train.feature_trs

    # Discriminator features, w.o. limit on n_packets
    disc_feats = FeatureTrs(feature_names=discriminator.features, n_packets=None)

    dl_train = dl_(ds_train, bs=cfg.batch_size, collate_fn=None, shuffle=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    k = round(cfg.obs_max_silence_s / cfg.obs_time_step_s)
    obs_max_silence_s = k * cfg.obs_time_step_s
    if not np.isclose(cfg.obs_max_silence_s, obs_max_silence_s).all():
        raise ValueError(
            f"obs_max_silence_s must be multiple of obs_time_step_s, got {cfg.obs_max_silence_s} and {cfg.obs_time_step_s}"
        )

    obs = AGENT1(
        time_step=cfg.obs_time_step_s,
        max_silence_s=obs_max_silence_s,
        zero_init=False,
        hsize=128,
        nlayers=1,
    ).to(device)

    critic = CRITIC01(obs, hsize=128, nlayers=1, use_machine_id=False).to(
        device
    )  # (256, 3)

    discriminator = discriminator.to(device)
    discriminator_orig = discriminator_orig.to(device)
    disc_league = _append_to_league([], discriminator_orig.state_dict())
    # disc_league = _append_to_league(disc_league, discriminator.state_dict())

    obs_league = _append_to_league([], obs.state_dict())

    lr = 0.005
    obs_optim = _get_optim(obs, lr=lr, lr_rnn=lr / 3)

    lr_critic = lr / 2
    critic_optim = _get_optim(critic, lr=lr_critic, lr_rnn=lr_critic / 3)

    disc_optim = _get_optim(discriminator, lr=0.001, lr_rnn=0.001)

    satlen = 50

    # Entropy scale
    entropy_scale = 0.01

    entropy_target = 1.2
    # We drive the entropy loss to 0.001 during satlen steps...

    # Padding reward scale
    padding_scale = 0.01
    padding_scale_max = 0.01
    padding_scale_step = (padding_scale_max - padding_scale) / satlen

    league_update_frac = 0.2
    active_league_idx = None
    disc_loss_thres = 3.5

    train_disc = False  # True

    disc_ema_loss = 10.0
    ema_decay = 0.96
    obs_train_frac = 0.0

    e = 0
    detach_period = cfg.h_detach_period
    with mlflow.start_run(log_system_metrics=True):
        while True:
            reward_scales = {"clf_scale": 10.0, "padding_scale": padding_scale}
            losses_metrics_d: dict[str, list[float] | float] = {
                "loss": [],
                "policy_loss": [],
                "value_loss": [],
                "disc_loss": [],
                "avg_return": [],
                "entropy": [],
                "entropy_loss": [],
                "mean_padding_frac": [],
                "mean_trace_len": [],
                "sel vs. cond std ratio": [],
                "train_disc": [],
            }
            losses_metrics_d.update(
                {"mean_reward_" + k.replace("_scale", ""): [] for k in reward_scales}
            )

            with torch.inference_mode():
                disc_league, active_league_idx, active_disc_league, weights = (
                    get_active_league(
                        active_league_idx=active_league_idx,
                        league=disc_league,
                        ds=ds_valid,
                        obs=obs,
                        critic=critic,
                        obs_feats=ds_train.feature_trs,
                        disc=discriminator,
                        disc_feats=disc_feats,
                        reward_scales=reward_scales,
                        device=device,
                        league_size=cfg.league_size,
                        league_update_frac=league_update_frac,
                        prune=len(disc_league) > 20,
                    )
                )
            # Force the current discriminator into the league - note this is not frozen!
            disc_idx = weights.argmax().item()
            active_disc_league[disc_idx] = (
                len(disc_league),
                discriminator.state_dict(),
            )

            # Train obs:
            # ===========================================
            obs.train()
            obs.cond_beta = 0.3

            with tqdm(
                dl_train,
                desc=f"epoch {e:02d}",
                ncols=2 * TQDM_W,
            ) as pbar:
                for X, y in pbar:
                    train_obs = False
                    if np.random.rand() < obs_train_frac:
                        train_obs = True

                    X = dict_to_device(X, device)
                    y = y.to(device)

                    obs_optim.zero_grad()
                    critic_optim.zero_grad()

                    context = torch.enable_grad() if train_obs else torch.no_grad()
                    with context:
                        timing = {}
                        (
                            log_ps,
                            sel_probs,
                            values,
                            league_rewards,
                            entropies,
                            _,
                            _,
                            Xobs,
                            fd,
                        ) = rollout(
                            obs=obs,
                            critic=critic,
                            disc=discriminator,
                            X=X,
                            y=y,
                            disc_features=disc_feats,
                            disc_league=active_disc_league,
                            detach_period=detach_period,
                            reward_scales=reward_scales,
                            timing=timing,
                        )

                    # Timing (ms)
                    for k in (
                        "rollout_total_ms",
                        "rollout_fd_ms",
                        "rollout_act_ms",
                        "rollout_send_exec_ms",
                        "rollout_disc_forward_ms",
                        "rollout_rewards_ms",
                        "rollout_critic_ms",
                    ):
                        if k in timing:
                            losses_metrics_d.setdefault(k, []).append(float(timing[k]))

                    action_seq_lens = get_action_seq_lens(fd)

                    time_mask = make_time_mask(
                        action_seq_lens, fd[Feats.TIMES].shape[1], device=device
                    )

                    # The G, and advanages (nleague, bs, T)
                    # are detached from the comput graph.
                    G, advantages = get_advantages(
                        league_rewards, values, action_seq_lens, cfg
                    )

                    # Compute losses, advantages and G ARE detached.
                    # weights.shape = (nleague, ), --> advantages.shape = (bs, T)
                    advantages = (weights[:, None, None] * advantages).sum(dim=0)

                    # _, adv_std = masked_mean_std(advantages, time_mask)

                    policy_loss_ = -(log_ps * advantages)
                    policy_loss = masked_mean(policy_loss_, time_mask)

                    # valus.shape = (nleague, bs, T), G.shape = (nleague, bs, T)
                    # -> value_loss_.shape = (bs, T)
                    value_loss_ = 0.5 * (
                        weights[:, None, None] * (values - G).pow(2)
                    ).sum(dim=0)
                    value_loss = masked_mean(value_loss_, time_mask)

                    entropy = masked_mean(entropies, time_mask)
                    entropy_loss = -entropy_scale * entropy

                    loss = policy_loss + entropy_loss

                    # Track the effect of selection vs conditional
                    sel_log_ps = torch.log(sel_probs)
                    sel_term = (advantages[..., None] * sel_log_ps).std()
                    cond_term = (
                        advantages[..., None] * (log_ps[..., None] - sel_log_ps)
                    ).std()
                    ratio = cond_term / (sel_term + 1e-8)

                    # The steps are only taken for obs_train_frac steps.
                    if train_obs:
                        # Obs step:
                        # ==========================================
                        # Gradient clipping
                        loss.backward()
                        nn.utils.clip_grad_norm_(
                            obs.parameters(),
                            cfg.grad_norm_clip,
                            error_if_nonfinite=True,
                        )

                        obs_optim.step()
                        # ==========================================

                        # Critic step:
                        # ==========================================
                        # Gradient clipping
                        value_loss.backward()
                        nn.utils.clip_grad_norm_(
                            critic.parameters(),
                            cfg.grad_norm_clip,
                            error_if_nonfinite=True,
                        )

                        critic_optim.step()
                        # ==========================================

                    # Discriminator:
                    # ==========================================
                    train_disc = disc_ema_loss > disc_loss_thres

                    disc_loss, _ = one_batch_train_disc(
                        disc=discriminator,
                        X=Xobs,
                        y=y,
                        disc_opm=disc_optim,
                        feature_trs=disc_feats,
                        train=train_disc,
                        grad_clip=3.0,
                        detach_period=10000,
                        get_accuracy=False,
                    )

                    disc_ema_loss = (
                        ema_decay * disc_ema_loss + (1 - ema_decay) * disc_loss
                    )

                    # Logging:
                    # =========================================
                    losses_metrics_d["loss"].append(loss.item())
                    losses_metrics_d["policy_loss"].append(policy_loss.item())
                    losses_metrics_d["value_loss"].append(value_loss.item())
                    losses_metrics_d["disc_loss"].append(disc_loss)
                    losses_metrics_d["entropy"].append(entropy.item())
                    losses_metrics_d["entropy_loss"].append(entropy_loss.item())
                    losses_metrics_d["sel vs. cond std ratio"].append(ratio.item())
                    losses_metrics_d["train_disc"].append(1 if train_disc else 0)
                    losses_metrics_d["avg_return"].append(
                        masked_mean_std(
                            (weights[:, None, None] * G).sum(dim=0), time_mask
                        )[0].item()
                    )

                    # (B, )
                    normal_packets = (
                        (Xobs[Feats.DIRS] != 0) & (Xobs[Feats.PADDING] == 0)
                    ).sum(dim=1)
                    # (B, )
                    padding_packets = (
                        (Xobs[Feats.DIRS] != 0) & (Xobs[Feats.PADDING] == 1)
                    ).sum(dim=1)

                    losses_metrics_d["mean_padding_frac"].append(
                        (padding_packets / normal_packets).mean().item()
                    )
                    losses_metrics_d["mean_trace_len"].append(
                        (Xobs[Feats.DIRS] != 0).sum(dim=1).float().mean().item()
                    )
                    for k, v in league_rewards.items():
                        losses_metrics_d[f"mean_reward_{k}"].append(v.mean().item())

                    nhist = 20
                    pbar.set_postfix(
                        {
                            "ret": np.mean(losses_metrics_d["avg_return"][-nhist:]),
                            "dloss": disc_ema_loss,
                            "d_tr_f": np.mean(losses_metrics_d["train_disc"][-nhist:]),
                        }
                    )

            # League handling:
            # =======================================
            # Append current discriminator to league
            disc_league = _append_to_league(disc_league, discriminator.state_dict())
            obs_league = _append_to_league(obs_league, obs.state_dict())

            # Padding and entropy scale updates:
            # =============================================
            losses_metrics_d["entropy_scale"] = entropy_scale
            if e < satlen:
                padding_scale += padding_scale_step

            losses_metrics_d["padding_scale"] = padding_scale

            el_vs_pl = np.mean(losses_metrics_d["entropy_loss"]) / np.mean(
                losses_metrics_d["policy_loss"]
            )
            losses_metrics_d["entropy_loss vs. policy_loss"] = el_vs_pl
            if abs(el_vs_pl) < 0.05:
                entropy_scale *= 1.1
            else:
                entropy_scale *= 0.9
            # =============================================

            # Disc loss thres update:
            # =============================================
            losses_metrics_d["disc_loss_thres"] = disc_loss_thres
            losses_metrics_d["obs_train_frac"] = obs_train_frac
            if (d_train_frac := np.mean(losses_metrics_d["train_disc"])) > 0.90:
                # If the disc has been trained most of the time, lower the thres.
                disc_loss_thres *= 1.01
                if d_train_frac == 1:
                    obs_train_frac *= 0.5
            elif d_train_frac == 0:
                obs_train_frac = 1.0
                disc_loss_thres *= 0.98

            # Logging:
            # =============================================
            for k, v in losses_metrics_d.items():
                if isinstance(v, list):
                    if len(v) == 0:
                        continue
                    else:
                        v = np.mean(v)
                elif isinstance(v, (int, float)):
                    pass
                else:
                    raise ValueError("Invalid value to be logged")

                k = keymap(k)
                logger.info(f"\t{k:<40} : {v:.03f}")

                mlflow.log_metric(k, v, step=e)

            if e > 0 and e % cfg.figs_period == 0:
                _plot_set(
                    cfg=cfg,
                    ds=ds_valid,
                    obs=obs,
                    critic=critic,
                    obs_features=ds_train.feature_trs,
                    disc_orig=discriminator_orig,
                    disc_trained=discriminator,
                    disc_features=disc_feats,
                    disc_league_idx=len(disc_league),
                    e=e,
                    reward_scales=reward_scales,
                    device=device,
                    ntraces=10,
                    max_len=20_000,
                )

            if e > 0 and e % 10 == 0:
                mlflow.pytorch.log_model(obs, name=f"rlobs-{e}", step=e)
                # mlflow.pytorch.log_model(discriminator, name=f"rldisc-{e}", step=e)

            key = "valid:obs_vs._disc"
            d = valid_metrics(
                disc=discriminator,
                obs=obs,
                ds_valid=ds_valid,
                n_packets=cfg.trace_len,
                device=device,
                key=key,
            )

            d["entropy_target"] = entropy_target

            for k, v in d.items():
                k = keymap(k)
                mlflow.log_metric(k, v, step=e)
                logger.info(f"\t{k:<40} : {v:.03f}")

            # =============================================

            e += 1


if __name__ == "__main__":
    main()
