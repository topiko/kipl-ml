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
from experiment.obsrl.utils import train_one_epoch
from experiment.trace_gan.data_utils import dl_
from experiment.utils import defence_builder
from kipl_ml.data.utils import Datasets, assets
from kipl_ml.data.wf_dataset import WFDataset, dict_to_device, get_train_valid_test
from kipl_ml.defences.base import NoDefence
from kipl_ml.defences.nndefs import RNNDef
from kipl_ml.logging.logger import TQDM_W, get_logger
from kipl_ml.metrics.clf_metrics import Accuracy
from kipl_ml.model_eval.evaluate import evaluate_model
from kipl_ml.models.trgen import AGENT1
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


def _plot_set(
    cfg: DictConfig,
    ds: WFDataset,
    obs: nn.Module,
    obs_features: FeatureTrs,
    disc_orig: nn.Module,
    disc_trained: nn.Module,
    disc_features: FeatureTrs,
    e: int,
    reward_scales: dict[str, float],
    device: torch.DeviceObjType,
    ntraces: int = 3,
    max_len: int = 10_000,
):
    rng = np.random.default_rng(seed=42)

    idxs = rng.choice(len(ds), size=ntraces, replace=False)

    for idx in idxs:
        _plot_single(
            cfg=cfg,
            ds=ds,
            obs=obs,
            obs_features=obs_features,
            disc_orig=disc_orig,
            disc_trained=disc_trained,
            disc_features=disc_features,
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
    obs_features: FeatureTrs,
    disc_orig: nn.Module,
    disc_trained: nn.Module,
    disc_features: FeatureTrs,
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
            disc_trained,
            X,
            y,
            disc_league=[disc_trained.state_dict()],
            disc_features=disc_features,
            reward_scales=reward_scales,
        )
    )

    rewards = league_rewards2rewards(league_rewards)
    G, _ = get_advantages(rewards, state_values, get_action_seq_lens(fd), cfg)

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
    plot_rewards(times, rewards, ax=ax_b)
    ax_b.set_title("Rewards, returns")

    # Plot returns
    ax_r = ax_b.twinx()
    ax_r.axes.spines["right"].set_visible(True)
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
    ax_b.set_xlabel("Time [s]")

    fig.canvas.draw()

    plt.show()
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

    values_detached = values.detach()

    if cfg.advantages.type == "mc":
        G = get_returns(rewards, seq_lens, gamma=cfg.discounting)
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

    if cfg.advantages.standardize:
        mask = make_time_mask(
            seq_lens.to(values.device), values.shape[1], device=values.device
        )
        mean, std = masked_mean_std(advantages, mask)
        advantages = (advantages - mean) / std
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


def _append_to_league(league: list[dict], state_dict: dict):
    league.append({k: v.cpu() for k, v in copy.deepcopy(state_dict).items()})


def _get_obs_def_dl(
    disc: nn.Module, obs: nn.Module, ds: WFDataset, n_packets: int, bs: int = 64
) -> WFDataset:
    # Set features the fetures:
    ds.feature_trs = FeatureTrs(feature_names=disc.features, n_packets=n_packets)

    # Set the defense:
    ds.defence = RNNDef((0, 0), (40_000, 40_000), obs.to("cpu"), n_packets=n_packets)

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
) -> dict[str, float]:
    orig_features_trs = ds_valid.feature_trs
    dl_valid = _get_obs_def_dl(
        disc=disc, obs=obs, ds=ds_valid, n_packets=n_packets, bs=32
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
    league: list[nn.Module],
    ds: WFDataset,
    obs: nn.Module,
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
    dl = dl_(ds, bs=128, collate_fn=None, shuffle=False, nworkers=None, sampler=sampler)

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
                league_rewards = rollout(
                    obs=obs,
                    disc=disc,
                    X=X,
                    y=y,
                    disc_features=disc_features,
                    disc_league=league,
                    detach_period=500,
                    reward_scales=reward_scales,
                )[3]
                # (nleague, nbatch, ntimesteps) -> (nleague, nbatch) -> (nleague, 1)
                rewards = {
                    k: v.mean(dim=1).mean(dim=1).unsqueeze(1)
                    for k, v in league_rewards.items()
                }
                rewards_ = sum(rewards.values())
                rewards_l.append(rewards_)

        league_scores = -torch.cat(rewards_l, dim=1).mean(dim=1)

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


@hydra.main(config_path=CONFIG_DIR_PATH, config_name="config", version_base=None)
def main(cfg: DictConfig):
    experiment_name = "obsrl"
    experiment_id = get_mlflow_expr(experiment_name=experiment_name)
    mlflow.set_experiment(experiment_id=experiment_id)
    discriminator_model_id = "m-a1bec780ec314b95b4f0caac4dec5f46"

    model_uri = mlflow.get_logged_model(discriminator_model_id).model_uri

    discriminator_orig = mlflow.pytorch.load_model(model_uri, map_location="cpu")
    discriminator = mlflow.pytorch.load_model(model_uri, map_location="cpu")
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
        **defence_builder.get_defence(cfg),
    )

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
    ).to(device)

    discriminator = discriminator.to(device)
    discriminator_orig = discriminator_orig.to(device)
    league: list[dict] = []
    _append_to_league(league, discriminator_orig.state_dict())
    active_league = None

    rnn_params = list(obs.rnn.parameters())
    other_params = [p for n, p in obs.named_parameters() if not n.startswith("rnn.")]

    lr = 0.001
    optim = torch.optim.AdamW(
        [
            {"params": rnn_params, "lr": lr * 0.1},
            {"params": other_params, "lr": lr},
        ]
    )

    disc_optim = torch.optim.Adam(discriminator.parameters(), lr=0.001)

    satlen = 50

    # Entropy scale
    entropy_scale = 0.01
    # We drive the entropy loss to 0 during satlen steps...
    entropy_scale_factor = 0.01 ** (1 / satlen)

    # Padding reward scale
    padding_scale = 0.001
    padding_scale_max = 0.01
    padding_scale_step = (padding_scale_max - padding_scale) / satlen

    # Disct training:
    disc_train_count = 4

    e = 0
    detach_period = cfg.h_detach_period
    with mlflow.start_run(log_system_metrics=True):
        train_disc = True
        while True:
            reward_scales = {"clf_scale": 10.0, "padding_scale": padding_scale}
            league_scores = []
            losses_metrics_d: dict[str, list[float] | float] = {
                "loss": [],
                "policy_loss": [],
                "value_loss": [],
                "avg_return": [],
                "entropy": [],
                "mean_padding_frac": [],
                "mean_trace_len": [],
                "sel vs. cond std ratio": [],
            }
            losses_metrics_d.update(
                {"mean_reward_" + k.replace("_scale", ""): [] for k in reward_scales}
            )

            obs.train()
            obs.cond_beta = 0.3
            if train_disc:
                discriminator.train()

            with tqdm(
                dl_train,
                desc=f"epoch {e:02d}",
                ncols=2 * TQDM_W,
            ) as pbar:
                for X, y in pbar:
                    X = dict_to_device(X, device)
                    y = y.to(device)

                    optim.zero_grad()

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
                        disc=discriminator,
                        X=X,
                        y=y,
                        disc_features=disc_feats,
                        disc_league=active_league or league,
                        detach_period=detach_period,
                        reward_scales=reward_scales,
                    )

                    action_seq_lens = get_action_seq_lens(fd)

                    time_mask = make_time_mask(
                        action_seq_lens, fd[Feats.TIMES].shape[1], device=device
                    )

                    rewards = league_rewards2rewards(league_rewards)

                    # The G, and advanages are detached from the comput graph.
                    G, advantages = get_advantages(
                        rewards, values, action_seq_lens, cfg
                    )

                    # Compute losses, advantages and G ARE detached.
                    policy_loss_ = -(log_ps * advantages)
                    policy_loss = masked_mean(policy_loss_, time_mask)

                    value_loss_ = 0.5 * (values - G).pow(2)
                    value_loss = masked_mean(value_loss_, time_mask)

                    entropy = masked_mean(entropies, time_mask)

                    loss = policy_loss + value_loss - entropy_scale * entropy

                    loss.backward()

                    # Track the effect of selection vs conditional
                    sel_log_ps = torch.log(sel_probs)
                    sel_term = (advantages[..., None] * sel_log_ps).std()
                    cond_term = (
                        advantages[..., None] * (log_ps[..., None] - sel_log_ps)
                    ).std()
                    ratio = cond_term / (sel_term + 1e-8)

                    # Sanity checks:
                    # ==========================================
                    if cfg.debug:
                        if (ratio > 10.0) or (ratio < 0.1):
                            logger.warning(
                                f"High/low cond/sel std ratio: {ratio:.2f}, sel_term: {sel_term:.6f}, cond_term: {cond_term:.6f}"
                            )

                        for k, v in rewards.items():
                            try:
                                assert_finite(f"rewards-{k}", v)
                            except ValueError:
                                breakpoint()
                        assert_finite("values", values)
                        assert_finite("log_ps", log_ps)

                        for k, v in losses_metrics_d.items():
                            if len(v) == 0:
                                continue

                            try:
                                assert_finite(k, v[-1])
                            except ValueError as er:
                                print(er)
                                breakpoint()

                        for name, p in obs.named_parameters():
                            if p.grad is not None:
                                try:
                                    assert_finite(f"{name}: grad", p.grad)
                                except ValueError:
                                    print(value_loss, policy_loss, entropy)
                                    breakpoint()

                        if not torch.isclose(
                            rewards["padding"].sum(dim=1),
                            -Xobs[Feats.PADDING].sum(dim=1)
                            * reward_scales["padding_scale"],
                        ).all():
                            breakpoint()
                            logger.warning("padding rewards issues")
                    # ==========================================

                    # Gradient clipping
                    nn.utils.clip_grad_norm_(
                        obs.parameters(), cfg.grad_norm_clip, error_if_nonfinite=True
                    )

                    optim.step()

                    losses_metrics_d["loss"].append(loss.item())
                    losses_metrics_d["policy_loss"].append(policy_loss.item())
                    losses_metrics_d["value_loss"].append(value_loss.item())
                    losses_metrics_d["avg_return"].append(G.mean().item())
                    losses_metrics_d["entropy"].append(entropy.item())
                    losses_metrics_d["sel vs. cond std ratio"].append(ratio.item())

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
                    if rewards is not None:
                        for k, v in rewards.items():
                            losses_metrics_d[f"mean_reward_{k}"].append(v.mean().item())

                    pbar.set_postfix(
                        {"avg_return": np.mean(losses_metrics_d["avg_return"][-30:])}
                    )

            # Train disc:
            # ============================================
            for _ in range(disc_train_count):
                dl_valid_ = _get_obs_def_dl(
                    disc=discriminator,
                    obs=obs,
                    ds=ds_train,
                    n_packets=cfg.trace_len,
                    bs=64,
                )

                loss = train_one_epoch(
                    clf=discriminator,
                    dl_train=dl_valid_,
                    optimG=disc_optim,
                    device=device,
                    grad_clip=cfg.grad_norm_clip,
                )
                _restore_obs_def_ds(ds_train, obs_features, obs, device)

            losses_metrics_d["disc_train_loss"] = loss

            # Padding and entropy scale updates:
            # =============================================
            if e < satlen:
                entropy_scale *= entropy_scale_factor
                padding_scale += padding_scale_step

            # Logging:
            # =============================================
            losses_metrics_d["entropy_scale"] = entropy_scale
            losses_metrics_d["padding_scale"] = padding_scale

            mlflow.log_metrics(
                {k: np.mean(l_) for k, l_ in losses_metrics_d.items()}, step=e
            )
            losses_metrics_d = {k: [] for k in losses_metrics_d}

            d = valid_metrics(
                disc=discriminator,
                obs=obs,
                ds_valid=ds_valid,
                n_packets=cfg.trace_len,
                device=device,
            )
            mlflow.log_metrics(d, step=e)

            if e % 5 == 0:
                _plot_set(
                    cfg=cfg,
                    ds=ds_valid,
                    obs=obs,
                    obs_features=ds_train.feature_trs,
                    disc_orig=discriminator_orig,
                    disc_trained=discriminator,
                    disc_features=disc_feats,
                    e=e,
                    reward_scales=reward_scales,
                    device=device,
                    ntraces=20,
                    max_len=20_000,
                )

            if e % 5 == 0:
                mlflow.pytorch.log_model(obs, name=f"rlobs-{e}", step=e)
                mlflow.pytorch.log_model(discriminator, name=f"rldisc-{e}", step=e)

            # =============================================

            # League handling:
            # =======================================

            # Append current discriminator to league
            _append_to_league(league, discriminator.state_dict())

            league_scores = get_league_scores(
                league=league,
                ds=ds_valid,
                obs=obs,
                obs_features=ds_train.feature_trs,
                disc=discriminator,
                disc_features=disc_feats,
                reward_scales=reward_scales,
                device=device,
                subset_indices=torch.randint(0, len(ds_valid), (1000,)).numpy(),
            )

            if len(league) > cfg.league_size:
                scores = np.array(league_scores.cpu().numpy())
                probs = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
                probs /= probs.sum()

                active_league_idx = np.random.choice(
                    len(league), cfg.league_size, p=probs, replace=False
                )
            else:
                active_league_idx = np.arange(len(league))

            active_league = [league[i] for i in active_league_idx]

            logger.info("League scores:")
            for i, s in enumerate(league_scores):
                str_ = ""
                if i in active_league_idx:
                    str_ = "*"

                logger.info(f"\t{i:4d}{str_} : {s:.4f}")
            # =======================================

            e += 1


if __name__ == "__main__":
    main()
