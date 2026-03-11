import os

import dotenv
import hydra
import mlflow
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch import nn
from tqdm import tqdm

from experiment.obsrl.plot_utils import _plot_set
from experiment.obsrl.sim import rollout
from experiment.obsrl.utils import (
    _append_to_league,
    _get_obs_def_dl,
    _get_optim,
    _restore_obs_def_ds,
    ema_update,
    get_action_seq_lens,
    get_active_league,
    get_advantages,
    keymap,
    load_leagues_from_children,
    log_lrs,
    make_time_mask,
    masked_mean,
    train_disc_one_epoch,
)
from experiment.trace_gan.data_utils import dl_
from experiment.utils import defence_builder
from kipl_ml.data.utils import DOWNLOAD, UPLOAD, Datasets, assets
from kipl_ml.data.wf_dataset import WFDataset, dict_to_device, get_train_valid_test
from kipl_ml.logging.logger import TQDM_W, get_logger
from kipl_ml.models.trgen import AGENT1, CRITIC01
from kipl_ml.rl.enums import Actions
from kipl_ml.tools.mlflow_utils import (
    find_parent_run_id,
    get_mlflow_expr,
    list_child_runs,
    next_child_idx,
)
from kipl_ml.trace.features import Feats, FeatureTrs, get_feature_tr

logger = get_logger(__name__)
dotenv.load_dotenv()

WORKING_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_DIR_PATH = os.path.join(WORKING_DIR, "config")

N_SPLITS = 5
TEST_XV = 0
TARGET = assets.PAGE_LABEL
DATASET = Datasets.BIGENOUGH


def train_obs_one_epoch(
    dl_train: torch.utils.data.DataLoader,
    obs: AGENT1,
    critic: CRITIC01 | None,
    discriminator: nn.Module,
    disc_feats: FeatureTrs,
    active_disc_league: list[nn.Module.state_dict],
    weights: torch.Tensor,
    reward_scales: dict[str, float],
    obs_optim: torch.optim.Optimizer,
    critic_optim: torch.optim.Optimizer | None,
    device: torch.device,
    e: int,
    cfg: DictConfig,
) -> dict[str, float]:
    ema_sel_entropy = None
    ema_cond_entropy = None
    ema_ret = None
    ema_decay = cfg.ema_decay

    losses_metrics_d: dict[str, list[float] | float] = {
        "loss": [],
        "policy_loss": [],
        "value_loss": [],
        "disc_loss": [],
        "avg_return": [],
        "sel_entropy": [],
        "cond_entropy": [],
        "entropy": [],
        "entropy_loss": [],
        "mean_padding_frac": [],
        "mean_padding_frac_up": [],
        "mean_padding_frac_down": [],
        "sel vs. cond std ratio": [],
        "train_disc": [],
        "grad_norm": [],
    }
    losses_metrics_d.update(
        {"mean_reward_" + k.replace("_scale", ""): [] for k in reward_scales}
    )

    with tqdm(
        dl_train,
        desc=f"Train obs {e:02d}",
        ncols=2 * TQDM_W,
    ) as pbar:
        for X, y in pbar:
            obs.eval()
            discriminator.eval()
            obs.train()

            if critic is not None:
                critic.train()
                if critic_optim is None:
                    raise ValueError("Critic given w.o. optimizer")
                critic_optim.zero_grad()

            X = dict_to_device(X, device)
            y = y.to(device)

            obs_optim.zero_grad()

            (
                log_ps,
                sel_probs,
                values,
                league_rewards,
                entropies,
                _,
                actions,
                X_obs,
                fd,
            ) = rollout(
                obs=obs,
                critic=critic,
                disc=discriminator,
                X=X,
                y=y,
                disc_features=disc_feats,
                disc_league=active_disc_league,
                detach_period=cfg.obs.detach_period,
                critic_detach_period=cfg.obs.critic_detach_period,
                reward_scales=reward_scales,
            )

            action_seq_lens = get_action_seq_lens(fd)

            time_mask = make_time_mask(
                action_seq_lens, fd[Feats.TIMES].shape[1], device=device
            )

            # The G, and advanages (nleague, bs, T)
            # are detached from the comput graph.
            G, advantages = get_advantages(league_rewards, values, action_seq_lens, cfg)

            # Compute losses, advantages and G ARE detached.
            # weights.shape = (nleague, ), --> advantages.shape = (bs, T)
            advantages = (weights[:, None, None] * advantages).sum(dim=0)

            policy_loss_ = -(log_ps * advantages)
            policy_loss = masked_mean(policy_loss_, time_mask, per_trace=True).mean()

            # valus.shape = (bs, T), G.shape = (nleague, bs, T)
            # -> value_loss_.shape = (bs, T)
            value_loss_ = 0.5 * (
                weights[:, None, None] * (values[None, ...] - G).pow(2)
            ).sum(dim=0)
            value_loss = masked_mean(value_loss_, time_mask, per_trace=True).mean()

            selection_entropy = masked_mean(
                entropies["selection_entropy"], time_mask, per_trace=True
            ).mean()
            conditional_entropy = masked_mean(
                entropies["conditional_entropy"], time_mask, per_trace=True
            ).mean()
            entropy_loss = float(cfg.obs.enable_entropy_loss) * (
                -cfg.obs.selection_entropy_scale * selection_entropy
                - cfg.obs.conditional_entropy_scale * conditional_entropy
            )

            if critic is not None:
                loss = policy_loss + entropy_loss

                # Critic step:
                # ==========================================
                value_loss.backward()
                nn.utils.clip_grad_norm_(
                    critic.parameters(),
                    cfg.grad_norm_clip,
                    error_if_nonfinite=True,
                )
                critic_optim.step()
                # ==========================================
            else:
                loss = policy_loss + value_loss + entropy_loss

            # Obs step:
            # ==========================================
            loss.backward()
            norm_ = nn.utils.clip_grad_norm_(
                obs.parameters(),
                cfg.grad_norm_clip,
                error_if_nonfinite=True,
            )
            obs_optim.step()
            # ==========================================

            losses_metrics_d["grad_norm"].append(norm_.item() > cfg.grad_norm_clip)

            ema_sel_entropy = ema_update(
                ema_sel_entropy, selection_entropy.item(), ema_decay
            )

            ema_cond_entropy = ema_update(
                ema_cond_entropy, conditional_entropy.item(), ema_decay
            )

            # scale_update_ = np.clip(
            #    (
            #        (sel_entropy_target - ema_sel_entropy)
            #        / sel_entropy_target
            #    )
            #    ** 3,
            #    -0.5,
            #    1.0,
            # )
            # selection_entropy_scale = np.clip(
            #    selection_entropy_scale * (1 + scale_update_), 1e-7, 1e-1
            # )

            # Track the effect of selection vs conditional
            sel_idx = actions[Actions.SELECTOR].to(torch.long)
            sel_p = (
                sel_probs.gather(-1, sel_idx.unsqueeze(-1)).squeeze(-1).clamp(min=1e-12)
            )
            sel_log_p = torch.log(sel_p)
            sel_term = (advantages * sel_log_p).std()
            cond_term = (advantages * (log_ps - sel_log_p)).std()
            ratio = cond_term / (sel_term + 1e-8)

            cur_ret = (
                masked_mean(
                    (weights[:, None, None] * G).sum(dim=0),
                    time_mask,
                    per_trace=True,
                )
                .mean()
                .item()
            )
            ema_ret = ema_update(ema_ret, cur_ret, ema_decay)

            # Logging:
            # =========================================
            losses_metrics_d["loss"].append(loss.item())
            losses_metrics_d["policy_loss"].append(policy_loss.item())
            losses_metrics_d["value_loss"].append(value_loss.item())
            losses_metrics_d["sel_entropy"].append(selection_entropy.item())
            losses_metrics_d["cond_entropy"].append(conditional_entropy.item())

            losses_metrics_d["entropy"].append(
                selection_entropy.item() + conditional_entropy.item()
            )
            losses_metrics_d["entropy_loss"].append(entropy_loss.item())
            losses_metrics_d["sel vs. cond std ratio"].append(ratio.item())
            losses_metrics_d["avg_return"].append(cur_ret)
            for k, v in league_rewards.items():
                losses_metrics_d[f"mean_reward_{k}"].append(
                    masked_mean(
                        (weights[:, None, None] * v).sum(dim=0),
                        time_mask,
                        per_trace=True,
                    )
                    .mean()
                    .item()
                )

            # (B, )
            normal_packets = (
                (X_obs[Feats.DIRS] != 0) & (X_obs[Feats.PADDING] == 0)
            ).sum(dim=1)
            # (B, )
            padding_packets_up = (
                (X_obs[Feats.DIRS] == UPLOAD) & (X_obs[Feats.PADDING] == 1)
            ).sum(dim=1)
            padding_packets_down = (
                (X_obs[Feats.DIRS] == DOWNLOAD) & (X_obs[Feats.PADDING] == 1)
            ).sum(dim=1)
            padding_packets = padding_packets_down + padding_packets_up

            losses_metrics_d["mean_padding_frac"].append(
                (padding_packets / normal_packets).mean().item()
            )
            losses_metrics_d["mean_padding_frac_up"].append(
                (padding_packets_up / normal_packets).mean().item()
            )
            losses_metrics_d["mean_padding_frac_down"].append(
                (padding_packets_down / normal_packets).mean().item()
            )

            postfix = {
                "pfu": np.mean(losses_metrics_d["mean_padding_frac_up"]),
                "pfd": np.mean(losses_metrics_d["mean_padding_frac_down"]),
                "ret": ema_ret,
                "Hs": ema_sel_entropy,
                "Hc": ema_cond_entropy,
            }

            pbar.set_postfix({k: f"{v:.03f}" for k, v in postfix.items()})

    # Logging:
    # =============================================
    mean_d = {}
    for k, v in losses_metrics_d.items():
        if isinstance(v, list):
            if len(v) == 0:
                continue
            v = np.mean(v)
        elif isinstance(v, (int, float)):
            pass
        else:
            raise ValueError("Invalid value to be logged")

        mean_d[k] = v

    return mean_d


def train_disc_on_league(
    discriminator: nn.Module,
    ds_train: WFDataset,
    disc_feats: FeatureTrs,
    obs: AGENT1,
    disc_league: list[tuple[int, torch.nn.Module.state_dict]],
    obs_league: list[tuple[int, dict]],
    device: torch.device,
    e: int,
    cfg: DictConfig,
) -> list[tuple[int, torch.nn.Module.state_dict]]:
    orig_feat_trs = ds_train.feature_trs
    dl_train_ = _get_obs_def_dl(
        disc=discriminator,
        disc_feats=disc_feats,
        obs=obs,
        ds=ds_train,
        n_packets=cfg.trace.len,
        bs=cfg.batch_size,
        obs_league=[d for _, d in obs_league[-cfg.league.size :]],
    )

    disc_optim = _get_optim(discriminator, lr=cfg.disc.lr)

    disc_lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=disc_optim, factor=0.8, patience=7
    )

    ed = 0
    while True:
        loss = train_disc_one_epoch(
            clf=discriminator,
            dl_train=dl_train_,
            optimG=disc_optim,
            device=device,
            grad_clip=cfg.grad_norm_clip,
            detach_period=cfg.disc.detach_period,
            epoch=ed,
        )

        if disc_lr_scheduler is not None:
            disc_lr_scheduler.step(loss)
            logger.info("Disc lrs:")
            log_lrs(disc_lr_scheduler)

        if loss < cfg.disc.loss_thres_roll:
            disc_league = _append_to_league(disc_league, discriminator.state_dict())
            logger.info(
                f"Disc train termination {e:02d}; cur loss {loss:.4f} "
                + f"< {cfg.disc.loss_thres_roll:.4f}"
            )
            break

        ed += 1
    _restore_obs_def_ds(ds_train, orig_feat_trs, obs, device)

    return disc_league


def train_obs_on_league(
    obs_league: list[tuple[int, dict]],
    obs: AGENT1,
    critic: CRITIC01 | None,
    ds_train: WFDataset,
    discriminator: nn.Module,
    disc_feats: FeatureTrs,
    active_disc_league: list[nn.Module.state_dict],
    weights: torch.Tensor,
    reward_scales: dict[str, float],
    device: torch.DeviceObjType,
    e: int,
    cfg: DictConfig,
) -> tuple[list[tuple[int, nn.Module.state_dict]], AGENT1, CRITIC01 | None]:
    reward_scales_ = reward_scales.copy()
    if not cfg.obs.reuse_obs_and_critic:
        obs, critic = get_agent_and_critic(cfg)

    obs.to(device)

    lr = cfg.obs.lr
    obs_optim = _get_optim(obs, lr=lr, lr_rnn=lr * cfg.obs.rnn_lr_reduction)
    obs_lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=obs_optim,
        factor=cfg.obs.lr_scheduler_factor,
        patience=cfg.obs.lr_scheduler_patience,
    )

    critic_optim = None
    critic_lr_scheduler = None
    if critic is not None:
        critic.to(device)
        lr_critic = cfg.obs.critic_lr
        critic_optim = _get_optim(
            critic, lr=lr_critic, lr_rnn=lr_critic * cfg.obs.rnn_lr_reduction
        )
        critic_lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=critic_optim,
            factor=cfg.obs.lr_scheduler_factor,
            patience=cfg.obs.lr_scheduler_patience,
        )

    ret_thres = cfg.obs.ret_thres_push

    eo = 1
    while True:
        metrics_d = train_obs_one_epoch(
            dl_train=dl_(ds_train, bs=cfg.batch_size, collate_fn=None, shuffle=True),
            obs=obs,
            critic=critic,
            discriminator=discriminator,
            disc_feats=disc_feats,
            active_disc_league=active_disc_league,
            weights=weights,
            reward_scales=reward_scales_,
            obs_optim=obs_optim,
            critic_optim=critic_optim,
            device=device,
            e=eo,
            cfg=cfg,
        )
        logger.info("Current ret: %.02f", metrics_d["avg_return"])

        # Log per-epoch metrics within this push run.
        for k, v in metrics_d.items():
            mlflow.log_metric(keymap(k), float(v), step=eo)
        mlflow.log_metric(
            "padding_scale", float(reward_scales_["padding_scale"]), step=eo
        )

        if obs_lr_scheduler is not None:
            obs_lr_scheduler.step(-metrics_d["avg_return"])
            logger.info("Obs lrs:")
            log_lrs(obs_lr_scheduler)

        if critic_lr_scheduler is not None and critic_optim is not None:
            critic_lr_scheduler.step(metrics_d["value_loss"])
            logger.info("Critic lrs:")
            log_lrs(critic_lr_scheduler)

        if (ret := metrics_d["avg_return"]) > ret_thres:
            logger.info(
                f"Achieved return {ret:.03f} > {ret_thres:.03f}, "
                + "stopping obs training!"
            )

            obs_league = _append_to_league(obs_league, obs.state_dict())
            logger.info(f"Obs league len: {len(obs_league)}")

            for k, v in metrics_d.items():
                k = keymap(k)
                logger.info(f"\t{k:<40} : {v:.03f}")
            break

        if eo % cfg.obs.rewards_rescale_epochs == 0:
            reward_scales_["padding_scale"] *= cfg.obs.padding_scale_reduction
            logger.info(f"At obs epoch {eo} re-scaled rewards -> ")
            for k, v in reward_scales_.items():
                logger.info(f"\t{k:>30} : {v:.04f}")

        if eo > cfg.obs.max_epochs_per_push:
            obs, critic = get_agent_and_critic(cfg)
            obs.to(device)
            if critic is not None:
                critic.to(device)
            logger.warning("No obs found; resetting...")
            return obs_league, obs, critic

        eo += 1
    return obs_league, obs, critic


def get_agent_and_critic(cfg: DictConfig) -> tuple[AGENT1, CRITIC01 | None]:
    eps = cfg.obs.prob_eps
    f_ = 0.5
    obs = AGENT1(
        time_step=cfg.obs.time_step_s,
        max_silence_s=cfg.obs.max_silence_s,
        hsize=cfg.obs.hsize,
        nlayers=cfg.obs.nhidden,
        prob_eps={
            Actions.SELECTOR: eps,
            Actions.SEND_COUNT_UP: f_ * eps,
            Actions.SEND_TIME_UP: f_ * eps,
            Actions.SEND_COUNT_DOWN: f_ * eps,
            Actions.SEND_TIME_DOWN: f_ * eps,
        },
        prefer_wait_bias=6.0 if cfg.obs.init_for_wait else 0.0,
        send_mode=cfg.obs.send_mode,
        enable_delay=getattr(cfg.obs, "enable_delay", False),
        train_env={"trim_beginning": cfg.trace.trim_beginning},
    )

    critic = None
    if cfg.obs.separate_critic:
        critic = CRITIC01(obs, hsize=256, nlayers=3)

    return obs, critic


@hydra.main(config_path=CONFIG_DIR_PATH, config_name="sisyphus", version_base=None)
def main(cfg: DictConfig):
    experiment_name = cfg.experiment_name
    experiment_id = get_mlflow_expr(experiment_name=experiment_name)
    mlflow.set_experiment(experiment_id=experiment_id)

    if not (parent_run_name := cfg.mlflow.parent_run_name):
        raise ValueError("cfg.mlflow.parent_run_name must be set")

    if (parent_run_id := find_parent_run_id(experiment_id, parent_run_name)) is None:
        logger.info("Creating parent MLflow run: %s", parent_run_name)
        with mlflow.start_run(run_name=parent_run_name):
            mlflow.set_tag("project", "obsrl.sisyphus")
            parent_run_id = mlflow.active_run().info.run_id

    child_runs = list_child_runs(experiment_id, parent_run_id)
    child_idx = next_child_idx(child_runs)
    child_run_name = f"{child_idx:03d}"
    logger.info("Parent run: %s (%s)", parent_run_name, parent_run_id)
    logger.info("Child run: %s", child_run_name)

    model_id = cfg.disc.no_defense_disc_id
    discriminator_orig = mlflow.pytorch.load_model(
        mlflow.get_logged_model(model_id).model_uri, map_location="cpu"
    )

    time_clamp = (0.0, cfg.trace.dur_max, True) if cfg.trace.dur_max is not None else None
    if discriminator_orig.feat_mode == "tam":
        feature_names = discriminator_orig.features
        tam_d = discriminator_orig.tam_dict
        tam_d["max_load_time_s"] = cfg.disc.tam_max_load_time_s

        tam_ww = tam_d["window_width_s"]
        if abs(tam_ww - float(cfg.obs.time_step_s)) > 1e-6:
            raise ValueError(
                "TAM window_width_s must match obs.time_step_s for TAM reward mapping. "
                + f"Got tam_ww={tam_ww:.6f}s and obs.time_step_s={float(cfg.obs.time_step_s):.6f}s."
            )
        npackets = None

        disc_trs = [
            get_feature_tr(fn, npackets, time_clamp, tam_kwargs=tam_d)
            for fn in feature_names
        ]
        disc_trs += [
            get_feature_tr(f, npackets, time_clamp, tam_kwargs=tam_d)
            for f in (Feats.TAM_DOWN_PAD, Feats.TAM_UP_PAD)
        ]
        # Expose integer TAM bins for reward mapping (disc ignores extra keys).
        disc_trs.append(
            get_feature_tr(Feats.TAM_BINS, npackets, time_clamp, tam_kwargs=tam_d)
        )

        disc_feats = FeatureTrs(feature_trs=disc_trs, n_packets=None)
    elif discriminator_orig.feat_mode == "dir":
        feature_names = discriminator_orig.features
        npackets = cfg.trace.len

        # Discriminator features, w.o. limit on n_packets
        disc_feats = FeatureTrs(feature_names=feature_names, n_packets=None)
    else:
        raise ValueError(
            f"Unknown feat_mode {discriminator_orig.feat_mode} in discriminator_orig!"
        )

    logging.info("Discriminator features_trs:")
    disc_feats.report()

    # This one already somewhat trained for obsfuscation.
    discriminator = mlflow.pytorch.load_model(
        mlflow.get_logged_model(model_id).model_uri, map_location="cpu"
    )
    discriminator.predict_ks = cfg.disc.predict_ks

    ds_train, ds_valid, _ = get_train_valid_test(
        dataset=DATASET,
        label=assets.PAGE_LABEL,
        n_splits=N_SPLITS,
        test_xv=TEST_XV,
        random_state=42,
        feature_trs=FeatureTrs(
            feature_names=[Feats.DIRS, Feats.TIMES],
            n_packets=cfg.trace.len,
            time_clamp=time_clamp,
        ),
        defence_aug_valid=0,
        n_min_packets=cfg.min_packets_in_trace,
        exclude_time_to_packets_n=cfg.trace.len,
        exclude_time_to_packets_s=cfg.exclude_longer_than_s,
        trim_raw=cfg.trace.trim_beginning,
        **defence_builder.get_defence(cfg),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Defense initialization:
    # ============================================

    obs_seed, critic = get_agent_and_critic(cfg)
    # ============================================

    # Load leagues + current models from existing children.
    (
        obs_league,
        disc_league,
        obs,
        discriminator,
        latest_critic_state,
    ) = load_leagues_from_children(
        experiment_id=experiment_id,
        parent_run_id=parent_run_id,
        discriminator_orig=discriminator_orig,
        discriminator_seed=discriminator,
        obs_seed=obs_seed,
        cfg=cfg,
    )

    if (critic is not None) and obs_league:
        if latest_critic_state is None:
            raise ValueError(
                "Critic league given but no critic state found in children!"
            )
        critic.load_state_dict(latest_critic_state)

    discriminator = discriminator.to(device)
    discriminator_orig = discriminator_orig.to(device)
    obs = obs.to(device)
    if critic is not None:
        critic = critic.to(device)

    if {
        discriminator.trim_beginning,
        discriminator_orig.trim_beginning,
        obs.train_env.get("trim_beginning", None),
    } != {cfg.trace.trim_beginning}:
        raise ValueError("Discrim trained on different trimming...")

    # If no prior children and init_for_wait is False, seed the disc league with
    # an additional (trainable) discriminator checkpoint.
    if len(disc_league) == 0:
        raise ValueError(
            "No discriminator in league; expected at least the original one!"
        )
    if len(obs_league) == 0:
        obs_league = _append_to_league([], obs.state_dict())

    active_league_idx = None

    reward_scales = {
        "clf_scale": 0.1,
        "d_clf_scale": cfg.rewards.d_clf,
        "padding_scale": cfg.rewards.padding_scale,
        "delay_scale": cfg.rewards.delay_scale,
    }

    # One process invocation == one child run (one push).
    with mlflow.start_run(run_id=parent_run_id):
        with mlflow.start_run(
            nested=True,
            run_name=child_run_name,
            log_system_metrics=True,
        ):
            mlflow.set_tag("project", "obsrl.sisyphus")
            mlflow.set_tag("sisyphus.parent_run_name", parent_run_name)
            mlflow.set_tag("sisyphus.child_idx", child_idx)
            mlflow.set_tag("sisyphus.child_run_name", child_run_name)

            d = OmegaConf.to_container(cfg, resolve=True)
            mlflow.log_params(d)

            e = child_idx
            logger.info(f"Push {e:03d} - Training discriminator on obs league...")
            disc_league = train_disc_on_league(
                discriminator=discriminator,
                ds_train=ds_train,
                disc_feats=disc_feats,
                obs=obs,
                disc_league=disc_league,
                obs_league=obs_league,
                device=device,
                e=e,
                cfg=cfg,
            )

            logger.info(f"Push {e:03d} - Getting active league and weights...")
            with torch.no_grad():
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
                        league_size=cfg.league.size,
                        league_random_frac=cfg.league.random_frac,
                        prune=len(disc_league) > cfg.league.size * 4,
                        enforce_orig=cfg.league.enforce_orig,
                        score_type=cfg.league.score_type,
                        n_packets=cfg.trace.len,
                    )
                )
            logger.info("\tRe-scaling league weights to uniform...")
            weights = torch.ones_like(weights) / len(weights)

            logger.info(f"Push {e:03d} - Training obs on disc league...")
            obs_league, obs, critic = train_obs_on_league(
                obs_league=obs_league,
                obs=obs,
                critic=critic,
                ds_train=ds_train,
                discriminator=discriminator,
                disc_feats=disc_feats,
                active_disc_league=active_disc_league,
                weights=weights,
                reward_scales=reward_scales,
                device=device,
                e=e,
                cfg=cfg,
            )

            _plot_set(
                cfg=cfg,
                ds=ds_valid,
                obs=obs,
                critic=critic,
                obs_features=ds_train.feature_trs,
                disc_orig=discriminator_orig,
                disc_trained=discriminator,
                disc_features=disc_feats,
                active_disc_league=active_disc_league,
                weights=weights,
                e=e,
                reward_scales=reward_scales,
                device=device,
                ntraces=20,
            )

            # Persist the current push endpoints for reconstruction on next invocation.
            mlflow.pytorch.log_model(obs, name=f"obs-{child_run_name}", step=child_idx)
            mlflow.pytorch.log_model(
                discriminator, name=f"disc-{child_run_name}", step=child_idx
            )
            if critic is not None:
                mlflow.pytorch.log_model(
                    critic, name=f"critic-{child_run_name}", step=child_idx
                )


if __name__ == "__main__":
    main()
