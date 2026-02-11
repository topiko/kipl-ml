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
    make_time_mask,
    masked_mean,
    train_one_epoch,
)
from experiment.trace_gan.data_utils import dl_
from experiment.utils import defence_builder
from kipl_ml.data.utils import DOWNLOAD, UPLOAD, Datasets, assets
from kipl_ml.data.wf_dataset import WFDataset, dict_to_device, get_train_valid_test
from kipl_ml.logging.logger import TQDM_W, get_logger
from kipl_ml.models.trgen import AGENT1, CRITIC01
from kipl_ml.rl.enums import Actions
from kipl_ml.tools.mlflow_utils import get_mlflow_expr
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
    ema_sel_entropy = 1.0
    ema_cond_entropy = 1.0
    enable_entropy_loss = cfg.enable_entropy_loss
    selection_entropy_scale = cfg.selection_entropy_scale
    conditional_entropy_scale = cfg.conditional_entropy_scale
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
        desc=f"epoch {e:02d}",
        ncols=2 * TQDM_W,
    ) as pbar:
        for X, y in pbar:
            obs.eval()
            discriminator.eval()
            obs.train()

            if critic is not None:
                critic.train()
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
                detach_period=cfg.h_detach_period,
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
            entropy_loss = float(enable_entropy_loss) * (
                -selection_entropy_scale * selection_entropy
                - conditional_entropy_scale * conditional_entropy
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

            losses_metrics_d["grad_norm"].append(norm_.item())

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
            sel_log_ps = torch.log(sel_probs)
            sel_term = (advantages[..., None] * sel_log_ps).std()
            cond_term = (advantages[..., None] * (log_ps[..., None] - sel_log_ps)).std()
            ratio = cond_term / (sel_term + 1e-8)

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
            losses_metrics_d["avg_return"].append(
                masked_mean(
                    (weights[:, None, None] * G).sum(dim=0),
                    time_mask,
                    per_trace=True,
                )
                .mean()
                .item()
            )
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
            normal_packets = ((Xobs[Feats.DIRS] != 0) & (Xobs[Feats.PADDING] == 0)).sum(
                dim=1
            )
            # (B, )
            padding_packets_up = (
                (Xobs[Feats.DIRS] == UPLOAD) & (Xobs[Feats.PADDING] == 1)
            ).sum(dim=1)
            padding_packets_down = (
                (Xobs[Feats.DIRS] == DOWNLOAD) & (Xobs[Feats.PADDING] == 1)
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

            nhist = 20
            postfix = {
                "pfu": np.mean(losses_metrics_d["mean_padding_frac_up"]),
                "pfd": np.mean(losses_metrics_d["mean_padding_frac_down"]),
            }
            if losses_metrics_d["avg_return"]:
                postfix["ret"] = np.mean(losses_metrics_d["avg_return"][-nhist:])
                postfix["Hs"] = ema_sel_entropy
                postfix["Hc"] = ema_cond_entropy

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
    disc_optim: torch.optim.Optimizer,
    disc_lr_csheduler: torch.optim.lr_scheduler | None,
    disc_league: list[tuple(int, torch.nn.Module.state_dict)],
    obs_league: list[tuple[int, dict]],
    device: torch.device,
    e: int,
    cfg: DictConfig,
) -> list[tuple(int, torch.nn.Module.state_dict)]:
    orig_feat_trs = ds_train.feature_trs
    dl_train_ = _get_obs_def_dl(
        disc=discriminator,
        disc_feats=disc_feats,
        obs=obs,
        ds=ds_train,
        n_packets=cfg.trace_len,
        bs=cfg.batch_size,
        obs_league=[d for _, d in obs_league[-cfg.league_size :]],
    )

    ed = 0
    while True:
        loss = train_one_epoch(
            clf=discriminator,
            dl_train=dl_train_,
            optimG=disc_optim,
            device=device,
            grad_clip=cfg.grad_norm_clip,
            detach_period=10000,
            epoch=ed,
        )

        if disc_lr_csheduler is not None:
            disc_lr_csheduler.step(loss)

        if loss < cfg.disc_loss_thres_roll:
            disc_league = _append_to_league(disc_league, discriminator.state_dict())
            break

        ed += 1
    _restore_obs_def_ds(ds_train, orig_feat_trs, obs, device)

    return disc_league


def train_obs_on_league(
    obs: nn.Module,
    obs_league: list[tuple[int, dict]],
    critic: CRITIC01 | None,
    ds_train: WFDataset,
    discriminator: nn.Module,
    disc_feats: FeatureTrs,
    active_disc_league: list[nn.Module.state_dict],
    weights: torch.Tensor,
    reward_scales: dict[str, float],
    obs_optim: torch.optim.Optimizer,
    obs_lr_scheduler: torch.optim.lr_scheduler | None,
    critic_optim: torch.optim.Optimizer | None,
    critic_lr_scheduler: torch.optim.lr_scheduler | None,
    device: torch.device,
    e: int,
    cfg: DictConfig,
) -> list[tuple[int, nn.Module.state_dict]]:
    ret_thres = cfg.ret_thres_push

    eo = 0
    while True:
        metrics_d = train_obs_one_epoch(
            dl_train=dl_(ds_train, bs=cfg.batch_size, collate_fn=None, shuffle=True),
            obs=obs,
            critic=critic,
            discriminator=discriminator,
            disc_feats=disc_feats,
            active_disc_league=active_disc_league,
            weights=weights,
            reward_scales=reward_scales,
            obs_optim=obs_optim,
            critic_optim=critic_optim,
            device=device,
            e=eo,
            cfg=cfg,
        )

        if obs_lr_scheduler is not None:
            obs_lr_scheduler.step(-metrics_d["avg_return"])

        if critic_lr_scheduler is not None and critic_optim is not None:
            critic_lr_scheduler.step(metrics_d["value_loss"])

        if (ret := metrics_d["avg_return"]) > ret_thres:
            logger.info(
                f"Achieved return {ret:.03f} > {ret_thres:.03f}, "
                + "stopping obs training!"
            )

            obs_league = _append_to_league(obs_league, obs.state_dict())
            logger.info(f"Obs league len: {len(obs_league)}")
            mlflow.pytorch.log_model(obs, name=f"rlobs-{e}", step=e)

            for k, v in metrics_d.items():
                k = keymap(k)
                logger.info(f"\t{k:<40} : {v:.03f}")
                mlflow.log_metric(k, v, step=e)
            break

        eo += 1

    return obs_league


@hydra.main(config_path=CONFIG_DIR_PATH, config_name="sisyphus", version_base=None)
def main(cfg: DictConfig):
    experiment_name = cfg.experiment_name
    experiment_id = get_mlflow_expr(experiment_name=experiment_name)
    mlflow.set_experiment(experiment_id=experiment_id)

    model_id = "m-e30a0dd900c740c8a33db9c0f45d8b37"
    discriminator_orig = mlflow.pytorch.load_model(
        mlflow.get_logged_model(model_id).model_uri, map_location="cpu"
    )

    if discriminator_orig.feat_mode == "tam":
        feature_names = discriminator_orig.features
        tam_d = discriminator_orig.tam_dict
        tam_d["max_load_time_s"] = 100
        npackets = None

        disc_feats = FeatureTrs(
            feature_trs=[
                get_feature_tr(fn, npackets, tam_kwargs=tam_d) for fn in feature_names
            ]
            + [
                get_feature_tr(f, npackets, tam_kwargs=tam_d)
                for f in (Feats.TAM_DOWN_PAD, Feats.TAM_UP_PAD)
            ],
            n_packets=None,
        )
    elif discriminator_orig.feat_mode == "dir":
        feature_names = discriminator_orig.features
        npackets = cfg.trace_len

        # Discriminator features, w.o. limit on n_packets
        disc_feats = FeatureTrs(feature_names=feature_names, n_packets=None)
    else:
        raise ValueError(
            f"Unknown feat_mode {discriminator_orig.feat_mode} in discriminator_orig!"
        )

    # This one already somewhat trained for obsfuscation.
    discriminator = mlflow.pytorch.load_model(
        mlflow.get_logged_model(model_id).model_uri, map_location="cpu"
    )
    discriminator.predict_ks = cfg.predict_ks

    ds_train, ds_valid, _ = get_train_valid_test(
        dataset=DATASET,
        label=assets.PAGE_LABEL,
        n_splits=N_SPLITS,
        test_xv=TEST_XV,
        random_state=42,
        feature_trs=FeatureTrs(
            feature_names=[Feats.DIRS, Feats.TIMES], n_packets=cfg.trace_len
        ),
        defence_aug_valid=0,
        n_min_packets=cfg.min_packets_in_trace,
        **defence_builder.get_defence(cfg),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Defense initialization:
    # ============================================
    eps = cfg.obs_prob_eps
    f_ = 0.5
    obs = AGENT1(
        time_step=cfg.obs_time_step_s,
        max_silence_s=cfg.obs_max_silence_s,
        hsize=128,
        nlayers=2,
        prob_eps={
            Actions.SELECTOR: cfg.obs_prob_eps,
            Actions.SEND_COUNT_UP: f_ * eps,
            Actions.SEND_TIME_UP: f_ * eps,
            Actions.SEND_COUNT_DOWN: f_ * eps,
            Actions.SEND_TIME_DOWN: f_ * eps,
        },
        prefer_wait_bias=6.0 if cfg.init_for_wait else 0.0,
    ).to(device)

    # Defense optimizers:
    # ============================================
    lr = 0.001
    obs_optim = _get_optim(obs, lr=lr, lr_rnn=lr * cfg.rnn_lr_reduction)
    obs_lr_scheduler = None

    critic = None
    critic_optim = None
    critic_lr_scheduler = None
    if cfg.separate_critic:
        critic = CRITIC01(obs, hsize=256, nlayers=3).to(device)  # (256, 3)

        lr_critic = lr / 2
        critic_optim = _get_optim(
            critic, lr=lr_critic, lr_rnn=lr_critic * cfg.rnn_lr_reduction
        )
    # ============================================

    # Disc optimizing:
    # ============================================
    disc_optim = _get_optim(discriminator, lr=0.001, lr_rnn=0.001)

    disc_lr_csheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=disc_optim, factor=0.8, patience=10
    )
    # ============================================

    discriminator = discriminator.to(device)
    discriminator_orig = discriminator_orig.to(device)
    disc_league = _append_to_league([], discriminator_orig.state_dict())
    if not cfg.init_for_wait:
        disc_league = _append_to_league(disc_league, discriminator.state_dict())
    obs_league = _append_to_league([], obs.state_dict())

    active_league_idx = None

    reward_scales = {
        "clf_scale": 0.1,
        "d_clf_scale": cfg.rewards.d_clf,
        "padding_scale": cfg.padding_scale,
    }
    e = 0
    with mlflow.start_run(log_system_metrics=True):
        d = OmegaConf.to_container(cfg, resolve=True)
        mlflow.log_params(d)

        while True:
            logger.info(f"Epoch {e:02d} - Training discriminator on league...")
            disc_league = train_disc_on_league(
                discriminator=discriminator,
                ds_train=ds_train,
                disc_feats=disc_feats,
                obs=obs,
                disc_optim=disc_optim,
                disc_lr_csheduler=disc_lr_csheduler,
                disc_league=disc_league,
                obs_league=obs_league,
                device=device,
                e=e,
                cfg=cfg,
            )

            logger.info(f"Epoch {e:02d} - Getting active league and weights...")
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
                        league_size=cfg.league_size,
                        league_update_frac=1.0,  # unused
                        prune=len(disc_league) > cfg.league_size * 2,
                        score_type=cfg.league_score_type,
                        n_packets=cfg.trace_len,
                    )
                )
            logger.info("\tRe-scaling league weights to uniform...")
            weights = torch.ones_like(weights) / len(weights)

            # Pushing:
            # ============================================
            logger.info(f"Epoch {e:02d} - Training obs on disc league...")
            obs_league = train_obs_on_league(
                obs=obs,
                obs_league=obs_league,
                critic=critic,
                ds_train=ds_train,
                discriminator=discriminator,
                disc_feats=disc_feats,
                active_disc_league=active_disc_league,
                weights=weights,
                reward_scales=reward_scales,
                obs_optim=obs_optim,
                obs_lr_scheduler=obs_lr_scheduler,
                critic_optim=critic_optim,
                critic_lr_scheduler=critic_lr_scheduler,
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
                ntraces=10,
                max_len=20_000,
            )

            if e > cfg.max_epochs:
                logger.info("Max epochs reached.")
                break

            e += 1


if __name__ == "__main__":
    main()
