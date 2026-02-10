import os

import dotenv
import hydra
import mlflow
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.utils.data import SubsetRandomSampler
from tqdm import tqdm

from experiment.obsrl.plot_utils import _plot_set
from experiment.obsrl.sim import rollout
from experiment.obsrl.utils import (
    _append_to_league,
    _get_optim,
    ema_update,
    get_action_seq_lens,
    get_active_league,
    get_advantages,
    keymap,
    make_time_mask,
    masked_mean,
    one_batch_train_disc,
    valid_metrics,
)
from experiment.trace_gan.data_utils import dl_
from experiment.utils import defence_builder
from kipl_ml.data.utils import DOWNLOAD, UPLOAD, Datasets, assets
from kipl_ml.data.wf_dataset import dict_to_device, get_train_valid_test
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


@hydra.main(config_path=CONFIG_DIR_PATH, config_name="config", version_base=None)
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
        tam_d["max_load_time_s"] = 1000
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

    # This one already somewhat trained for obsfuscation.
    discriminator = mlflow.pytorch.load_model(
        mlflow.get_logged_model(model_id).model_uri, map_location="cpu"
    )
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
    # obs_features = ds_train.feature_trs

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    k = round(cfg.obs_max_silence_s / cfg.obs_time_step_s)
    obs_max_silence_s = k * cfg.obs_time_step_s
    if not np.isclose(cfg.obs_max_silence_s, obs_max_silence_s).all():
        raise ValueError(
            "obs_max_silence_s must be multiple of obs_time_step_s,"
            + f" got {cfg.obs_max_silence_s} and {cfg.obs_time_step_s}"
        )

    eps = cfg.obs_prob_eps
    f_ = 0.5
    obs = AGENT1(
        time_step=cfg.obs_time_step_s,
        max_silence_s=obs_max_silence_s,
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

    lr = 0.001
    obs_optim = _get_optim(obs, lr=lr, lr_rnn=lr * cfg.rnn_lr_reduction)

    disc_optim = _get_optim(discriminator, lr=0.001, lr_rnn=0.001)

    critic = None
    if cfg.separate_critic:
        critic = CRITIC01(obs, hsize=256, nlayers=3).to(device)  # (256, 3)

        lr_critic = lr / 2
        critic_optim = _get_optim(
            critic, lr=lr_critic, lr_rnn=lr_critic * cfg.rnn_lr_reduction
        )

    discriminator = discriminator.to(device)
    discriminator_orig = discriminator_orig.to(device)
    disc_league = _append_to_league([], discriminator_orig.state_dict())
    if not cfg.init_for_wait:
        disc_league = _append_to_league(disc_league, discriminator.state_dict())
    # obs_league = _append_to_league([], obs.state_dict())

    satlen = 50

    # Entropy scale
    selection_entropy_scale = 0.005
    conditional_entropy_scale = 0.0002
    ema_sel_entropy = 1.0
    ema_cond_entropy = 1.0

    sel_entropy_target = 0.5

    # Padding reward scale
    padding_scale = cfg.padding_scale
    padding_scale_max = 10 * padding_scale
    padding_scale_step = (padding_scale_max - padding_scale) / satlen

    league_update_frac = 0.2
    active_league_idx = None
    disc_loss_thres = 2.0
    min_disc_loss_thres = 0.3
    disc_loss_step = 0.1
    disc_loss_p_buffer = 0.1
    disc_train_min_p = 0.01
    disc_train_count = 0

    disc_ema_loss = 10.0
    ema_decay = 0.95
    obs_train_frac = 0.0

    enable_entropy_loss = float(cfg.enable_entropy_loss)

    e = 0
    with mlflow.start_run(log_system_metrics=True):
        d = OmegaConf.to_container(cfg, resolve=True)
        mlflow.log_params(d)

        while True:
            reward_scales = {
                "clf_scale": 0.1,
                "d_clf_scale": cfg.rewards.d_clf,
                "padding_scale": padding_scale,
            }
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
                        league_update_frac=league_update_frac,
                        prune=len(disc_league) > cfg.league_size * 2,
                        score_type=cfg.league_score_type,
                        n_packets=cfg.trace_len,
                    )
                )

            # Train obs:
            # ===========================================
            sampler = SubsetRandomSampler(
                np.random.choice(
                    len(ds_train),
                    size=int(len(ds_train) * cfg.train_subset_frac),
                    replace=False,
                )
            )
            dl_train = dl_(
                ds_train,
                bs=cfg.batch_size,
                collate_fn=None,
                shuffle=False,
                sampler=sampler,
            )

            with tqdm(
                dl_train,
                desc=f"epoch {e:02d}",
                ncols=2 * TQDM_W,
            ) as pbar:
                for X, y in pbar:
                    train_obs = False
                    obs.eval()
                    discriminator.eval()
                    if np.random.rand() < obs_train_frac:
                        train_obs = True
                        obs.train()

                    if critic is not None:
                        critic.eval()
                        if train_obs:
                            critic.train()
                        critic_optim.zero_grad()

                    X = dict_to_device(X, device)
                    y = y.to(device)

                    obs_optim.zero_grad()

                    context = torch.enable_grad() if train_obs else torch.no_grad()
                    with context:
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
                            reward_scales=reward_scales if train_obs else None,
                        )

                    if train_obs:
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

                        policy_loss_ = -(log_ps * advantages)
                        policy_loss = masked_mean(
                            policy_loss_, time_mask, per_trace=True
                        ).mean()

                        # valus.shape = (bs, T), G.shape = (nleague, bs, T)
                        # -> value_loss_.shape = (bs, T)
                        value_loss_ = 0.5 * (
                            weights[:, None, None] * (values[None, ...] - G).pow(2)
                        ).sum(dim=0)
                        value_loss = masked_mean(
                            value_loss_, time_mask, per_trace=True
                        ).mean()

                        selection_entropy = masked_mean(
                            entropies["selection_entropy"], time_mask, per_trace=True
                        ).mean()
                        conditional_entropy = masked_mean(
                            entropies["conditional_entropy"], time_mask, per_trace=True
                        ).mean()
                        entropy_loss = enable_entropy_loss * (
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
                        cond_term = (
                            advantages[..., None] * (log_ps[..., None] - sel_log_ps)
                        ).std()
                        ratio = cond_term / (sel_term + 1e-8)

                    # Discriminator:
                    # ==========================================
                    train_disc = np.random.rand() < (
                        1
                        - np.clip(
                            (disc_loss_thres + disc_loss_p_buffer - disc_ema_loss)
                            / disc_loss_p_buffer,
                            0,
                            1 - disc_train_min_p,
                        )
                    )

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

                    disc_ema_loss = ema_update(disc_ema_loss, disc_loss, ema_decay)

                    # Logging:
                    # =========================================
                    if train_obs:
                        losses_metrics_d["loss"].append(loss.item())
                        losses_metrics_d["policy_loss"].append(policy_loss.item())
                        losses_metrics_d["value_loss"].append(value_loss.item())
                        losses_metrics_d["sel_entropy"].append(selection_entropy.item())
                        losses_metrics_d["cond_entropy"].append(
                            conditional_entropy.item()
                        )

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

                    losses_metrics_d["train_disc"].append(1 if train_disc else 0)
                    losses_metrics_d["disc_loss"].append(disc_loss)
                    # (B, )
                    normal_packets = (
                        (Xobs[Feats.DIRS] != 0) & (Xobs[Feats.PADDING] == 0)
                    ).sum(dim=1)
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
                        "dloss": disc_ema_loss,
                        "d_tr_f": np.mean(losses_metrics_d["train_disc"][-nhist:]),
                        "pfu": np.mean(losses_metrics_d["mean_padding_frac_up"]),
                        "pfd": np.mean(losses_metrics_d["mean_padding_frac_down"]),
                    }
                    if losses_metrics_d["avg_return"]:
                        postfix["ret"] = np.mean(
                            losses_metrics_d["avg_return"][-nhist:]
                        )
                        postfix["Hs"] = ema_sel_entropy
                        postfix["Hc"] = ema_cond_entropy

                    pbar.set_postfix({k: f"{v:.03f}" for k, v in postfix.items()})

                    if disc_ema_loss < disc_loss_thres and obs_train_frac == 0:
                        pbar.close()
                        logger.info("Early termination")
                        obs_train_frac = 1.0
                        break

            # League handling:
            # =======================================

            # Append current discriminator to league
            disc_train_count += sum(losses_metrics_d["train_disc"])
            if disc_train_count > 25 and obs_train_frac > 0.0:
                disc_league = _append_to_league(disc_league, discriminator.state_dict())
                # mlflow.pytorch.log_model(
                #     discriminator, name=f"rldisc-{disc_league[-1][0]}", step=e
                # )
                disc_train_count = 0

                disc_loss_thres -= disc_loss_step
                if disc_loss_thres < min_disc_loss_thres:
                    disc_loss_thres = min_disc_loss_thres

            if (
                np.mean(np.array(losses_metrics_d["grad_norm"]) > cfg.grad_norm_clip)
                > 0.05
            ):
                logger.warning("Gradient clipping occurred often!")
            # obs_league = _append_to_league(obs_league, obs.state_dict())

            # Padding and entropy scale updates:
            # =============================================
            losses_metrics_d["entropy_scale"] = selection_entropy_scale
            if e < satlen:
                padding_scale += padding_scale_step

            losses_metrics_d["padding_scale"] = padding_scale

            if losses_metrics_d["entropy_loss"] and losses_metrics_d["policy_loss"]:
                el_vs_pl = np.mean(losses_metrics_d["entropy_loss"]) / np.mean(
                    losses_metrics_d["policy_loss"]
                )
                losses_metrics_d["entropy_loss vs. policy_loss"] = el_vs_pl

            # =============================================

            # Disc loss thres update:
            # =============================================
            losses_metrics_d["disc_l_thres"] = disc_loss_thres
            losses_metrics_d["obs_train_frac"] = obs_train_frac
            # if (d_train_frac := np.mean(losses_metrics_d["train_disc"])) > 0.90:
            #     if d_train_frac == 1:
            #         obs_train_frac *= 0.8
            # elif d_train_frac < 0.5:
            #     obs_train_frac = 1.0

            # disc_loss_thres -= disc_loss_step

            # Logging:
            # =============================================
            for k, v in losses_metrics_d.items():
                if isinstance(v, list):
                    if len(v) == 0:
                        continue
                    v = np.mean(v)
                elif isinstance(v, (int, float)):
                    pass
                else:
                    raise ValueError("Invalid value to be logged")

                k = keymap(k)
                logger.info(f"\t{k:<40} : {v:.03f}")

                mlflow.log_metric(k, v, step=e)

            if e % cfg.figs_period == 0:
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

                key = "valid:obs_vs._disc"
                d = valid_metrics(
                    disc=discriminator,
                    obs=obs,
                    ds_valid=ds_valid,
                    n_packets=cfg.trace_len,
                    device=device,
                    key=key,
                )

                d["selection_entropy_target"] = sel_entropy_target

                for k, v in d.items():
                    k = keymap(k)
                    mlflow.log_metric(k, v, step=e)
                    logger.info(f"\t{k:<40} : {v:.03f}")

            # =============================================
            if e % 5 == 0:
                mlflow.pytorch.log_model(obs, name=f"rlobs-{e}", step=e)

            if cfg.max_epochs > 0 and e >= cfg.max_epochs:
                logger.info("Max epochs %s reached!" % e)
                break

            e += 1


if __name__ == "__main__":
    main()
