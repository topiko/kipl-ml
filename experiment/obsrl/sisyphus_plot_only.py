from __future__ import annotations

import argparse
import copy
import random

import mlflow
import numpy as np
import torch
from hydra import compose, initialize_config_dir
from torch import nn
from tqdm import tqdm

from experiment.obsrl.plot_utils import _plot_single
from experiment.obsrl.sim import rollout
from experiment.obsrl.sisyphus import CONFIG_DIR_PATH, get_agent_and_critic
from experiment.obsrl.utils import get_advantages
from experiment.utils import defence_builder
from kipl_ml.data.utils import assets
from kipl_ml.data.wf_dataset import (
    WFDataset,
    dict_to_device,
    get_network_context_ranges,
    get_train_valid_test,
)
from kipl_ml.logging.logger import TQDM_W, get_logger
from kipl_ml.rl.enums import (
    ActDelayDown,
    ActDelayUp,
    Actions,
    ActSendUp,
    NoAction,
    StepAction,
)
from kipl_ml.trace.enums import Feats
from kipl_ml.trace.features import FeatureTrs, get_feature_tr

logger = get_logger(__name__)

TEST_XV = 0


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Sisyphus plot-only runner: no training, same rollout+figure path as sisyphus.py"
        )
    )
    ap.add_argument(
        "--cfg-overrides",
        nargs="*",
        default=[],
        help=(
            "Hydra config overrides, e.g. obs.enable_delay=true trace.n_packets=5000"
        ),
    )
    ap.add_argument(
        "--idxs",
        type=str,
        default="",
        help="Comma-separated dataset indices to plot. If empty, random sampling is used.",
    )
    ap.add_argument(
        "--ntraces",
        type=int,
        default=3,
        help="How many traces to sample when --idxs is not set.",
    )
    ap.add_argument(
        "--delay_between",
        type=str,
        nargs="+",
        default=None,
        metavar=("START_S", "END_S"),
        help=(
            "Force DELAY_UP/DELAY_DOWN=1 for action times in [START_S, END_S). "
            "Accepted forms: '--delay_between 0 2' or '--delay_between [0,2]'. "
            "This is applied on top of the model outputs."
        ),
    )
    ap.add_argument(
        "--action-policy",
        choices=(
            "model",
            "do-nothing",
            "occasional-send-up",
            "occasional-send-up-delay",
        ),
        default="model",
        help=(
            "Select the plot-only action override policy. 'model' keeps the agent"
            " outputs, 'do-nothing' forces NoAction, and 'occasional-send-up'"
            " keeps mostly NoAction with periodic SEND_UP actions."
        ),
    )
    ap.add_argument(
        "--outdir",
        type=str,
        default="experiment/obsrl/plot_only_out",
        help="Directory where figures are written.",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--show", dest="show", action="store_true")
    ap.add_argument("--no-show", dest="show", action="store_false")
    ap.set_defaults(show=False)
    ap.add_argument(
        "--backend",
        type=str,
        default="",
        help="Optional matplotlib backend override (e.g. TkAgg, QtAgg).",
    )
    ap.add_argument(
        "--verify-delay",
        dest="verify_delay",
        action="store_true",
        help="When --delay_between is set, run an extra baseline rollout and print diffs.",
    )
    ap.add_argument("--no-verify-delay", dest="verify_delay", action="store_false")
    ap.set_defaults(verify_delay=True)
    return ap.parse_args()


def _parse_delay_between(values: list[str] | None) -> tuple[float, float] | None:
    if values is None:
        return None

    parts: list[str]
    if len(values) == 1:
        s = values[0].strip().replace("[", "").replace("]", "")
        parts = [p.strip() for p in s.split(",") if p.strip()]
    else:
        parts = [p.strip() for p in values if p.strip()]

    if len(parts) != 2:
        raise ValueError(
            "--delay_between expects two values, e.g. '--delay_between 0 2' "
            "or '--delay_between [0,2]'"
        )

    return float(parts[0]), float(parts[1])


def _compose_cfg(overrides: list[str]):
    with initialize_config_dir(config_dir=CONFIG_DIR_PATH, version_base=None):
        cfg = compose(config_name="sisyphus", overrides=overrides)
    return cfg


def _build_disc_features(discriminator_orig: nn.Module, cfg) -> FeatureTrs:
    time_clamp = (
        (0.0, cfg.trace.dur_max_s, True) if cfg.trace.dur_max_s is not None else None
    )

    feature_names = discriminator_orig.features
    tam_d = dict(discriminator_orig.tam_dict)
    tam_d["max_load_time_s"] = cfg.disc.tam_max_load_time_s

    tam_ww = float(tam_d["window_width_s"])
    if abs(tam_ww - float(cfg.obs.time_step_s)) > 1e-6:
        raise ValueError(
            "TAM window_width_s must match obs.time_step_s for TAM reward mapping. "
            + f"Got tam_ww={tam_ww:.6f}s and obs.time_step_s={float(cfg.obs.time_step_s):.6f}s."
        )

    disc_trs = [
        get_feature_tr(fn, None, time_clamp, tam_kwargs=tam_d) for fn in feature_names
    ]
    disc_trs += [
        get_feature_tr(f, None, time_clamp, tam_kwargs=tam_d)
        for f in (Feats.TAM_DOWN_DECOY, Feats.TAM_UP_DECOY)
    ]
    disc_trs.append(get_feature_tr(Feats.TAM_BINS, None, time_clamp, tam_kwargs=tam_d))
    return FeatureTrs(feature_trs=disc_trs)


def _load_dataset(cfg) -> tuple[WFDataset, WFDataset]:
    time_clamp = (
        (0.0, cfg.trace.dur_max_s, True) if cfg.trace.dur_max_s is not None else None
    )

    ds_train, ds_valid, _ = get_train_valid_test(
        dataset=cfg.dataset.name,
        label=assets.PAGE_LABEL,
        n_splits=cfg.dataset.n_splits,
        test_xv=TEST_XV,
        random_state=42,
        feature_trs=FeatureTrs(
            feature_names=[Feats.DIRS, Feats.TIMES],
            n_packets=cfg.trace.n_packets,
            time_clamp=time_clamp,
        ),
        defence_aug_valid=0,
        n_min_packets=cfg.trace.min_packets,
        trim_raw=cfg.trace.trim_beginning,
        **get_network_context_ranges(cfg),
        **defence_builder.get_defence(cfg),
    )
    return ds_train, ds_valid


def _install_delay_between_override(obs, start_s: float, end_s: float) -> None:
    if start_s >= end_s:
        raise ValueError("--delay_between requires START_S < END_S")
    if not getattr(obs, "enable_delay", False):
        raise ValueError("Delay override requested, but obs.enable_delay is False")

    dt_s = float(obs.time_step_s)
    original_act = obs.act

    def _act_override(
        x: dict[Feats, torch.Tensor],
        h: torch.Tensor | None = None,
        h_detach_period: int | None = None,
        seq_lens: torch.Tensor | None = None,
        sample: bool = True,
    ):
        (
            action_times,
            actions,
            log_probs,
            sel_probs,
            values,
            entropies,
            h_out,
        ) = original_act(
            x,
            h=h,
            h_detach_period=h_detach_period,
            seq_lens=seq_lens,
            sample=sample,
        )

        action_times_s = action_times.to(torch.float32) * dt_s
        mask = (
            (action_times >= 0)
            & (action_times_s >= float(start_s))
            & (action_times_s < float(end_s))
        )

        # Enforce delay only inside the requested interval.
        actions[Actions.DELAY_UP] = torch.where(
            mask,
            torch.ones_like(actions[Actions.DELAY_UP]),
            torch.zeros_like(actions[Actions.DELAY_UP]),
        )
        actions[Actions.DELAY_DOWN] = torch.where(
            mask,
            torch.ones_like(actions[Actions.DELAY_DOWN]),
            torch.zeros_like(actions[Actions.DELAY_DOWN]),
        )

        # If policy selected delay outside the forced interval, convert to do-nothing.
        outside_delay = (~mask) & (actions[Actions.SELECTOR] >= 4)
        if bool(outside_delay.any().item()):
            actions[Actions.SELECTOR][outside_delay] = 0
            actions[Actions.DO_NOTHING][outside_delay] = 1
            actions[Actions.SEND_UP][outside_delay] = 0
            actions[Actions.SEND_DOWN][outside_delay] = 0
            actions[Actions.DELAY_UP][outside_delay] = 0
            actions[Actions.DELAY_DOWN][outside_delay] = 0

            sel_probs = sel_probs.clone()
            sel_probs[outside_delay] = 0
            sel_probs[..., 0][outside_delay] = 1

            log_probs = log_probs.clone()
            log_probs[outside_delay] = 0.0

            entropies = {k: v.clone() for k, v in entropies.items()}
            entropies["selection_entropy"][outside_delay] = 0.0
            entropies["conditional_entropy"][outside_delay] = 0.0

        if bool(mask.any().item()):
            actions[Actions.DO_NOTHING][mask] = 0
            actions[Actions.SEND_UP][mask] = 0
            actions[Actions.SEND_DOWN][mask] = 0
            actions[Actions.DELAY_UP][mask] = 1
            actions[Actions.DELAY_DOWN][mask] = 1
            actions[Actions.SELECTOR][mask] = 6

            sel_probs = sel_probs.clone()
            sel_probs[mask] = 0
            sel_probs[..., 4][mask] = 1

            log_probs = log_probs.clone()
            log_probs[mask] = 0.0

            entropies = {k: v.clone() for k, v in entropies.items()}
            entropies["selection_entropy"][mask] = 0.0
            entropies["conditional_entropy"][mask] = 0.0

        return action_times, actions, log_probs, sel_probs, values, entropies, h_out

    obs.act = _act_override


def _install_do_nothing_override(obs) -> None:
    original_act = obs.act

    def _act_override(
        x: dict[Feats, torch.Tensor],
        h: torch.Tensor | None = None,
        h_detach_period: int | None = None,
        seq_lens: torch.Tensor | None = None,
        sample: bool = True,
    ):
        action_times, actions, log_probs, sel_probs, values, entropies, h_out = (
            original_act(
                x,
                h=h,
                h_detach_period=h_detach_period,
                seq_lens=seq_lens,
                sample=sample,
            )
        )

        times = action_times.reshape(-1)
        send_up_mask = (times >= 0) & ((times % 4) == 0)
        actions = [
            NoAction(time=int(t.item()))
            if not bool(send_up_mask[i].item())
            else StepAction(
                time_bin=int(t.item()),
                _actions={Actions.SEND_UP: ActSendUp(count=100, after_steps=0)},
            )
            for i, t in enumerate(times)
        ]
        return action_times, actions, log_probs, sel_probs, values, entropies, h_out

    obs.act = _act_override


def _install_action_policy_override(obs, policy: str) -> None:
    if policy == "model":
        logger.info("Action policy override disabled; using model outputs")
        return
    if policy == "do-nothing":
        original_act = obs.act

        def _act_override(
            x: dict[Feats, torch.Tensor],
            h: torch.Tensor | None = None,
            h_detach_period: int | None = None,
            seq_lens: torch.Tensor | None = None,
            sample: bool = True,
        ):
            action_times, _, log_probs, sel_probs, values, entropies, h_out = (
                original_act(
                    x,
                    h=h,
                    h_detach_period=h_detach_period,
                    seq_lens=seq_lens,
                    sample=sample,
                )
            )

            actions = [NoAction(time=int(t.item())) for t in action_times.reshape(-1)]
            return action_times, actions, log_probs, sel_probs, values, entropies, h_out

        obs.act = _act_override
        logger.info("Do-nothing override active for all actions")
        return

    if policy == "occasional-send-up":
        _install_do_nothing_override(obs)
        logger.info("Do-nothing override active with occasional SEND_UP actions")
        return

    if policy == "occasional-send-up-delay":
        original_act = obs.act

        def _act_override(
            x: dict[Feats, torch.Tensor],
            h: torch.Tensor | None = None,
            h_detach_period: int | None = None,
            seq_lens: torch.Tensor | None = None,
            sample: bool = True,
        ):
            action_times, _, log_probs, sel_probs, values, entropies, h_out = (
                original_act(
                    x,
                    h=h,
                    h_detach_period=h_detach_period,
                    seq_lens=seq_lens,
                    sample=sample,
                )
            )

            times = action_times.reshape(-1)
            send_up_mask = (times >= 0) & ((times % 4) == 0)
            delay_mask = (times >= 0) & ((times % 6) == 0)
            actions = []
            for i, t in enumerate(times):
                if not bool(send_up_mask[i].item()) and not bool(delay_mask[i].item()):
                    actions.append(NoAction(time=int(t.item())))
                    continue

                action_dict = {}
                if bool(send_up_mask[i].item()):
                    action_dict[Actions.SEND_UP] = ActSendUp(count=100, after_steps=0)
                if bool(delay_mask[i].item()):
                    action_dict[Actions.DELAY_UP] = ActDelayUp(steps=2)
                    action_dict[Actions.DELAY_DOWN] = ActDelayDown(steps=3)
                actions.append(StepAction(time_bin=int(t.item()), _actions=action_dict))

            return action_times, actions, log_probs, sel_probs, values, entropies, h_out

        obs.act = _act_override
        logger.info(
            "Do-nothing override active with occasional SEND_UP and DELAY actions"
        )
        return

    raise ValueError(f"Unknown action policy: {policy}")


def _choose_indices(ds: WFDataset, idxs_s: str, ntraces: int, seed: int) -> list[int]:
    if idxs_s.strip():
        idxs = [int(s) for s in idxs_s.split(",") if s.strip()]
        if not idxs:
            raise ValueError("--idxs was provided but no valid indices were parsed")
        return idxs

    rng = np.random.default_rng(seed=seed)
    n = min(int(ntraces), len(ds))
    return [int(i) for i in rng.choice(len(ds), size=n, replace=False).tolist()]


def _plot_indices(
    *,
    cfg,
    ds: WFDataset,
    idxs: list[int],
    obs,
    critic,
    obs_features: FeatureTrs | None,
    disc_orig,
    disc_trained,
    disc_features: FeatureTrs,
    active_disc_league,
    weights: torch.Tensor,
    reward_scales: dict[str, float],
    device: torch.device,
    epoch_tag: int,
) -> None:
    disc_orig.eval()
    disc_trained.eval()
    obs.eval()
    if critic is not None:
        critic.eval()

    X_orig_l: list[dict[Feats, torch.Tensor]] = []
    y_l: list[torch.Tensor] = []
    X_rollin_l: list[dict[Feats, torch.Tensor]] = []

    for idx in idxs:
        X_i, y_i = ds[int(idx)]
        X_orig_l.append(X_i)
        y_l.append(y_i)
        X_rollin_l.append(obs_features(X_i) if obs_features is not None else X_i)

    keys = list(X_rollin_l[0].keys())
    X_batch = {k: torch.stack([x[k] for x in X_rollin_l], dim=0) for k in keys}
    y_batch = torch.stack(y_l, dim=0)

    X_batch = dict_to_device(X_batch, device)
    y_batch = y_batch.to(device)

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
            add_tail_s=cfg.obs.train.add_tail_s,
        )

    action_seq_lens = fd[Feats.SEQ_LENS]
    G, advantages = get_advantages(league_rewards, values, action_seq_lens, cfg)

    Xd_batch = disc_features.transform_batch(X_batch)
    Xd_batch = dict_to_device(Xd_batch, device)
    logits_d, _ = disc_orig(Xd_batch)
    probs_d = nn.functional.softmax(logits_d, dim=-1)

    X_obs_d = disc_features.transform_batch(X_obs)
    X_obs_d = dict_to_device(X_obs_d, device)
    logits_o, _ = disc_trained(X_obs_d)
    probs_o = nn.functional.softmax(logits_o, dim=-1)

    with tqdm(list(enumerate(idxs)), desc="Plot traces", ncols=TQDM_W) as pbar:
        for batch_i, ds_idx in pbar:
            _plot_single(
                cfg=cfg,
                disc_orig=disc_orig,
                disc_trained=disc_trained,
                disc_features=disc_features,
                e=epoch_tag,
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
                show=True,
            )


def main() -> None:
    args = _parse_args()
    delay_between = _parse_delay_between(args.delay_between)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    cfg = _compose_cfg(args.cfg_overrides)

    if delay_between is not None:
        cfg.obs.enable_delay = True

    model_id = cfg.disc.no_defense_disc_id
    discriminator_orig = mlflow.pytorch.load_model(
        mlflow.get_logged_model(model_id).model_uri,
        map_location="cpu",
    )
    discriminator = mlflow.pytorch.load_model(
        mlflow.get_logged_model(model_id).model_uri,
        map_location="cpu",
    )

    disc_features = _build_disc_features(discriminator_orig, cfg)
    ds_train, ds_valid = _load_dataset(cfg)

    obs, critic = get_agent_and_critic(cfg)
    obs_base = None
    if delay_between is not None and args.verify_delay:
        obs_base = copy.deepcopy(obs)

    if delay_between is not None:
        dense_step_s = float(obs.time_step_s)
        if abs(float(obs.max_silence_s) - dense_step_s) > 1e-12:
            logger.info(
                "delay_between active: overriding max_silence_s %.6f -> %.6f "
                + "to enforce per-bin action windows",
                float(obs.max_silence_s),
                dense_step_s,
            )
            obs.max_silence_s = dense_step_s
            if obs_base is not None:
                obs_base.max_silence_s = dense_step_s

    if delay_between is not None:
        _install_delay_between_override(
            obs,
            start_s=float(delay_between[0]),
            end_s=float(delay_between[1]),
        )
        logger.info(
            "Forced delay override active for action times in [%.3f, %.3f) s",
            float(delay_between[0]),
            float(delay_between[1]),
        )

    _install_action_policy_override(obs, args.action_policy)

    if args.device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but unavailable; falling back to CPU")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    discriminator = discriminator.to(device)
    discriminator_orig = discriminator_orig.to(device)
    obs = obs.to(device)
    if obs_base is not None:
        obs_base = obs_base.to(device)
    if critic is not None:
        critic = critic.to(device)

    reward_scales = {
        "clf_scale": 0.1,
        "d_clf_scale": cfg.rewards.d_clf,
        "decoy_scale": cfg.rewards.decoy_scale,
        "delay_scale": cfg.rewards.delay_scale,
    }

    active_disc_league = [(0, None)]
    weights = torch.ones((1,), device=device)

    idxs = _choose_indices(ds_valid, args.idxs, args.ntraces, args.seed)
    logger.info("Plotting idxs=%s", idxs)

    if (
        delay_between is not None
        and args.verify_delay
        and obs_base is not None
        and idxs
    ):
        _verify_delay_effect(
            ds=ds_valid,
            idx=int(idxs[0]),
            obs_delay=obs,
            obs_base=obs_base,
            critic=critic,
            obs_features=ds_train.feature_trs,
            disc=discriminator,
            disc_features=disc_features,
            reward_scales=reward_scales,
            device=device,
            dt_s=float(cfg.obs.time_step_s),
            delay_between=delay_between,
            max_dur_s=cfg.trace.dur_max_s,
        )

    _plot_indices(
        cfg=cfg,
        ds=ds_valid,
        idxs=idxs,
        obs=obs,
        critic=critic,
        obs_features=ds_train.feature_trs,
        disc_orig=discriminator_orig,
        disc_trained=discriminator,
        disc_features=disc_features,
        active_disc_league=active_disc_league,
        weights=weights,
        reward_scales=reward_scales,
        device=device,
        epoch_tag=0,
    )


if __name__ == "__main__":
    main()
