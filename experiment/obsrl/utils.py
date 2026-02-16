import copy

import dotenv
import numpy as np
import torch
from omegaconf import DictConfig
from torch import nn
from torch.utils.data import SubsetRandomSampler
from tqdm import tqdm

from experiment.obsrl.sim import rollout
from experiment.trace_gan.data_utils import dl_
from kipl_ml.data.wf_dataset import WFDataset, dict_to_device
from kipl_ml.defences.base import NoDefence
from kipl_ml.defences.nndefs import RNNDef
from kipl_ml.logging.logger import TQDM_W, get_logger
from kipl_ml.metrics.clf_metrics import Accuracy
from kipl_ml.model_eval.evaluate import evaluate_model
from kipl_ml.rl.advantages import get_gae, get_returns
from kipl_ml.trace.features import Feats, FeatureTrs

logger = get_logger(__name__)
dotenv.load_dotenv()


def keymap(key: str) -> str:
    if "_ms" in key:
        key = f"timings / {key}"
    if "loss" in key:
        key = f"losses / {key}"
    if "entropy" in key:
        key = f"entropies / {key}"

    return key


def _append_to_league(
    league: list[tuple[int, dict]], state_dict: dict
) -> list[tuple[int, nn.Module.state_dict]]:
    if league:
        id_ = max(id_ for id_, _ in league) + 1
    else:
        id_ = 0
    league.append((id_, {k: v.cpu() for k, v in copy.deepcopy(state_dict).items()}))

    return league


def _get_obs_def_dl(
    disc: nn.Module,
    disc_feats: FeatureTrs,
    obs: nn.Module,
    ds: WFDataset,
    n_packets: int,
    bs: int = 64,
    obs_league: list[nn.Module.state_dict] | None = None,
    sampler: SubsetRandomSampler | None = None,
) -> WFDataset:
    # Set features the fetures:
    ds.feature_trs = disc_feats

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

    return dl_(
        ds, bs=bs, collate_fn=None, shuffle=False, nworkers=None, sampler=sampler
    )


def _restore_obs_def_ds(
    ds: WFDataset,
    feature_trs: FeatureTrs | None,
    obs: nn.Module,
    device: torch.DeviceObjType,
):
    # Restore no defence
    ds.defence = NoDefence(network_delay_millis=(25, 250), network_pps=(40_000, 40_000))
    # Restore no features.
    ds.feature_trs = feature_trs

    obs = obs.to(device)


def valid_metrics(
    disc: nn.Module,
    disc_feats: FeatureTrs,
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
        disc_feats=disc_feats,
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


def get_league_scores(
    league: list[tuple[int, nn.Module]],
    ds: WFDataset,
    obs: nn.Module,
    critic: nn.Module,
    obs_features: FeatureTrs | None,
    disc: nn.Module,
    disc_features: FeatureTrs,
    reward_scales: dict[str, float],
    device: torch.DeviceObjType,
    subset_indices: torch.Tensor,
    score_type: str = "acc",
    n_packets: int | None = None,
) -> torch.Tensor:
    # Set the defence and features:
    orig_features = ds.feature_trs
    obs.eval()
    sampler = SubsetRandomSampler(subset_indices)
    if score_type == "acc":
        if n_packets is None:
            raise ValueError("Provide npackets")
        dl = _get_obs_def_dl(
            disc=disc,
            disc_feats=disc_features,
            obs=obs,
            ds=ds,
            n_packets=n_packets,
            bs=32,
            obs_league=None,
            sampler=sampler,
        )

    elif score_type == "neg_rewards":
        ds.feature_trs = obs_features
        dl = dl_(
            ds, bs=64, collate_fn=None, shuffle=False, nworkers=None, sampler=sampler
        )

    with torch.no_grad():
        if score_type == "acc":
            orig_state_d = {k: v.detach().clone() for k, v in disc.state_dict().items()}
            scores_ = []
            for _, state_d in league:
                disc.load_state_dict(state_d)
                d = evaluate_model(disc, dl, metrics=[Accuracy()])
                scores_.append(d["accuracy"])

            disc.load_state_dict(orig_state_d)
            league_scores = torch.tensor(scores_).to(device)

            _restore_obs_def_ds(ds, orig_features, obs, device)

        elif score_type == "neg_rewards":
            with tqdm(
                dl,
                desc="league scoring",
                ncols=TQDM_W,
            ) as pbar:
                rewards_l = []
                for X, y in pbar:
                    X = dict_to_device(X, device)
                    y = y.to(device)
                    _, league_rewards, _, _, _, _, fd = rollout(
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
                            [
                                masked_mean_std(v[i], time_mask)[0]
                                for i in range(v.shape[0])
                            ]
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


def masked_mean(
    x: torch.Tensor, mask: torch.Tensor, per_trace: bool = False, eps: float = 1e-8
) -> torch.Tensor:
    if isinstance(x, dict):
        return masked_mean(sum(x.values()), mask, per_trace, eps)
    return masked_mean_std(x, mask, per_trace, eps)[0]


def masked_mean_std(
    x: torch.Tensor, mask: torch.Tensor, per_trace: bool = False, eps: float = 1e-8
) -> tuple[torch.Tensor, torch.Tensor]:
    m = mask.to(dtype=x.dtype)

    if per_trace:
        seq_lens = m.sum(dim=1)
        mean = (x * m).sum(dim=1) / seq_lens
        var = ((x - mean[:, None]) * m).pow(2).sum(dim=1) / seq_lens
        std = (var + eps).sqrt()
        # (bs, )
        return mean, std

    denom = m.sum().clamp(min=1.0)
    mean = (x * m).sum() / denom
    var = ((x - mean) * m).pow(2).sum() / denom
    std = (var + eps).sqrt()
    # (, )
    return mean, std


def _get_optim(
    nn: nn.Module, lr: float, lr_rnn: float | None = None
) -> torch.optim.Optimizer:
    lr_rnn = lr_rnn or lr
    rnn_params = list(nn.rnn.parameters())
    other_params = [p for n, p in nn.named_parameters() if not n.startswith("rnn.")]
    return torch.optim.Adam(
        [
            {"params": rnn_params, "lr": lr_rnn},
            {"params": other_params, "lr": lr},
        ]
    )


def log_lrs(lr_scheduler: torch.optim.lr_scheduler.LRScheduler):
    lrs = lr_scheduler.get_last_lr()

    for i, lr_ in enumerate(lrs):
        logger.info(f"\tlr group {i:2d}: {lr_:.4f}")


def get_active_league(
    active_league_idx: np.ndarray | None,
    league: list[tuple[int, nn.Module.state_dict]],
    ds: WFDataset,
    obs: nn.Module,
    critic: nn.Module,
    obs_feats: FeatureTrs | None,
    disc: nn.Module,
    disc_feats: FeatureTrs,
    reward_scales: dict[str, float],
    device: torch.DeviceObjType,
    league_size: int,
    league_update_frac: float,
    prune: bool = False,
    score_type: str = "acc",
    n_packets: int | None = None,
) -> tuple[
    list[tuple[int, nn.Module.state_dict]],
    np.ndarray,
    list[nn.Module.state_dict],
    torch.Tensor,
]:
    rng = np.random.default_rng()
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
        subset_indices=rng.choice(np.arange(len(ds)), 500, replace=False),
        score_type=score_type,
        n_packets=n_packets,
    )

    if prune:
        logger.info("Pruning disc league.")
        val = league_scores.min().item()
        mask = league_scores > val
        # The latest disc shall not be removed..
        mask[-1] = True
        league_scores = league_scores[mask]
        league = [l_ for i, l_ in enumerate(league) if mask[i]]

    # We ensure that the latest disc is always in the leaque
    cur_disc_pos = len(league) - 1
    cur_disc_id = league[-1][0]

    if active_league_idx is None:
        active_league_idx = np.random.choice(
            len(league), min(len(league), league_size), replace=False
        )
        active_league_idx[-1] = cur_disc_pos

    if league_size == 1:
        active_league_idx = np.array([cur_disc_pos])

    elif len(league) > league_size:
        scores = np.array(league_scores.cpu().numpy())
        if len(np.unique(scores)) == 1:
            logger.warning("All scores equal!?")
            probs = np.ones_like(scores) / len(scores)
        else:
            probs = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
            probs /= probs.sum()

        # probs = np.clip(probs, 1e-6, 1.0)
        active_league_idx = np.random.choice(
            len(league), league_size, p=probs, replace=False
        )
    else:
        active_league_idx = np.arange(len(league))

    active_league_idx.sort()
    # =======================================

    active_league = [league[i] for i in active_league_idx]

    # If the latest disc is already in the active_league
    if cur_disc_pos in active_league_idx:
        cur_disc_idx_ = np.where(active_league_idx == cur_disc_pos)[0]
        if len(cur_disc_idx_) != 1:
            raise ValueError("Disc several times in league!?")
        if cur_disc_idx_ != len(active_league_idx) - 1:
            raise KeyError("Cur disc at wrong position")
        cur_disc_idx_ = cur_disc_idx_[0]
    else:
        cur_disc_idx_ = -1

    # The acive disc is a special one in the league..
    active_league[cur_disc_idx_] = (cur_disc_id, None)
    active_league_idx[cur_disc_idx_] = cur_disc_pos

    if len(active_league_idx) > 1:
        weights = league_scores[active_league_idx]

        if score_type != "acc":
            weights -= weights.min()
            if weights.max() == 0:
                logger.warning("Same score for several discs!")
                weights = torch.ones_like(weights) / weights.numel()
            else:
                weights /= weights.max()
            weights = torch.clamp(weights, 1 / (10 * league_size), 1.0)
    else:
        weights = torch.tensor([1.0], device=device)

    weights /= weights.sum()

    logger.info(f"League scores ({score_type}):")
    for i, s in enumerate(league_scores):
        str_ = "           "
        if i in active_league_idx:
            w = weights[active_league_idx == i][0].item()
            str_ = f"* [w={w:.03f}]"

        logger.info(f"\t{i:4d} == {league[i][0]:4d}{str_} : {s:.4f}")

    return league, active_league_idx, active_league, weights


def get_action_seq_lens(fd: dict[Feats, torch.Tensor]) -> torch.Tensor:
    return fd[Feats.TIMES].isnan().logical_not().sum(dim=1)


def ema_update(value: float | None, cur_value: float, ema_decay: float) -> float:
    if value is None:
        return cur_value
    return ema_decay * value + (1 - ema_decay) * cur_value


def league_rewards2rewards(
    league_rewards: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    return {k: v.mean(dim=0) for k, v in league_rewards.items()}


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
        if cfg.advantages.standardize and cfg.league_size > 1:
            raise NotImplementedError("You should not standardize here for league")

        G_l = []
        advantages_l = []
        for i in range(rewards.shape[0]):
            G_, advantages_ = get_advantages(rewards[i], values, seq_lens, cfg)
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

    if cfg.advantages.divide_by_Z:
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
            seq_lens.to(values.device), values.shape[0], device=values.device
        )
        mean, std = masked_mean_std(advantages, mask, per_trace=True)
        advantages = (advantages) / std
        # Optional: keep padding at 0
        advantages = advantages * mask.to(advantages.dtype)

    return G, advantages


def train_disc_one_epoch(
    clf: nn.Module,
    dl_train: torch.utils.data.DataLoader,
    optimG: torch.optim.Optimizer,
    device: torch.DeviceObjType,
    grad_clip: float,
    detach_period: int = 10000,
    epoch: int = -1,
) -> float:
    loss_mean = 0.0
    n = 1
    clf.train()
    with tqdm(dl_train, desc=f"Train disc {epoch:02d}", ncols=TQDM_W) as pbar:
        for X, y in pbar:
            X = dict_to_device(X, device, non_blocking=True)
            y = y.to(device)

            loss, _ = train_disc_one_batch(
                clf,
                X,
                y,
                optimG,
                feature_trs=None,
                train=True,
                grad_clip=grad_clip,
                detach_period=detach_period,
                get_accuracy=False,
            )

            loss_mean += (loss - loss_mean) / n

            pbar.set_postfix({"l": loss_mean})

            n += 1

    return loss_mean


def train_disc_one_batch(
    disc: nn.Module,
    X: dict[Feats, torch.Tensor],
    y: torch.Tensor,
    disc_opm: torch.optim.Optimizer,
    feature_trs: FeatureTrs | None,
    seq_lens: torch.Tensor | None = None,
    train: bool = True,
    grad_clip: float = 3.0,
    detach_period: int = 1000,
    get_accuracy: bool = True,
) -> tuple[float, float | None]:
    if train:
        disc.train()
    else:
        disc.eval()
        detach_period = X[Feats.DIRS].shape[1]  # No detaching needed

    if feature_trs is not None:
        X = feature_trs.transform_batch(X)

    if seq_lens is None:
        seq_lens = disc.seq_len_fun(X)

    h = None
    i = 0
    loss_mean = 0.0
    context = torch.enable_grad() if train else torch.inference_mode()
    max_len = seq_lens.max()
    with context:
        while True:
            if i * detach_period >= max_len - 1:
                break
            disc_opm.zero_grad()

            X_chunk = {
                k: v[:, i * detach_period : (i + 1) * detach_period]
                for k, v in X.items()
            }
            chunk_seq_lens = (seq_lens - i * detach_period).clamp(
                min=0, max=detach_period
            )  # (X_chunk[Feats.DIRS] != 0).sum(dim=1)

            logits, h = disc.pack_and_forward(X_chunk, h, chunk_seq_lens.cpu())

            if logits is None:
                breakpoint()
                raise ValueError("None logits...")

            h = tuple(h_.detach() for h_ in h)

            # (B, T)
            target = y.unsqueeze(-1).repeat(1, logits.shape[1])
            # Use PAD value to ignore loss on padded tokens

            # (B, T)
            mask = torch.arange(
                logits.shape[1], device=chunk_seq_lens.device
            ).unsqueeze(0) >= (chunk_seq_lens + (chunk_seq_lens != 0)).unsqueeze(
                1
            )  # +1 for EOS

            # Where we are over seq. len --> ignore
            target = target.masked_fill(mask, -100)

            # Only operate on the seqs. that still are valid
            target = target[chunk_seq_lens > 0]

            loss = nn.functional.cross_entropy(
                logits.permute(0, 2, 1),
                target,
                ignore_index=-100,
            )

            if train:
                loss.backward()

                # Gradient clipping
                try:
                    nn.utils.clip_grad_norm_(
                        disc.parameters(), grad_clip, error_if_nonfinite=True
                    )
                except RuntimeError:
                    print("Nan in grads")
                    breakpoint()
                    return loss_mean, None

                disc_opm.step()

            i += 1
            loss_mean += (loss.item() - loss_mean) / i

    disc.eval()
    accuracy = None
    if get_accuracy:
        accuracy = (disc.predict(X)[1] == y).float().mean().item()

    return loss_mean, accuracy
