import dotenv
import torch
from omegaconf import DictConfig
from torch import nn
from tqdm import tqdm

from kipl_ml.data.wf_dataset import dict_to_device
from kipl_ml.logging.logger import TQDM_W, get_logger
from kipl_ml.rl.advantages import get_gae, get_returns
from kipl_ml.trace.features import Feats, FeatureTrs

logger = get_logger(__name__)
dotenv.load_dotenv()


def get_action_seq_lens(fd: dict[Feats, torch.Tensor]) -> torch.Tensor:
    return fd[Feats.TIMES].isnan().logical_not().sum(dim=1)


def ema_update(value: float, cur_value: float, ema_decay: float) -> float:
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


def train_one_epoch(
    clf: nn.Module,
    dl_train: torch.utils.data.DataLoader,
    optimG: torch.optim.Optimizer,
    device: torch.DeviceObjType,
    grad_clip: float,
    detach_period: int = 10000,
) -> float:
    loss_mean = 0.0
    n = 1
    clf.train()
    with tqdm(dl_train, desc="Train disc:", ncols=TQDM_W) as pbar:
        for X, y in pbar:
            X = dict_to_device(X, device, non_blocking=True)
            y = y.to(device)

            loss, _ = one_batch_train_disc(
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


def one_batch_train_disc(
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
            seq_lens = (seq_lens - i * detach_period).clamp(
                min=0
            )  # (X_chunk[Feats.DIRS] != 0).sum(dim=1)

            logits, h = disc.pack_and_forward(X_chunk, h, seq_lens.cpu())

            if logits is None:
                breakpoint()
                raise ValueError("None logits...")

            h = tuple(h_.detach() for h_ in h)

            # (B, T)
            target = y.unsqueeze(-1).repeat(1, logits.shape[1])
            # Use PAD value to ignore loss on padded tokens

            # (B, T)
            mask = torch.arange(logits.shape[1], device=seq_lens.device).unsqueeze(
                0
            ) >= (seq_lens + (seq_lens != 0)).unsqueeze(1)  # +1 for EOS

            # Where we are over seq. len --> ignore
            target = target.masked_fill(mask, -100)

            # Only operate on the seqs. that still are valid
            target = target[seq_lens > 0]

            loss = nn.functional.cross_entropy(
                logits.permute(0, 2, 1),
                target,
                ignore_index=-100,
            )

            if train:
                loss.backward()

                # Gradient clipping
                nn.utils.clip_grad_norm_(
                    disc.parameters(), grad_clip, error_if_nonfinite=False
                )

                disc_opm.step()

            i += 1
            loss_mean += (loss.item() - loss_mean) / i

    disc.eval()
    accuracy = None
    if get_accuracy:
        accuracy = (disc.predict(X)[1] == y).float().mean().item()

    return loss_mean, accuracy
