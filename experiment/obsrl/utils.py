import dotenv
import torch
from torch import nn
from tqdm import tqdm

from kipl_ml.data.wf_dataset import dict_to_device
from kipl_ml.logging.logger import TQDM_W, get_logger
from kipl_ml.trace.features import Feats, FeatureTrs

logger = get_logger(__name__)
dotenv.load_dotenv()


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

            break

    return loss_mean


def one_batch_train_disc(
    disc: nn.Module,
    X: dict[Feats, torch.Tensor],
    y: torch.Tensor,
    disc_opm: torch.optim.Optimizer,
    feature_trs: FeatureTrs | None,
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

    h = None
    i = 0
    loss_mean = 0.0
    context = torch.enable_grad() if train else torch.inference_mode()
    max_len = (X[Feats.DIRS] != 0).sum(dim=1).max()
    with context:
        while True:
            if i * detach_period >= max_len - 1:
                break
            disc_opm.zero_grad()

            X_chunk = {
                k: v[:, i * detach_period : (i + 1) * detach_period]
                for k, v in X.items()
            }
            seq_lens = (X_chunk[Feats.DIRS] != 0).sum(dim=1)

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
