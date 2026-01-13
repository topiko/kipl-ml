import dotenv
import torch
from torch import nn

from kipl_ml.logging.logger import get_logger
from kipl_ml.trace.features import Feats

logger = get_logger(__name__)
dotenv.load_dotenv()


def one_batch_train_disc(
    disc: nn.Module,
    X: dict[Feats, torch.Tensor],
    y: torch.Tensor,
    disc_opm: torch.optim.Optimizer,
    train: bool = True,
    grad_clip: float = 3.0,
    detach_period: int = 1000,
    get_accuracy: bool = True,
) -> tuple[float, float | None]:
    if train:
        disc.train()

        h = None
        i = 0
        while True:
            if i * detach_period >= X[Feats.DIRS].shape[1] - 1:
                break
            disc_opm.zero_grad()

            X_chunk = {
                k: v[:, i * detach_period : (i + 1) * detach_period]
                for k, v in X.items()
            }
            seq_lens = (X_chunk[Feats.DIRS] != 0).sum(dim=1)

            logits, h = disc.pack_and_forward(X_chunk, h, seq_lens.cpu())

            if logits is None:
                raise ValueError("None logits...")

            h = tuple(h_.detach() for h_ in h)

            # (B, T)
            target = y.unsqueeze(-1).repeat(1, logits.shape[1])
            # Use PAD value to ignore loss on padded tokens

            # (B, T)
            mask = torch.arange(logits.shape[1], device=seq_lens.device).unsqueeze(
                0
            ).repeat(logits.shape[0], 1) >= seq_lens.unsqueeze(1)

            # Where we are over seq. len --> ignore
            target = target.masked_fill(mask, -100)

            # Only operate on the seqs. that still are valid
            target = target[seq_lens != 0]

            loss = nn.functional.cross_entropy(
                logits.permute(0, 2, 1),
                target,
                ignore_index=-100,
            )
            loss.backward()

            # Gradient clipping
            nn.utils.clip_grad_norm_(
                disc.parameters(), grad_clip, error_if_nonfinite=False
            )

            disc_opm.step()
            i += 1
    else:
        disc.eval()
        logits, _ = disc(X)
        loss = nn.functional.cross_entropy(
            logits.permute(0, 2, 1), y.unsqueeze(-1).repeat(1, logits.shape[1])
        )

    loss_val = loss.item()
    disc.eval()

    accuracy = None
    if get_accuracy:
        accuracy = (disc.predict(X)[1] == y).float().mean().item()

    return loss_val, accuracy
