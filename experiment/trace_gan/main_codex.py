import math
import os
from dataclasses import dataclass

import dotenv
import hydra
import mlflow
import torch
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from kipl_ml.data.utils import load_dataset_meta_df
from kipl_ml.data.wf_dataset import WFDataset, dict_to_device
from kipl_ml.defences.base import NoDefence
from kipl_ml.logging.logger import TQDM_W, get_logger
from kipl_ml.models.df import DF
from kipl_ml.models.trgen import TRGENResNet
from kipl_ml.models.utils import count_parameters
from kipl_ml.trace.features import Feats, FeatureTrs

logger = get_logger(__name__)
dotenv.load_dotenv()

WORKING_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_DIR_PATH = os.path.join(WORKING_DIR, "config")

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
if MLFLOW_TRACKING_URI:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
else:
    logger.warning(
        "MLFLOW_TRACKING_URI not set; continuing without MLflow tracking for this run."
    )


class RandomTraceDataset(Dataset):
    """Lightweight synthetic dataset for quick GAN smoke tests."""

    def __init__(
        self,
        trace_len: int,
        num_samples: int,
        n_classes: int,
        seed: int = 13,
    ) -> None:
        super().__init__()
        self.trace_len = trace_len
        self.num_samples = num_samples
        self.n_classes = n_classes
        self.generator = torch.Generator().manual_seed(seed)
        self.data = self._generate_samples()
        self.labels = torch.randint(
            0, n_classes, (num_samples,), generator=self.generator
        )

    def _generate_samples(self) -> torch.Tensor:
        base_noise = torch.randn(
            self.num_samples, self.trace_len, generator=self.generator
        )
        cumulative = torch.cumsum(base_noise * 0.05, dim=1)
        bursts = torch.sin(torch.linspace(0, 8 * math.pi, self.trace_len)).unsqueeze(0)
        traces = torch.tanh(cumulative + bursts)
        flicker = torch.sign(
            torch.randn(self.num_samples, self.trace_len, generator=self.generator)
        )
        mask = (
            torch.rand(self.num_samples, self.trace_len, generator=self.generator) > 0.9
        )
        traces = traces.where(~mask, flicker)
        return traces.float()

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> tuple[dict[Feats, torch.Tensor], torch.Tensor]:
        return {Feats.DIRS: self.data[idx]}, self.labels[idx]


@dataclass
class TrainingArtifacts:
    d_loss: float
    g_loss: float


def _build_dataloader(cfg: DictConfig, trace_len: int) -> DataLoader:
    batch_size: int = cfg.batch_size
    num_workers = 4

    dataset_name = cfg.dataset.name
    meta_df = load_dataset_meta_df(dataset_name)
    defence = NoDefence(network_delay_millis=(0, 0), network_pps=(0, 0))
    feature_names = [Feats.DIRS]
    dataset = WFDataset(
        dataset=f"{dataset_name}",
        meta_df=meta_df,
        defence=defence,
        feature_trs=FeatureTrs(feature_names=feature_names, n_packets=trace_len),
    )
    shuffle = True

    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
    )


def _train_epoch(
    *,
    dataloader: DataLoader,
    generator: TRGENResNet,
    discriminator: DF,
    device: torch.device,
    seed_dim: int,
    optim_d: torch.optim.Optimizer,
    optim_g: torch.optim.Optimizer,
    generator_mltp: int,
    recon_weight: float,
    feature_weight: float,
    max_batches: int | None = None,
) -> TrainingArtifacts:
    criterion = nn.CrossEntropyLoss()

    generator.train()
    discriminator.train()

    running_d = 0.0
    running_g = 0.0
    batches = 0

    with tqdm(dataloader, desc="train", ncols=TQDM_W) as pbar:
        for X, _ in pbar:
            X = dict_to_device(X, device)
            batch_size = next(iter(X.values())).shape[0]

            optim_d.zero_grad(set_to_none=True)
            labels_real = torch.full((batch_size,), 1, dtype=torch.long, device=device)
            preds_real = discriminator(X)
            loss_real = criterion(preds_real, labels_real)

            z = torch.randn(batch_size, seed_dim, device=device)
            fake_batch = generator(z)
            preds_fake = discriminator({k: v.detach() for k, v in fake_batch.items()})
            labels_fake = torch.zeros(batch_size, dtype=torch.long, device=device)
            loss_fake = criterion(preds_fake, labels_fake)

            d_loss = loss_real + loss_fake
            d_loss.backward()
            optim_d.step()

            real_dirs = X[Feats.DIRS]
            g_losses: list[torch.Tensor] = []
            for _ in range(generator_mltp):
                optim_g.zero_grad(set_to_none=True)
                z = torch.randn(batch_size, seed_dim, device=device)
                synth_batch = generator(z)
                preds_synth = discriminator(synth_batch)
                adv_loss = criterion(preds_synth, labels_real)
                total_g = adv_loss

                synth_dirs = synth_batch[Feats.DIRS]

                if recon_weight > 0.0:
                    recon_loss = F.l1_loss(synth_dirs, real_dirs, reduction="mean")
                    total_g = total_g + recon_weight * recon_loss

                if feature_weight > 0.0:
                    real_feat = discriminator.feature_extraction(real_dirs.unsqueeze(1))
                    synth_feat = discriminator.feature_extraction(
                        synth_dirs.unsqueeze(1)
                    )
                    feat_loss = F.l1_loss(
                        synth_feat.mean(dim=2), real_feat.mean(dim=2), reduction="mean"
                    )
                    total_g = total_g + feature_weight * feat_loss

                total_g.backward()
                g_losses.append(total_g.detach())
                optim_g.step()

            mean_g = torch.stack(g_losses).mean().item() if g_losses else 0.0

            batches += 1
            running_d = running_d + (d_loss.item() - running_d) / batches
            running_g = running_g + (mean_g - running_g) / batches

            pbar.set_postfix({"D": f"{running_d:.4f}", "G": f"{running_g:.4f}"})

            if max_batches is not None and batches >= max_batches:
                break

    return TrainingArtifacts(d_loss=running_d, g_loss=running_g)


@hydra.main(config_path=CONFIG_DIR_PATH, config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    trace_len = cfg.trace_len
    seed_dim = cfg.generator.get("seed_dim", 128)
    generator_kwargs = {
        "base_channels": cfg.generator.get("base_channels", 128),
        "min_initial_resolution": cfg.generator.get("min_initial_resolution", 64),
        "min_channels": cfg.generator.get("min_channels", 32),
        "attn_threshold": cfg.generator.get("attn_threshold", 512),
        "attn_heads": cfg.generator.get("attn_heads", 4),
        "use_spectral_norm": cfg.generator.get("use_spectral_norm", True),
    }

    generator = TRGENResNet(
        trace_len=trace_len,
        features=[Feats.DIRS],
        seed_dim=seed_dim,
        **generator_kwargs,
    )
    discriminator = DF(cfg.get("n_classes", 2), large_input=False)

    logger.info("Generator parameters: %s", count_parameters(generator))
    logger.info("Discriminator parameters: %s", count_parameters(discriminator))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator.to(device)
    discriminator.to(device)

    optim_d = torch.optim.Adam(discriminator.parameters(), lr=2e-4, betas=(0.5, 0.999))
    optim_g = torch.optim.Adam(generator.parameters(), lr=2e-4, betas=(0.5, 0.999))

    dataloader = _build_dataloader(cfg, trace_len)

    recon_weight = cfg.generator.get("recon_weight", 0.0)
    feature_weight = cfg.generator.get("feature_weight", 0.0)
    generator_mltp = cfg.get("generator_mltp", 1)
    epochs = cfg.get("epochs", 1)
    max_batches = cfg.get("max_batches", None)

    for epoch in range(epochs):
        logger.info("Epoch %d/%d", epoch + 1, epochs)
        metrics = _train_epoch(
            dataloader=dataloader,
            generator=generator,
            discriminator=discriminator,
            device=device,
            seed_dim=seed_dim,
            optim_d=optim_d,
            optim_g=optim_g,
            generator_mltp=generator_mltp,
            recon_weight=recon_weight,
            feature_weight=feature_weight,
            max_batches=max_batches,
        )
        logger.info(
            "epoch=%d d_loss=%.4f g_loss=%.4f",
            epoch + 1,
            metrics.d_loss,
            metrics.g_loss,
        )


if __name__ == "__main__":
    main()
