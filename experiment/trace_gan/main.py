import os

import dotenv
import hydra
import mlflow
import torch
from omegaconf import DictConfig
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from kipl_ml.data.utils import load_dataset_meta_df
from kipl_ml.data.wf_dataset import WFDataset, dict_to_device
from kipl_ml.defences.base import NoDefence
from kipl_ml.logging.logger import TQDM_W, get_logger
from kipl_ml.models.df import DF
from kipl_ml.models.trgen import TRGEN1
from kipl_ml.models.utils import count_parameters
from kipl_ml.trace.features import Feats, FeatureTrs

logger = get_logger(__name__)
dotenv.load_dotenv()

WORKING_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_DIR_PATH = os.path.join(WORKING_DIR, "config")

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")

assert MLFLOW_TRACKING_URI is not None, "MLFLOW_TRACKING_URI must be set in .env file."
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)


@hydra.main(config_path=CONFIG_DIR_PATH, config_name="config", version_base=None)
def main(cfg: DictConfig):
    n_classes = 2
    trace_len = cfg.trace_len
    dataset = cfg.dataset.name
    meta_df = load_dataset_meta_df(dataset)

    defense = NoDefence(network_delay_millis=(0, 0), network_pps=(0, 0))

    feature_names = [Feats.DIRS]
    ds = WFDataset(
        dataset=f"{dataset}",
        meta_df=meta_df,
        defence=defense,
        feature_trs=FeatureTrs(feature_names=feature_names, n_packets=trace_len),
    )

    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, num_workers=4)

    discriminator = DF(n_classes, large_input=False)
    seed_dim = cfg.generator.seed_dim

    generator = TRGEN1(
        trace_len=trace_len,
        features=feature_names,
        seed_dim=seed_dim,
    )
    # generator

    loss = torch.nn.CrossEntropyLoss()
    optimD = torch.optim.Adam(discriminator.parameters(), lr=0.01)
    optimG = torch.optim.Adam(generator.parameters(), lr=0.01)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    discriminator.to(device)
    generator.to(device)

    seeds = torch.randn(cfg.batch_size, seed_dim, device=device)
    while True:
        for X, y in dl:
            X = dict_to_device(X, device)

            X_gen = generator(seeds)

            loss = ((X_gen["dirs"] - X["dirs"]) ** 2).sum()

            loss.backward()
            optimG.step()

            print(loss.mean().item())
            print(X_gen["dirs"][0])
            print(X["dirs"][0])
            print()
            break

    logger.info("Discriminator params: %s" % count_parameters(discriminator))
    logger.info("Generator params: %s" % count_parameters(generator))

    logger.info("Starting training... on device: %s" % device)
    epoch = 0
    while True:
        logger.info("Epoch: %04d" % epoch)
        dloss_mean = 0
        gloss_mean = 0
        i = 0
        with tqdm(dl, desc=f"epoch {epoch: 03d}", ncols=TQDM_W) as pbar:
            for X, y in pbar:
                # Discriminator
                optimD.zero_grad()
                labels = torch.ones(len(y), device=device, dtype=torch.long)
                preds = discriminator(dict_to_device(X, device))

                # Loss
                # L1 = log(D(x))
                # L0 = log(1 - D(G(z))) (discriminator)

                l1 = loss(preds, labels)

                seeds = torch.randn(cfg.batch_size, seed_dim, device=device)
                X_gen = generator(seeds)

                preds_gen = discriminator({k: x.detach() for k, x in X_gen.items()})
                labels_gen = torch.zeros(len(seeds), dtype=torch.long, device=device)

                l0 = loss(preds_gen, labels_gen)

                lt = l1 + l0
                lt.backward()
                optimD.step()

                dloss = lt.item()

                # Generator
                for _ in range(cfg.generator_mltp):
                    optimG.zero_grad()
                    seeds = torch.randn(cfg.batch_size, seed_dim, device=device)
                    X_gen = generator(seeds)
                    preds_gen = discriminator(X_gen)
                    labels = torch.ones(len(preds_gen), dtype=torch.long, device=device)

                    lg = loss(preds_gen, labels)

                    lg.backward()
                    optimG.step()

                gloss = lg.item()

                i += 1
                dloss_mean = dloss_mean + (dloss - dloss_mean) / i
                gloss_mean = gloss_mean + (gloss - gloss_mean) / i

                pbar.set_postfix(
                    {"Dloss": f"{dloss_mean:1.4f}", "Gloss": f"{gloss_mean:1.4f}"}
                )

            preds_gen = F.softmax(preds_gen, dim=1)
            print(X_gen)
            print(torch.unique(preds_gen.argmax(dim=1), return_counts=True))
        epoch += 1


if __name__ == "__main__":
    main()
