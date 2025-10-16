import os

import dotenv
import hydra
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import torch
from omegaconf import DictConfig
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from kipl_ml.data import assets
from kipl_ml.data.utils import load_dataset_meta_df
from kipl_ml.data.wf_dataset import WFDataset, dict_to_device
from kipl_ml.defences.base import NoDefence
from kipl_ml.logging.logger import TQDM_W, get_logger
from kipl_ml.models.trgen import TRGEN3
from kipl_ml.tools.plottr import plot_trace
from kipl_ml.trace.features import Feats, FeatureTrs

logger = get_logger(__name__)
dotenv.load_dotenv()

WORKING_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_DIR_PATH = os.path.join(WORKING_DIR, "config")

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")

assert MLFLOW_TRACKING_URI is not None, "MLFLOW_TRACKING_URI must be set in .env file."
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)


def mimic_loss(pred_dirs: torch.Tensor, X: dict[Feats, torch.Tensor]) -> torch.Tensor:
    dir_loss_ = nn.functional.cross_entropy(
        pred_dirs[:, :-1].permute(0, 2, 1) + 1, X[Feats.DIRS][:, 1:].long() + 1
    )

    return dir_loss_


@hydra.main(config_path=CONFIG_DIR_PATH, config_name="config", version_base=None)
def main(cfg: DictConfig):
    trace_len = cfg.trace_len
    dataset = cfg.dataset.name
    meta_df = load_dataset_meta_df(dataset)

    y_ = 0
    mask = meta_df.loc[:, assets.PAGE_LABEL] == y_
    meta_df = meta_df.loc[mask].sample(5)

    defense = NoDefence(network_delay_millis=(0, 0), network_pps=(0, 0))

    feature_names = [Feats.DIRS, Feats.IATS_MAX_NORMALIZED]
    ds = WFDataset(
        meta_df=meta_df,
        defence=defense,
        feature_trs=FeatureTrs(feature_names=feature_names, n_packets=trace_len),
    )

    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, num_workers=4)

    generator = TRGEN3(features=feature_names, hsize=256, nlayer=2)

    optimG = torch.optim.Adam(generator.parameters(), lr=0.0001)

    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimG, factor=0.8, patience=3
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    generator.to(device)

    min_loss = np.inf
    patience = 5
    e = 0
    c = 0
    while True:
        with tqdm(dl, desc=f"epoch {e:02d}", ncols=TQDM_W) as pbar:
            loss_ = 0
            n = 1

            for X, y in pbar:
                X = dict_to_device(X, device)
                y = y.to(device)

                h = None
                optimG.zero_grad()

                dirs, h = generator(X, y, h)

                print(dirs.shape)
                print(X[Feats.DIRS].shape, X[Feats.DIRS])

                loss = mimic_loss(dirs, X)

                loss.backward()
                optimG.step()

                loss_ += (loss.item() - loss_) / n

                pbar.set_postfix({"l": loss_, "lr": lr_scheduler.get_last_lr()[0]})

                n += 1
            e += 1

            lr_scheduler.step(loss_)

        c += 1
        if loss_ < min_loss:
            logger.info("Improved loss! %.4f -> %.4f", min_loss, loss_)
            min_loss = loss_
            c = 0

        if c > patience:
            break

        rng = np.random.default_rng(seed=42)
        N = 3
        _, axarr = plt.subplots(2, N, figsize=(12, 9))
        for i, axcol in zip(rng.integers(0, len(meta_df), N), axarr.T):
            X, y = ds[i]
            X = {k: x_.unsqueeze(0) for k, x_ in dict_to_device(X, device).items()}
            y = y.to(device).reshape(1)

            plot_trace(X, idx=0, ax=axcol[0])
            axcol[0].set_title("Original")

            h = None

            dirs, h = generator(X, y, h)

            plot_trace({Feats.DIRS: dirs.argmax(-1) - 1}, idx=0, ax=axcol[1])
            axcol[1].set_title(f"Generated, loss {mimic_loss(dirs, X):.3f}")
            plt.suptitle(f"Epoch {e:03d}")
        plt.savefig(f"figs/trace_plot-{e:05d}.png")
        plt.close()


if __name__ == "__main__":
    main()
