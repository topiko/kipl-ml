import os
from functools import partial

import dotenv
import hydra
import mlflow
import numpy as np
import torch
from omegaconf import DictConfig
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from kipl_ml.data.utils import load_dataset_meta_df
from kipl_ml.data.wf_dataset import WFDataset, dict_to_device
from kipl_ml.defences.base import NoDefence
from kipl_ml.logging.logger import TQDM_W, get_logger
from kipl_ml.models.trgen import TRGEN2
from kipl_ml.trace.features import Feats, FeatureTrs

logger = get_logger(__name__)
dotenv.load_dotenv()

WORKING_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_DIR_PATH = os.path.join(WORKING_DIR, "config")

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")

assert MLFLOW_TRACKING_URI is not None, "MLFLOW_TRACKING_URI must be set in .env file."
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)


def collate_fn(
    batch: list[tuple[dict[Feats, torch.Tensor], torch.Tensor]], seq_len: int = 100
) -> tuple[dict[Feats, torch.Tensor], dict[Feats, torch.Tensor], torch.Tensor]:
    bs = len(batch)
    features = batch[0][0].keys()
    X = {f: torch.ones((bs, seq_len), dtype=torch.float) for f in features}
    Xnext = {f: torch.ones((bs,), dtype=torch.float) for f in features}
    y = torch.zeros((bs,), dtype=torch.long)
    for i, (x_, y_) in enumerate(batch):
        start_idx = 0
        for f in features:
            xtmp = x_[f][start_idx : start_idx + seq_len + 1]
            if f == Feats.BURST_DIRS:
                xtmp += 1

            if len(xtmp) < seq_len + 1:
                xnext = 1.0
                lenx = len(xtmp)
            else:
                xnext = xtmp[-1]
                lenx = len(xtmp) - 1

            X[f][i, :lenx] = xtmp[:lenx]
            Xnext[f][i] = xnext
        y[i] = y_

    return X, Xnext, y


@hydra.main(config_path=CONFIG_DIR_PATH, config_name="config", version_base=None)
def main(cfg: DictConfig):
    trace_len = cfg.trace_len
    dataset = cfg.dataset.name
    meta_df = load_dataset_meta_df(dataset)

    # y_ = 0
    # mask = meta_df.loc[:, assets.PAGE_LABEL] == y_
    # meta_df = meta_df.loc[mask]
    defense = NoDefence(network_delay_millis=(0, 0), network_pps=(0, 0))

    feature_names = [Feats.BURST_LENS, Feats.BURST_DIRS]
    ds = WFDataset(
        dataset=f"{dataset}",
        meta_df=meta_df,
        defence=defense,
        feature_trs=FeatureTrs(feature_names=feature_names, n_packets=trace_len),
    )

    ds.report()

    dl = DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=12,
        collate_fn=partial(collate_fn, seq_len=cfg.train_seq_len),
    )

    generator = TRGEN2(features=feature_names, in_channels=2, hsize=256, nlayer=1)

    optimG = torch.optim.Adam(generator.parameters(), lr=0.01)

    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimG, factor=0.8, patience=3
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    generator.to(device)

    dir_loss = nn.CrossEntropyLoss()
    len_loss = nn.MSELoss()

    e = 0
    min_loss = np.inf
    c = 0
    patience = 10
    while True:
        with tqdm(dl, desc="Training", ncols=TQDM_W) as pbar:
            loss_ = 0
            n = 1
            for X, Xnext, _ in pbar:
                X = dict_to_device(X, device)
                Xnext = dict_to_device(Xnext, device)

                h = None
                optimG.zero_grad()
                (dirs, lens), h = generator(X, h)

                dir_loss_ = dir_loss(
                    dirs[:, :-1].permute(0, 2, 1), X[Feats.BURST_DIRS][:, 1:].long()
                )
                blens = X[Feats.BURST_LENS][:, 1:]

                len_loss_ = len_loss(lens[:, :-1], blens)

                loss = dir_loss_ + len_loss_

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
    N = 200
    for i in rng.integers(0, 15000, 10):
        X, _ = ds[i]
        (dirs, lens), _ = generator(dict_to_device(X, device), None)
        dirs_true = X[Feats.BURST_DIRS].to(device)
        lens_true = X[Feats.BURST_LENS].to(device)
        N = min(len(dirs_true) - 1, N)
        print("Dirs true / pred")
        print(dirs_true[1 : N + 1] - (dirs[:N].argmax(-1) - 1))
        print("Lens true / pred")
        lt = lens_true[1 : N + 1]
        lp = lens[:N].round()

        print(lt)
        print(lp)
        print()


if __name__ == "__main__":
    main()
