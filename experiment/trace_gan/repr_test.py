import os

import dotenv
import hydra
import mlflow
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
    X = {f: torch.zeros((bs, seq_len), dtype=torch.float) for f in features}
    Xnext = {f: torch.zeros((bs,), dtype=torch.float) for f in features}
    y = torch.zeros((bs,), dtype=torch.long)
    for i, (x_, y_) in enumerate(batch):
        start_idx = 0
        for f in features:
            xtmp = x_[f][start_idx : start_idx + seq_len + 1]
            if f == Feats.BURST_DIRS:
                xtmp += 1

            if len(xtmp) < seq_len + 1:
                xnext = 0.0
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
        num_workers=0,
        collate_fn=collate_fn,
    )

    generator = TRGEN2(features=feature_names, in_channels=2, hsize=256, nlayer=2)

    optimG = torch.optim.Adam(generator.parameters(), lr=0.01)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    generator.to(device)

    dir_loss = nn.CrossEntropyLoss()
    len_loss = nn.MSELoss()

    e = 0
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

                dir_loss_ = dir_loss(dirs[:, -1], Xnext[Feats.BURST_DIRS].long())
                len_loss_ = len_loss(lens[:, -1], Xnext[Feats.BURST_LENS])

                loss = dir_loss_ + len_loss_

                loss.backward()
                optimG.step()

                loss_ += (loss.item() - loss_) / n

                pbar.set_postfix(
                    {
                        "l": loss_,
                        "dl": dir_loss_.item(),
                        "ll": len_loss_.item(),
                    }
                )

                n += 1
            e += 1


if __name__ == "__main__":
    main()
