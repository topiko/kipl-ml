import os

import dotenv
import hydra
import matplotlib.pyplot as plt
import mlflow
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from tqdm import tqdm

from kipl_ml.data import assets
from kipl_ml.data.utils import load_dataset_meta_df
from kipl_ml.data.wf_dataset import WFDataset, dict_to_device
from kipl_ml.defences.base import NoDefence
from kipl_ml.logging.logger import TQDM_W, get_logger
from kipl_ml.models.trgen import TRGEN2
from kipl_ml.tools.plottr import plot_trace
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
    trace_len = cfg.trace_len
    dataset = cfg.dataset.name
    meta_df = load_dataset_meta_df(dataset)

    y_ = 0
    mask = meta_df.loc[:, assets.PAGE_LABEL] == y_
    meta_df = meta_df.loc[mask].sample(5)

    defense = NoDefence(network_delay_millis=(0, 0), network_pps=(0, 0))

    feature_names = [Feats.DIRS]
    ds = WFDataset(
        dataset=f"{dataset}",
        meta_df=meta_df,
        defence=defense,
        feature_trs=FeatureTrs(feature_names=feature_names, n_packets=trace_len),
    )

    dl = DataLoader(ds, batch_size=16, shuffle=True, num_workers=4)

    generator = TRGEN2(features=feature_names, inlen=1, hsize=1024, nlayer=4)

    optimG = torch.optim.Adam(generator.parameters(), lr=0.0001)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    generator.to(device)

    trlen = 500
    X_, _ = ds[0]
    X_ = X_[Feats.DIRS].to(device).unsqueeze(0)[:, :trlen]

    e = 0
    while True:
        with tqdm(dl, desc="Training", ncols=TQDM_W) as pbar:
            loss_ = 0
            n = 1
            for X, _ in pbar:
                X = dict_to_device(X, device)[Feats.DIRS]

                h = None
                for t in range(trace_len - 1):
                    optimG.zero_grad()
                    dirs, h = generator(X[:, t].unsqueeze(-1), h)

                    mask = X[:, t] != 0

                    loss = ((dirs[mask] - X[mask, t + 1]) ** 2).mean()
                    # loss = ((torch.sign(dirs) - X[:, t + 1]) ** 2).mean()

                    loss_ += (loss.item() - loss_) / n

                    if t % 1000 == 0:
                        loss.backward()
                        optimG.step()
                        h = tuple(s.detach() for s in h)
                        pbar.set_postfix({"loss": loss_})

                    if t == 0:
                        _, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 3), sharex=True)

                        X_gen = torch.zeros_like(X_)
                        h_ = None
                        for t_ in range(trlen - 1):
                            if t_ < trlen - 100:
                                x_ = X_[:, t_ : t_ + 1]
                            else:
                                x_ = X_gen[:, t_ : t_ + 1]
                            dirs_, h_ = generator(x_, h_)
                            X_gen[0, t_ + 1] = dirs_
                        plot_trace({Feats.DIRS: X_gen}, idx=0, ax=ax1)
                        plot_trace({Feats.DIRS: X_}, idx=0, ax=ax2)
                        plt.savefig(f"figs/rnn/rnn_step_{e:04d}.png")
                        e += 1
                        plt.close()
                        # plt.show()

                    n += 1


if __name__ == "__main__":
    main()
