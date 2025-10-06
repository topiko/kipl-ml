import os

import dotenv
import hydra
import mlflow
import torch
from omegaconf import DictConfig
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


@hydra.main(config_path=CONFIG_DIR_PATH, config_name="config", version_base=None)
def main(cfg: DictConfig):
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

    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=False, num_workers=4)

    generator = TRGEN2(features=feature_names, inlen=1)

    optimG = torch.optim.Adam(generator.parameters(), lr=0.01)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    generator.to(device)

    while True:
        with tqdm(dl, desc="Training", ncols=TQDM_W) as pbar:
            for X, _ in pbar:
                X = dict_to_device(X, device)[Feats.DIRS]

                h = None
                for t in range(trace_len - 1):
                    optimG.zero_grad()
                    dirs, h = generator(X[:, t : t + 1], h)
                    loss = ((dirs - X[:, t + 1]) ** 2).mean()

                    if t % 100 == 0:
                        loss.backward()
                        optimG.step()
                        h = tuple(s.detach() for s in h)
                        pbar.set_postfix({"loss": loss.item()})
        break


if __name__ == "__main__":
    main()
