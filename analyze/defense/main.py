import hashlib
import json
import multiprocessing
from enum import StrEnum
from pathlib import Path

import dotenv
import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from experiment.utils import defence_builder
from kipl_ml.data import assets
from kipl_ml.data.wf_dataset import get_train_valid_test
from kipl_ml.logging.logger import TQDM_W, get_logger
from kipl_ml.trace.enums import Feats
from kipl_ml.trace.features import FeatureTrs

logger = get_logger(__name__)
dotenv.load_dotenv()


class Keys(StrEnum):
    BW_OVERHEAD = "bw_overhead"
    LOG10_BW_OVERHEAD = "log10_bw_overhead"
    N_NORMAL = "n_normal_pkts"
    N_DECOY = "n_decoy_pkts"
    DUR_BINS = "dur_bins"
    PKT_BINS = "pkt_bins"
    DURATION = "duration"
    LABEL = "label"
    TIME_NS = "time [ns]"


def _get_cache_path(cfg: DictConfig, maxt: float) -> Path:
    payload = {
        "cache_version": 1,
        "seed": cfg.seed,
        "dataset": OmegaConf.to_container(cfg.dataset, resolve=True),
        "defence": OmegaConf.to_container(cfg.defence, resolve=True),
        "network": OmegaConf.to_container(cfg.network, resolve=True),
        "n_min_packets": None,
        "maxt": maxt,
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode()
    ).hexdigest()
    repo_root = Path(hydra.utils.get_original_cwd())
    return repo_root / ".cache" / f"{digest}.pkl"


def _build_df(cfg: OmegaConf, maxt: float) -> pd.DataFrame:

    test_xv = 0
    _, _, ds = get_train_valid_test(
        dataset=cfg.dataset.name,
        label=assets.PAGE_LABEL,
        n_splits=cfg.dataset.n_splits,
        test_xv=test_xv,
        random_state=cfg.seed,
        feature_trs=FeatureTrs(
            feature_names=[Feats.DIRS, Feats.TIMES, Feats.DECOY],
            n_packets=100_000,
            time_clamp=None,
        ),
        defence_aug_valid=0,
        n_min_packets=None,
        trim_raw=0,
        **defence_builder.get_defence(cfg),
    )

    meta_df = ds.meta_df
    mask = meta_df.loc[:, Keys.TIME_NS] / 1e9 < maxt
    ds.meta_df = meta_df[mask]

    nworkers = multiprocessing.cpu_count() // 8 * 7
    bs = 64
    dl = DataLoader(
        ds,
        batch_size=bs,
        shuffle=False,
        num_workers=nworkers,
    )

    durs = []
    n_normal = []
    n_decoy = []
    labels = []
    with tqdm(dl, ncols=2 * TQDM_W) as pbar:
        for X, y in pbar:
            n_packets_ = (X[Feats.DIRS] != 0).sum(dim=1).numpy()
            n_decoy_ = X[Feats.DECOY].sum(dim=1).numpy()

            durs.append(X[Feats.TIMES].max(dim=1).values.numpy())
            n_normal.append(n_packets_ - n_decoy_)
            n_decoy.append(n_decoy_)
            labels.append(y.numpy())

    df = pd.DataFrame()
    for k, vals_l in (
        (Keys.DURATION, durs),
        (Keys.N_NORMAL, n_normal),
        (Keys.N_DECOY, n_decoy),
        (Keys.LABEL, labels),
    ):
        df.loc[:, k] = np.concatenate(vals_l)

    return df


@hydra.main(config_path="config/", config_name="config", version_base=None)
def main(cfg: DictConfig):

    MAXT = 100

    cache_path = _get_cache_path(cfg, MAXT)
    if cache_path.exists():
        logger.info("Loading cached dataframe from %s", cache_path)
        df = pd.read_pickle(cache_path)
    else:
        df = _build_df(cfg, MAXT)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_pickle(cache_path)
        logger.info("Cached dataframe to %s", cache_path)

    nbins = 200
    MAXPKTS = 50_000
    time_bins = np.arange(nbins + 1) * MAXT / nbins
    pkt_bins = np.arange(nbins + 1) * MAXPKTS / nbins
    df.loc[:, Keys.DUR_BINS] = pd.cut(df.loc[:, Keys.DURATION], time_bins)
    df.loc[:, Keys.PKT_BINS] = pd.cut(df.loc[:, Keys.N_NORMAL], pkt_bins)

    df.loc[:, Keys.BW_OVERHEAD] = df.loc[:, Keys.N_DECOY] / df.loc[:, Keys.N_NORMAL]
    df.loc[:, Keys.LOG10_BW_OVERHEAD] = np.log10(df.loc[:, Keys.BW_OVERHEAD] + 1e-6)

    metric = Keys.BW_OVERHEAD

    df_plot = (
        df.groupby(by=[Keys.DUR_BINS, Keys.PKT_BINS])
        .agg(
            {
                Keys.N_DECOY: "mean",
                Keys.BW_OVERHEAD: "mean",
                Keys.LOG10_BW_OVERHEAD: "mean",
            },
            observed=True,
        )
        .dropna()
    )

    df_pivot = (
        df_plot.reset_index(drop=False)
        .pivot(index=Keys.DUR_BINS, columns=Keys.PKT_BINS, values=metric)
        .iloc[::-1]
    )

    _, ax = plt.subplots(figsize=(12, 14))
    sns.heatmap(
        df_pivot,
        annot=False,
        cbar_kws={"label": metric},
        ax=ax,
        annot_kws={"fontsize": 4},
        vmin=0 if metric == Keys.BW_OVERHEAD else -1,
        vmax=100 if metric == Keys.BW_OVERHEAD else 3,
    )
    plt.suptitle(cfg.defence.type)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
