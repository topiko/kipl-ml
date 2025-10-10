import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn

from kipl_ml.data import assets
from kipl_ml.data.wf_dataset import WFDataset, dict_to_device
from kipl_ml.defences.base import _Def
from kipl_ml.tools.plottr import plot_bursts
from kipl_ml.trace.features import Feats, FeatureTrs

mpl.rcParams["axes.spines.right"] = False
mpl.rcParams["axes.spines.top"] = False


def simple_burst_plot(
    meta_df: pd.DataFrame,
    nsamples: int,
    generator: nn.Module,
    feature_names: list[Feats],
    defense: _Def | None = None,
    lbl: int = 0,
    device: torch.device = torch.device("cpu"),
) -> plt.Figure:
    nsamples = 3

    mask = meta_df.loc[:, assets.PAGE_LABEL] == lbl
    wf_ = WFDataset(
        meta_df=meta_df.loc[mask],
        defence=defense,
        feature_trs=FeatureTrs(feature_names=feature_names, n_packets=None),
    )

    fig, axarr = plt.subplots(
        nsamples,
        2,
        figsize=(12, nsamples * 2),
        sharex=True,
        sharey=True,
    )
    rng = np.random.default_rng(seed=42)

    for axrow in axarr:
        idx = rng.integers(0, len(wf_))
        X, _ = wf_[idx]
        ax = plot_bursts(X, ax=axrow[0])
        (dirs, lens), _ = generator(dict_to_device(X, device), None)

        dirs = dirs.argmax(dim=-1) - 1
        lens = lens.round()

        X = {Feats.BURST_DIRS: dirs.squeeze(), Feats.BURST_LENS: lens.squeeze()}
        ax = plot_bursts(X, ax=axrow[1])

    axarr[0, 0].set_title("True")
    axarr[0, 1].set_title("Generated")
    plt.tight_layout()
    return fig
