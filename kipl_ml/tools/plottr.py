import matplotlib.pyplot as plt
import numpy as np
import torch

from kipl_ml.trace.features import Feats


def plot_trace(
    trace_dict: dict[Feats, torch.tensor],
    idx: int | None = None,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    try:
        dirs = trace_dict[Feats.DIRS].detach().cpu().numpy()
    except KeyError as e:
        raise ValueError("To plot trace dict must contain 'dirs' feature.") from e

    if (idx is None) and (dirs.ndim >= 2):
        raise ValueError("Must provide idx if batch size > 1.")

    if idx is not None:
        dirs = dirs[idx]

    try:
        times = trace_dict[Feats.TIMES].detach().cpu().numpy()
        if idx is not None:
            times = times[idx]
    except KeyError:
        times = np.arange(len(dirs))

    ax = ax or plt.subplots(figsize=(12, 3))[1]

    ax.vlines(
        times, 0, dirs, colors=["red" if d > 0 else "blue" for d in dirs], alpha=0.5
    )

    return ax


def plot_bursts(
    trace_dict: dict[Feats, torch.Tensor],
    idx: int | None = None,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    try:
        burst_lens = trace_dict[Feats.BURST_LENS].detach().cpu().numpy()
    except KeyError as e:
        raise ValueError("To plot trace dict must contain 'burst_lens' feature.") from e

    try:
        burst_dirs = trace_dict[Feats.BURST_DIRS].detach().cpu().numpy()
    except KeyError as e:
        raise ValueError("To plot trace dict must contain 'burst_dirs' feature.") from e

    if idx is not None:
        burst_lens = burst_lens[idx]
        burst_dirs = burst_dirs[idx]

    burst_edges = burst_lens.cumsum()

    ax = ax or plt.subplots(figsize=(12, 3))[1]

    for ud in (-1, 1):
        burst_dirs_ = burst_dirs.copy()

        mask = burst_dirs == ud

        burst_dirs_[mask] = 0

        ax.fill_between(
            burst_edges,
            0,
            burst_dirs_,
            step="pre",
            color="blue" if ud < 0 else "red",
            alpha=0.5,
        )

    return ax
