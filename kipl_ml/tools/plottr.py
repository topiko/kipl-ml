from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from kipl_ml.logging.logger import get_logger
from kipl_ml.trace.features import Feats

logger = get_logger(__name__)

style_path = (
    Path(__file__).resolve().parent.parent.parent / ".config" / "plotstyle.mplstyle"
)

plt.style.use(style_path)


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


def _squeeze_batched(arr: np.ndarray | torch.Tensor, idx: int | None) -> np.ndarray:
    if isinstance(arr, torch.Tensor):
        arr = arr.cpu().detach().numpy()

    if arr.ndim == 1:
        return arr
    if arr.ndim >= 2:
        if arr.shape[0] == 1:
            return arr.squeeze(0)
        if idx is None:
            raise ValueError("Batched input but no index provided")

        return arr[idx]


def plot_bursts(
    trace_dict: dict[Feats, torch.Tensor],
    idx: int | None = None,
    ax: plt.Axes | None = None,
    cl_probs: torch.Tensor | None = None,
) -> plt.Axes:
    try:
        burst_lens = trace_dict[Feats.BURST_LENS].detach().cpu().numpy()
    except KeyError as e:
        raise ValueError("To plot trace dict must contain 'burst_lens' feature.") from e

    burst_lens = _squeeze_batched(burst_lens, idx)

    burst_dirs = np.ones_like(burst_lens)
    burst_dirs[1::2] = -1

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

    if cl_probs is not None:
        cl_probs = _squeeze_batched(cl_probs, idx)

        ax2 = ax.twinx()
        max_p = cl_probs.max(axis=1)
        ax2.plot(burst_edges, max_p)

        ax3 = ax.twinx()
        ax3.spines["right"].set_position(("outward", 60))  # offset by 60 points

        ax3.plot(burst_edges, cl_probs.argmax(axis=1))

        ax2.spines["right"].set_visible(True)
        ax3.spines["right"].set_visible(True)

    return ax
