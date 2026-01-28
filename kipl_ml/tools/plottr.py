from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from kipl_ml.data.utils import DOWNLOAD, UPLOAD
from kipl_ml.logging.logger import get_logger
from kipl_ml.rl.enums import Actions
from kipl_ml.trace.features import Feats

logger = get_logger(__name__)

style_path = (
    Path(__file__).resolve().parent.parent.parent / ".config" / "plotstyle.mplstyle"
)

plt.style.use(style_path)

UP_COLOR = "blue"
DOWN_COLOR = "red"
PAD_COLOR = "black"


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


def _plot_probs(
    cl_probs: torch.Tensor,
    xs: np.ndarray,
    ax: plt.Axes,
    idx: int | None = None,
    true_class: int | None = None,
) -> None:
    cl_probs = _squeeze_batched(cl_probs, idx)

    ax2 = ax.twinx()
    if true_class is not None:
        max_p = cl_probs[:, true_class]
        label = f"Class {true_class} prob."
    else:
        max_p = cl_probs.max(axis=1).numpy()
        label = "Max class prob."

    x = xs

    ax2.plot(x, max_p, "-", lw=0.5, color="gray", alpha=1, label=label)

    preds = cl_probs.argmax(axis=1)
    if true_class is None:
        ax3 = ax.twinx()
        ax3.spines["right"].set_position(("outward", 40))  # offset by 40 points
        ax3.plot(x, preds, color="black", lw=1)

        ax3.hlines(
            true_class,
            ls="--",
            color="black",
            xmin=x.min(),
            xmax=x.max(),
            alpha=1.0,
            lw=0.5,
        )
        ax3.spines["right"].set_visible(True)

    ax2.spines["right"].set_visible(True)

    mask = preds == true_class
    ax2.scatter(x[mask], max_p[mask], marker="D", color="green", s=10)

    ax2.legend(frameon=False, loc=4)


def _plot_boxes(
    x: np.ndarray, widths: np.ndarray, heights: np.ndarray, ax: plt.Axes, **kwargs
) -> None:
    for xi, w, h in zip(x, widths, heights):
        if h == 0:
            continue
        if h < 0:
            y = h
            h = -h
        else:
            y = 0

        rect = plt.Rectangle((xi, y), w, h, **kwargs)
        ax.add_patch(rect)


def plot_trace(
    trace_dict: dict[Feats, torch.tensor],
    idx: int | None = None,
    ax: plt.Axes | None = None,
    cl_probs: torch.Tensor | None = None,
    true_class: int | None = None,
) -> plt.Axes:
    try:
        dirs = trace_dict[Feats.DIRS].detach().cpu().numpy()
    except KeyError:
        dirs = trace_dict[Feats.DIR_PROBS].argmax(dim=-1).detach().cpu().numpy() - 1

    dirs = _squeeze_batched(dirs, idx)

    try:
        times = trace_dict[Feats.TIMES].detach().cpu().numpy()
    except KeyError:
        times = np.arange(len(dirs))

    times = _squeeze_batched(times, idx)

    ax = ax or plt.subplots(figsize=(12, 3))[1]

    info_d = {
        "nup": (dirs == 1).sum(),
        "ndown": (dirs == -1).sum(),
        "maxt": f"{times.max():.02f}",
    }

    colors = np.empty_like(dirs, dtype=object)
    colors[:] = "cyan"
    colors[dirs == UPLOAD] = UP_COLOR
    colors[dirs == DOWNLOAD] = DOWN_COLOR
    if Feats.PADDING in trace_dict:
        pad = _squeeze_batched(trace_dict[Feats.PADDING].bool(), idx)
        colors[pad] = PAD_COLOR

        info_d["pad nup"] = (pad & (dirs == 1)).sum()
        info_d["pad ndown"] = (pad & (dirs == -1)).sum()

        info_d["nup"] -= info_d["pad nup"]
        info_d["ndown"] -= info_d["pad ndown"]

        dirs[pad] *= 0.7

    ax.vlines(
        times,
        0,
        dirs,
        colors=colors,
        alpha=1,
        lw=0.5,
    )

    ax.text(
        0.98,
        0.99,
        "\n".join([f"{k} : {v}" for k, v in info_d.items()]),
        transform=ax.transAxes,
        va="top",
        ha="right",
    )

    if cl_probs is not None:
        _plot_probs(cl_probs, times, ax, idx, true_class)

    return ax


def plot_packet_buffer(
    buffer: dict[Feats, torch.Tensor],
    idx: int | None = None,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(12, 3))

    buffer_up = _squeeze_batched(buffer[Feats.UP_BUFFER], idx)
    buffer_down = _squeeze_batched(buffer[Feats.DOWN_BUFFER], idx)
    times = _squeeze_batched(buffer[Feats.TIMES], idx)

    ax.step(times, buffer_up, c="blue", where="post", lw=1, label="Up Buffer")
    ax.step(times, buffer_down, c="red", where="post", lw=1, label="Down Buffer")

    return ax


def plot_actions(
    times: torch.Tensor,
    actions: dict[Actions, torch.Tensor],
    idx: int | None = None,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(12, 3))

    times = _squeeze_batched(times, idx)
    actions = {k: _squeeze_batched(v, idx) for k, v in actions.items()}

    max_c = 0
    for ackt in (
        Actions.WAIT,
        (Actions.SEND_COUNT_UP, Actions.SEND_TIME_UP),
        (Actions.SEND_COUNT_DOWN, Actions.SEND_TIME_DOWN),
    ):
        if isinstance(ackt, tuple):
            counts = actions[ackt[0]]
            counts[counts == 0] = 0.05
            durs = actions[ackt[1]]

            match ackt[0]:
                case Actions.SEND_COUNT_UP:
                    color = UP_COLOR
                case Actions.SEND_COUNT_DOWN:
                    color = DOWN_COLOR
                    counts = -counts
                case _:
                    raise ValueError(f"Unknown action type: {ackt[0]}")

            max_c = max(max_c, np.absolute(counts).max())

            _plot_boxes(
                x=times, widths=durs, heights=counts, color=color, alpha=0.2, ax=ax
            )
        elif ackt == Actions.WAIT:
            counts = np.zeros_like(times)
            durs = np.diff(times, append=np.array([times[-1]]), axis=0)
            counts[actions[ackt] == 1] = 1

            _plot_boxes(times, durs, counts, color="gray", alpha=0.2, ax=ax)

        ax.vlines(times, -0.1, 0.1, color="black", lw=0.5)

    ax.set_ylim(-max_c * 1.1, max_c * 1.1)
    return ax


def plot_obs_features(
    fd: dict[Feats, torch.Tensor], idx: int | None = None, ax: plt.Axes | None = None
) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(12, 3))

    fd_ = {k: _squeeze_batched(v, idx) for k, v in fd.items()}

    times = fd_[Feats.TIMES]
    up_count = fd_[Feats.UP_COUNT]
    down_count = fd_[Feats.DOWN_COUNT]
    Dt = fd_[Feats.Dt]

    _plot_boxes(times[:-1], Dt[1:], up_count[:-1], color=UP_COLOR, alpha=0.5, ax=ax)
    _plot_boxes(
        times[:-1], Dt[1:], -down_count[:-1], color=DOWN_COLOR, alpha=0.5, ax=ax
    )
    _plot_boxes(
        times[:-1],
        Dt[1:],
        (up_count[:-1] == 0) & (down_count[:-1] == 0),
        color="gray",
        alpha=0.5,
        ax=ax,
    )

    ax.set_ylim(-down_count.max() * 1.1, up_count.max() * 1.1)

    return ax


def plot_rewards(
    times: torch.Tensor,
    rewards: dict[str, torch.Tensor],
    idx: int | None = None,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(12, 3))

    times = _squeeze_batched(times, idx)
    rewards["sum"] = sum(rewards.values())

    for k, r in rewards.items():
        r_ = _squeeze_batched(r, idx)
        ax.plot(times, r_, lw=1, label=f"rewards, {k}")

    return ax


def plot_bursts(
    trace_dict: dict[Feats, torch.Tensor],
    idx: int | None = None,
    ax: plt.Axes | None = None,
    cl_probs: torch.Tensor | None = None,
    true_class: int | None = None,
) -> plt.Axes:
    try:
        burst_lens = trace_dict[Feats.BURST_LENS].detach().cpu().numpy()
    except KeyError as e:
        raise ValueError("To plot trace dict must contain 'burst_lens' feature.") from e

    burst_lens = _squeeze_batched(burst_lens, idx)

    burst_edges = np.concat((np.zeros(1), burst_lens.cumsum()))
    burst_dirs = np.ones_like(burst_edges)
    burst_dirs[0::2] = -1

    ax = ax or plt.subplots(figsize=(12, 3))[1]

    for ud in (-1, 1):
        burst_dirs_ = burst_dirs.copy()

        mask = burst_dirs == ud

        burst_dirs_[mask] = 0

        ax.fill_between(
            burst_edges,
            0,
            burst_dirs_,
            step="post",
            color="blue" if ud < 0 else "red",
            alpha=0.5,
            lw=0,
        )

    if cl_probs is not None:
        _plot_probs(cl_probs, burst_edges[1:], ax, idx, true_class)

    return ax
