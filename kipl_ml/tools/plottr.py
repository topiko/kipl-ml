from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from kipl_ml.data.utils import DOWNLOAD, UPLOAD
from kipl_ml.logging.logger import get_logger
from kipl_ml.rl.enums import Actions, StepAction, StepActions
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
    move_right: bool = False,
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

    ax2.plot(x, max_p, "-|", lw=0.5, ms=3, color="gray", alpha=1, label=label)

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
    if move_right:
        ax2.spines["right"].set_position(("outward", 40))  # offset by 40 points

    mask = preds == true_class
    ax2.scatter(x[mask], max_p[mask], marker="D", color="green", s=10)

    ax2.legend(frameon=False, loc=4)


def _plot_boxes(
    x: np.ndarray,
    widths: np.ndarray,
    heights: np.ndarray,
    ax: plt.Axes,
    start_heights: np.ndarray | float = 0.0,
    **kwargs,
) -> None:
    if isinstance(start_heights, (float, int)):
        start_heights = np.full_like(x, start_heights)

    for xi, w, h, hs in zip(x, widths, heights, start_heights):
        if h == 0:
            continue
        if h < 0:
            if hs > 0:
                raise ValueError(
                    "Negative height with positive start height not supported"
                )
            y = h + hs
            h = -h
        else:
            y = hs

        rect = plt.Rectangle((xi, y), w, h, edgecolor=None, **kwargs)
        ax.add_patch(rect)


def _get_lims(*args, lim="max") -> float:
    if lim == "max":
        op = max
        v_ = float("-inf")
    elif lim == "min":
        op = min
        v_ = float("inf")
    else:
        raise KeyError("Only min and max allowed")

    for v in args:
        if isinstance(v, (float, int)):
            v_ = op(v_, v)
        elif isinstance(v, np.ndarray):
            if len(v) > 0:
                v_ = op(v_, op(v))
        else:
            raise TypeError(f"Invalid value type {type(v)}")

    return v_


def _format_val(v: int | float) -> str:
    if isinstance(v, (int, np.integer)):
        return f"{v}"
    if isinstance(v, (float, np.floating)):
        return f"{v:.02f}"

    raise NotImplementedError(f"Not implemented for type {type(v)}")


def plot_tam(
    trace_dict: dict[Feats, torch.Tensor],
    window_width: float,
    idx: int | None = None,
    ax: plt.Axes | None = None,
    cl_probs: torch.Tensor | None = None,
    true_class: int | None = None,
) -> plt.Axes:
    tam_d_c = _squeeze_batched(
        trace_dict[Feats.TAM_DOWN_COUNTS].detach().cpu().numpy(), idx
    )
    tam_u_c = _squeeze_batched(
        trace_dict[Feats.TAM_UP_COUNTS].detach().cpu().numpy(), idx
    )
    tam_times = _squeeze_batched(
        trace_dict[Feats.TAM_TIMES].detach().cpu().numpy(), idx
    )

    try:
        tam_u_pad = _squeeze_batched(
            trace_dict[Feats.TAM_UP_DECOY].detach().cpu().numpy(), idx
        )
        tam_d_pad = _squeeze_batched(
            trace_dict[Feats.TAM_DOWN_DECOY].detach().cpu().numpy(), idx
        )
    except KeyError:
        tam_u_pad = np.zeros_like(tam_u_c, dtype=int)
        tam_d_pad = np.zeros_like(tam_d_c, dtype=int)

    ax = ax or plt.subplots(figsize=(12, 3))[1]

    miny = (-1) * _get_lims(tam_u_c, tam_d_c, 1)
    maxy = _get_lims(tam_u_c, tam_d_c, 1)

    ax.set_ylim(miny, maxy)
    ax.set_title(f"TAM counts ww={window_width:.02f} s")
    ax.set_ylabel("TAM count")

    if tam_u_pad.sum() > 0:
        _plot_boxes(
            tam_times,
            np.ones_like(tam_times) * window_width,
            tam_u_pad,
            ax,
            start_heights=0.0,
            color=PAD_COLOR,
            alpha=0.5,
        )
    if tam_d_pad.sum() > 0:
        _plot_boxes(
            tam_times,
            np.ones_like(tam_times) * window_width,
            -tam_d_pad,
            ax,
            start_heights=0.0,
            color=PAD_COLOR,
            alpha=0.5,
        )

    _plot_boxes(
        tam_times,
        np.ones_like(tam_times) * window_width,
        tam_u_c - tam_u_pad,
        ax,
        start_heights=tam_u_pad,
        color=UP_COLOR,
        alpha=0.5,
    )
    _plot_boxes(
        tam_times,
        np.ones_like(tam_times) * window_width,
        -tam_d_c + tam_d_pad,
        ax,
        start_heights=-tam_d_pad,
        color=DOWN_COLOR,
        alpha=0.5,
    )

    info_d = {
        "nup": tam_u_c.sum(),
        "ndown": tam_d_c.sum(),
        "maxt": tam_times[(tam_u_c != 0) | (tam_d_c != 0)].max().round(2),
    }

    if tam_u_pad.sum() > 0:
        info_d["pad nup"] = tam_u_pad.sum()
        info_d["nup"] -= tam_u_pad.sum()
    if tam_d_pad.sum() > 0:
        info_d["pad ndown"] = tam_d_pad.sum()
        info_d["ndown"] -= tam_d_pad.sum()

    ax.text(
        0.98,
        0.99,
        "\n".join([f"{k} : {_format_val(v)}" for k, v in info_d.items()]),
        transform=ax.transAxes,
        va="top",
        ha="right",
    )

    if cl_probs is not None:
        _plot_probs(cl_probs, tam_times, ax, idx=idx, true_class=true_class)

    ax.set_xlim(0, info_d["maxt"] * 1.01)

    return ax


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
    times_ = times

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

    if Feats.DECOY in trace_dict:
        pad = _squeeze_batched(trace_dict[Feats.DECOY].bool(), idx)

        info_d["pad nup"] = (pad & (dirs == 1)).sum()
        info_d["pad ndown"] = (pad & (dirs == -1)).sum()

        info_d["nup"] -= info_d["pad nup"]
        info_d["ndown"] -= info_d["pad ndown"]

        colors[pad] = PAD_COLOR

    ax.vlines(
        times_,
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
        _plot_probs(cl_probs, times, ax, idx, true_class, move_right=False)

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
    actions: StepActions,
    ax: plt.Axes | None = None,
    dt_s: float | None = None,
) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(12, 3))

    if not actions:
        return ax

    action_times = np.asarray([float(a.time) for a in actions], dtype=np.float64)
    if dt_s is not None and dt_s > 0:
        action_times *= dt_s

    max_c = 1.0

    def _next_time(i: int) -> float:
        if i + 1 < len(action_times):
            return float(action_times[i + 1])
        return float(action_times[i] + (dt_s if dt_s is not None and dt_s > 0 else 1.0))

    def _plot_send(key: Actions, color: str, sign: int) -> None:
        nonlocal max_c
        xs: list[float] = []
        hs: list[float] = []
        for i, sa in enumerate(actions):
            if key not in sa:
                continue
            act = sa[key]
            if hasattr(act, "count") and int(act.count) != 0:
                xs.append(
                    float(action_times[i] + float(act.after_steps) * (dt_s or 1.0))
                )
                hs.append(float(act.count) if sign > 0 else -float(act.count))
        if not xs:
            return
        widths = np.ones(len(xs), dtype=np.float64) * (
            float(dt_s) if dt_s and dt_s > 0 else 0.005
        )
        _plot_boxes(
            np.asarray(xs), widths, np.asarray(hs), color=color, alpha=0.2, ax=ax
        )
        max_c = max(max_c, float(np.max(np.abs(hs))))

    _plot_send(Actions.SEND_UP, UP_COLOR, +1)
    _plot_send(Actions.SEND_DOWN, DOWN_COLOR, -1)

    def _plot_delay(key: Actions, color: str, ymin: float, ymax: float) -> None:
        for i, sa in enumerate(actions):
            if key not in sa:
                continue
            act = sa[key]
            if not hasattr(act, "steps") or int(act.steps) <= 0:
                continue
            d = float(act.steps) * (dt_s if dt_s is not None and dt_s > 0 else 1.0)
            ax.axvspan(
                float(action_times[i]),
                float(action_times[i] + d),
                ymin=ymin,
                ymax=ymax,
                color=color,
                alpha=0.18,
                lw=0,
                zorder=0,
            )

    # Delay spans: upward delays in upper half, downward delays in lower half.
    _plot_delay(Actions.DELAY_UP, "#d97706", 0.5, 1.0)
    _plot_delay(Actions.DELAY_DOWN, "#b45309", 0.0, 0.5)

    # Do-nothing spans on the remaining steps.
    for i, sa in enumerate(actions):
        if Actions.DO_NOTHING not in sa:
            continue
        if (
            int(sa[Actions.DO_NOTHING].bypass) != 0
        ):  # keep only the presence of the action
            pass
        t0 = float(action_times[i])
        t1 = _next_time(i)
        ax.axvspan(t0, t1, color="gray", alpha=0.12, lw=0, zorder=0)

    ax.vlines(action_times, -1, 1, color="black", lw=0.7)

    ax.set_ylim(-max_c * 1.1, max_c * 1.1)
    return ax


def plot_obs_features(
    fd: dict[Feats, torch.Tensor],
    idx: int | None = None,
    ax: plt.Axes | None = None,
    dt_s: float | None = None,
) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(12, 3))

    fd_ = {k: _squeeze_batched(v, idx) for k, v in fd.items()}

    times = fd_[Feats.TIME_BINS]
    up_count = fd_[Feats.UP_COUNT]
    down_count = fd_[Feats.DOWN_COUNT]
    # Use Dt (seconds) if available, otherwise convert Dt_BINS.
    if Feats.Dt in fd_:
        Dt = fd_[Feats.Dt]
    else:
        Dt = fd_[Feats.Dt_BINS]
        if dt_s is not None and dt_s > 0:
            Dt = Dt.astype(np.float64) * dt_s

    # fd is padded with -1 sentinel after seq end (int bins).
    # Filter those out so plotting and limits stay finite.
    mask = times >= 0
    times = times[mask]
    up_count = up_count[mask]
    down_count = down_count[mask]
    Dt = Dt[mask]

    # Convert int bins to seconds for plotting.
    if dt_s is not None and dt_s > 0:
        times = times.astype(np.float64) * dt_s

    _plot_boxes(times, Dt, up_count, color=UP_COLOR, alpha=0.5, ax=ax)
    _plot_boxes(times, Dt, -down_count, color=DOWN_COLOR, alpha=0.5, ax=ax)
    _plot_boxes(
        times,
        Dt,
        (up_count == 0) & (down_count == 0),
        color="gray",
        alpha=0.5,
        ax=ax,
    )

    if times.size > 0:
        nup = int(round(float(np.nansum(up_count))))
        ndown = int(round(float(np.nansum(down_count))))
        n_windows = int(times.size)
        maxt = float(np.nanmax(times))
    else:
        nup = 0
        ndown = 0
        n_windows = 0
        maxt = 0.0

    info_d = {
        "nup": nup,
        "ndown": ndown,
        "nwindows": n_windows,
        "maxt": maxt,
    }

    ax.text(
        0.98,
        0.99,
        "\n".join([f"{k} : {_format_val(v)}" for k, v in info_d.items()]),
        transform=ax.transAxes,
        va="top",
        ha="right",
    )

    if times.size == 0:
        ax.set_ylim(-1.0, 1.0)
    else:
        ymax = float(max(np.max(up_count), np.max(down_count), 1.0))
        ax.set_ylim(-ymax * 1.1, ymax * 1.1)

    return ax


def plot_rewards(
    times: torch.Tensor,
    rewards: dict[str, torch.Tensor],
    idx: int | None = None,
    ax: plt.Axes | None = None,
    dt_s: float | None = None,
    only_sum: bool = False,
    **plot_kwargs,
) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(12, 3))

    times = _squeeze_batched(times, idx)
    # Convert int bins to seconds.
    if dt_s is not None and dt_s > 0:
        times = times.astype(np.float64) * dt_s

    rewards["sum"] = sum(rewards.values())

    plot_kwargs.setdefault("lw", 0.7)
    plot_kwargs.setdefault("alpha", 1.0)
    plot_kwargs.setdefault("ls", "-")
    plot_kwargs.setdefault("marker", "|")

    for k, r in rewards.items():
        if only_sum and k != "sum":
            continue
        label = f"reward, {k}"
        if k == "sum":
            label = None
        r_ = _squeeze_batched(r, idx)
        ax.plot(times, r_, **plot_kwargs, label=label)

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
