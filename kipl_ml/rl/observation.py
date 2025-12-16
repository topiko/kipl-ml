from __future__ import annotations

import torch

from kipl_ml.data.utils import DOWNLOAD, UPLOAD
from kipl_ml.logging.logger import get_logger
from kipl_ml.rl.utils import _fill_w_last, _flush_left
from kipl_ml.trace.enums import Feats

logger = get_logger(__name__)


def _add_actions_to_silence_periods(
    feature_dict: dict[Feats, torch.Tensor], max_silence_s: float
) -> dict[Feats, torch.Tensor]:
    times = feature_dict[Feats.TIMES]
    dts = times.diff(
        dim=1, prepend=torch.zeros((times.shape[0], 1), device=times.device)
    )

    dt_ = 1e-3
    if dts.max() <= max_silence_s + dt_:
        return feature_dict

    lt = (dts // max_silence_s).sum(dim=1).max().ceil().int()
    add_action_times = torch.zeros((times.shape[0], lt), device=times.device)

    add_feature_dict: dict[Feats, torch.Tensor] = {
        Feats.UP_COUNT: add_action_times.clone(),
        Feats.DOWN_COUNT: add_action_times.clone(),
        Feats.TIMES: add_action_times,
    }
    for row in range(dts.shape[0]):
        mask = dts[row] > max_silence_s + dt_
        silence_starts = times[row, mask.roll(-1)]
        deltas = times[row, mask] - silence_starts

        counts = deltas // max_silence_s

        new_times_l = []
        for start, count in zip(silence_starts, counts):
            new_times = (
                torch.arange(1, count + 1, device=times.device) * max_silence_s + start
            )
            new_times_l.append(new_times)

        if not new_times_l:
            continue

        new_times = torch.cat(new_times_l)
        add_action_times[row, :] = times[row, -1]
        add_action_times[row, : new_times.shape[0]] = new_times

    feature_dict = {
        k: torch.cat([v, add_feature_dict[k]], dim=1) for k, v in feature_dict.items()
    }
    sort_idx = torch.argsort(feature_dict[Feats.TIMES], dim=1)
    feature_dict = {k: torch.gather(v, 1, sort_idx) for k, v in feature_dict.items()}

    return feature_dict


def get_window_feature_dict(
    X: dict[Feats, torch.Tensor], dt: float, max_silence_s: float, features: list[Feats]
) -> dict[Feats, torch.Tensor]:
    # (B, L)
    times = X[Feats.TIMES]

    bin_idx = (times // dt).long()

    feature_dict: dict[Feats, torch.Tensor] = {}

    up_counts = torch.zeros_like(times).scatter_add_(
        1, bin_idx, (X[Feats.DIRS] == UPLOAD).float()
    )
    down_counts = torch.zeros_like(times).scatter_add_(
        1, bin_idx, (X[Feats.DIRS] == DOWNLOAD).float()
    )
    times_ = torch.zeros_like(times).scatter_(1, bin_idx, bin_idx.float() * dt)

    mask = (up_counts != 0) | (down_counts != 0)
    max_l = mask.sum(dim=1).max()
    feature_dict[Feats.UP_COUNT] = _flush_left(up_counts, mask)[:, :max_l]
    feature_dict[Feats.DOWN_COUNT] = _flush_left(down_counts, mask)[:, :max_l]

    times_ = _flush_left(times_, mask)[:, :max_l]
    times_ = _fill_w_last(times_, pad_val=0)
    feature_dict[Feats.TIMES] = times_

    feature_dict = _add_actions_to_silence_periods(feature_dict, max_silence_s)

    dts = feature_dict[Feats.TIMES].diff(
        dim=1, prepend=torch.zeros((times.shape[0], 1), device=times.device)
    )

    feature_dict[Feats.Dt] = dts
    mask = dts != 0
    mask[:, 0] = True  # Keep the first time point.
    max_l = mask.sum(dim=1).max()
    # dict[Feats, Tensor (B, max_l)]
    feature_dict = {k: _flush_left(v, mask)[:, :max_l] for k, v in feature_dict.items()}

    if (feature_dict[Feats.Dt].diff(dim=1).max()) > max_silence_s:
        breakpoint()
        logger.warning("max_silence_s is not implemented yet in get_feature_dict!")

    if not all(f in feature_dict for f in features):
        raise ValueError("Some requested features are missing!")

    return feature_dict
