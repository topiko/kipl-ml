from __future__ import annotations

import torch

from kipl_ml.data.utils import DOWNLOAD, UPLOAD
from kipl_ml.logging.logger import get_logger
from kipl_ml.rl.utils import _fill_after_seq_end, _flush_left
from kipl_ml.trace.enums import Feats

logger = get_logger(__name__)


def _add_actions_to_silence_periods(
    feature_dict: dict[Feats, torch.Tensor], max_silence_s: float
) -> dict[Feats, torch.Tensor]:
    times = feature_dict[Feats.TIMES]
    dts = times.diff(
        dim=1, prepend=torch.zeros((times.shape[0], 1), device=times.device)
    )

    if dts[dts.isfinite()].max() <= max_silence_s:
        return feature_dict

    lt = (
        torch.where(dts.isfinite(), dts // max_silence_s, 0)
        .sum(dim=1)
        .max()
        .ceil()
        .int()
    )

    add_action_times = torch.zeros((times.shape[0], lt), device=times.device)

    add_feature_dict: dict[Feats, torch.Tensor] = {
        Feats.UP_COUNT: add_action_times.clone(),
        Feats.DOWN_COUNT: add_action_times.clone(),
        Feats.TIMES: add_action_times,
    }
    dt_ = 1e-3
    for row in range(dts.shape[0]):
        mask = dts[row] > (max_silence_s + dt_)
        silence_starts = times[row, mask.roll(-1)]
        deltas = times[row, mask] - silence_starts

        counts = deltas // (max_silence_s + dt_)

        new_times_l = []
        for start, count in zip(silence_starts, counts):
            new_times = (
                torch.arange(1, count + 1, device=times.device) * max_silence_s + start
            )
            if torch.isin(new_times, feature_dict[Feats.TIMES][row]).any():
                raise ValueError("New times collide with existing times!")

            new_times_l.append(new_times)

        if not new_times_l:
            continue

        new_times = torch.cat(new_times_l)
        add_action_times[row, :] = torch.nan
        add_action_times[row, : new_times.shape[0]] = new_times

    feature_dict = {
        k: torch.cat([v, add_feature_dict[k]], dim=1) for k, v in feature_dict.items()
    }
    sort_idx = torch.argsort(feature_dict[Feats.TIMES], dim=1)
    feature_dict = {k: v.gather(1, sort_idx) for k, v in feature_dict.items()}

    return feature_dict


def get_window_feature_dict(
    X: dict[Feats, torch.Tensor], dt: float, max_silence_s: float, features: list[Feats]
) -> dict[Feats, torch.Tensor]:
    # (B, L)
    times = X[Feats.TIMES]
    bin_idx = (times // dt).long()

    if bin_idx.min() < 0:
        raise ValueError("Negative bin indices found!")

    feature_dict: dict[Feats, torch.Tensor] = {}
    device = times.device
    shape = (times.shape[0], bin_idx.max() + 1)

    up_counts = torch.zeros(shape, device=device).scatter_add_(
        1, bin_idx, (X[Feats.DIRS] == UPLOAD).float()
    )
    down_counts = torch.zeros(shape, device=device).scatter_add_(
        1, bin_idx, (X[Feats.DIRS] == DOWNLOAD).float()
    )
    times_ = torch.zeros(shape, device=device).scatter_(
        1, bin_idx, bin_idx.float() * dt
    )

    mask = (up_counts != 0) | (down_counts != 0)
    max_l = mask.sum(dim=1).max()
    feature_dict[Feats.UP_COUNT] = _flush_left(up_counts, mask)[:, :max_l]
    feature_dict[Feats.DOWN_COUNT] = _flush_left(down_counts, mask)[:, :max_l]

    times_ = _flush_left(times_, mask)[:, :max_l]

    times_ = _fill_after_seq_end(times_, pad_val=0)
    feature_dict[Feats.TIMES] = times_
    feature_dict = _add_actions_to_silence_periods(feature_dict, max_silence_s)

    dts = feature_dict[Feats.TIMES].diff(
        dim=1, prepend=torch.zeros((times.shape[0], 1), device=times.device)
    )

    feature_dict[Feats.Dt] = dts
    mask = feature_dict[Feats.TIMES].isfinite()
    max_l = mask.sum(dim=1).max()
    # dict[Feats, Tensor (B, max_l)]
    feature_dict = {
        k: _flush_left(v, mask, pad_val=torch.nan)[:, :max_l]
        for k, v in feature_dict.items()
    }

    feature_dict[Feats.SEQ_LENS] = mask.sum(dim=1)

    if Feats.SILENCE_FLAG in features:
        feature_dict[Feats.SILENCE_FLAG] = (
            (feature_dict[Feats.UP_COUNT] == 0) & (feature_dict[Feats.DOWN_COUNT] == 0)
        ).float()

    if (max_s := feature_dict[Feats.Dt].diff(dim=1).max()) > max_silence_s:
        logger.warning(
            f"Found max silence {max_s:.4f}, whereas you wish max silence = {max_silence_s:.4f}."
        )

    if not all(f in feature_dict for f in features):
        raise ValueError("Some requested features are missing!")

    for f, k in zip((Feats.UP_COUNT, Feats.DOWN_COUNT), (UPLOAD, DOWNLOAD)):
        if (
            torch.where(feature_dict[f].isfinite(), feature_dict[f], 0).sum(dim=1)
            != (X[Feats.DIRS] == k).sum(dim=1)
        ).any():
            print(f, k)
            print(feature_dict[f].sum(dim=1), (X[Feats.DIRS] == k).sum(dim=1))
            breakpoint()
            raise ValueError("Missing packets")

    return feature_dict
