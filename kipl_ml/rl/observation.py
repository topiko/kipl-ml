from __future__ import annotations

import torch

from kipl_ml.data.utils import DOWNLOAD, UPLOAD
from kipl_ml.logging.logger import get_logger
from kipl_ml.rl.utils import _fill_w_last, _flush_left
from kipl_ml.trace.enums import Feats

logger = get_logger(__name__)


def _add_actions_to_silence_periods(
    feature_dict: dict[Feats, torch.Tensor], max_silence_s: float
) -> tuple[dict[Feats, torch.Tensor], bool]:
    times = feature_dict[Feats.TIMES]
    dts = times.diff(
        dim=1, prepend=torch.zeros((times.shape[0], 1), device=times.device)
    )

    dt_ = 1e-3
    if dts.max() <= max_silence_s + dt_:
        return feature_dict, False

    add_action_times = torch.zeros(
        (times.shape[0], (dts > max_silence_s + dt_).sum(dim=1).max()),
        device=times.device,
    )

    add_feature_dict: dict[Feats, torch.Tensor] = {
        Feats.UP_COUNT: add_action_times.clone(),
        Feats.DOWN_COUNT: add_action_times.clone(),
        Feats.TIMES: add_action_times,
    }
    for row in range(dts.shape[0]):
        mask = dts[row] > max_silence_s + dt_
        n = mask.sum()
        silence_starts = times[row, mask.roll(-1)]
        add_action_times[row, :] = times[row, -1]
        add_action_times[row, :n] = silence_starts + max_silence_s

        add_feature_dict[Feats.UP_COUNT][row, :n] = 0
        add_feature_dict[Feats.DOWN_COUNT][row, :n] = 0

    feature_dict = {
        k: torch.cat([v, add_feature_dict[k]], dim=1) for k, v in feature_dict.items()
    }
    sort_idx = torch.argsort(feature_dict[Feats.TIMES], dim=1)
    feature_dict = {k: torch.gather(v, 1, sort_idx) for k, v in feature_dict.items()}

    return feature_dict, True


def get_window_feature_dict(
    X: dict[Feats, torch.Tensor], dt: float, max_silence_s: float, features: list[Feats]
) -> dict[Feats, torch.Tensor]:
    # (B, L)
    times = X[Feats.TIMES]
    max_t = times.max().item()

    feature_dict_l: dict[Feats, list[torch.Tensor]] = {}
    for t in torch.arange(0, max_t + dt, dt, device=times.device):
        t1 = t
        t2 = t + dt
        mask = (times >= t1) & (times < t2)

        n_rows = times.shape[0]
        n_cols = mask.sum(dim=1).max()

        row_idxs = (
            torch.arange(n_rows, device=times.device)
            .unsqueeze(1)
            .expand(-1, times.shape[1])
        )[mask]
        col_idxs = (mask.cumsum(dim=1) - 1)[mask]

        fdirs = torch.zeros((n_rows, n_cols), device=times.device)
        fdirs[row_idxs, col_idxs] = X[Feats.DIRS][mask]

        ftimes = torch.zeros((n_rows, n_cols), device=times.device)
        ftimes[row_idxs, col_idxs] = X[Feats.TIMES][mask]

        if Feats.UP_COUNT in features:
            feature_dict_l.setdefault(Feats.UP_COUNT, []).append(
                (fdirs == UPLOAD).sum(dim=1, keepdim=True)
            )
        if Feats.DOWN_COUNT in features:
            feature_dict_l.setdefault(Feats.DOWN_COUNT, []).append(
                (fdirs == DOWNLOAD).sum(dim=1, keepdim=True)
            )

        feature_dict_l.setdefault(Feats.TIMES, []).append(
            torch.ones((n_rows, 1), device=times.device) * t
        )

    feature_dict: dict[Feats, torch.Tensor] = {
        k: torch.cat(v, dim=1) for k, v in feature_dict_l.items()
    }

    # (B, L)
    mask = (feature_dict[Feats.UP_COUNT] != 0) | (feature_dict[Feats.DOWN_COUNT] != 0)

    max_l = mask.sum(dim=1).max()

    # dict[Feats, Tensor (B, max_l)]
    feature_dict = {k: _flush_left(v, mask)[:, :max_l] for k, v in feature_dict.items()}

    # Fix the zero padding at the end of time seqs.
    feature_dict[Feats.TIMES] = _fill_w_last(feature_dict[Feats.TIMES], pad_val=0)

    updated = True
    while updated:
        feature_dict, updated = _add_actions_to_silence_periods(
            feature_dict, max_silence_s
        )
        # for k, v in feature_dict.items():
        # print(k, v.shape)
        # if k == Feats.TIMES:
        #     print(v.diff(dim=1).max(), updated)
        # print()

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
