from __future__ import annotations

import torch

from kipl_ml.data.utils import DOWNLOAD, UPLOAD
from kipl_ml.logging.logger import get_logger
from kipl_ml.rl.enums import Actions
from kipl_ml.rl.utils import _flush_left
from kipl_ml.trace.enums import Feats

logger = get_logger(__name__)


def _sort_feature_dict(
    feature_dict: dict[Feats, torch.Tensor],
) -> dict[Feats, torch.Tensor]:
    sorted_times, indices = torch.sort(feature_dict[Feats.TIMES], dim=1)

    dirs = feature_dict[Feats.DIRS].gather(1, indices)
    sorted_times = _flush_left(sorted_times, dirs != 0)
    padding = _flush_left(feature_dict[Feats.PADDING].gather(1, indices), dirs != 0)
    dirs = _flush_left(dirs, dirs != 0)

    feature_dict[Feats.TIMES] = sorted_times
    feature_dict[Feats.DIRS] = dirs
    feature_dict[Feats.PADDING] = padding

    if set(feature_dict.keys()) != {Feats.TIMES, Feats.DIRS, Feats.PADDING}:
        raise NotImplementedError(
            "Sorting for additional features not implemented yet."
        )

    return feature_dict


def send_exec(
    X: dict[Feats, torch.Tensor],
    times: torch.Tensor,
    actions: dict[Actions, torch.Tensor],
) -> dict[Feats, torch.Tensor]:
    # TODO: improve this by removing the batch dim loops...
    send_up_c = actions[Actions.SEND_COUNT_UP]
    times_up = actions[Actions.SEND_TIME_UP]
    send_down_c = actions[Actions.SEND_COUNT_DOWN]
    times_down = actions[Actions.SEND_TIME_DOWN]

    if Feats.PADDING not in X:
        X[Feats.PADDING] = torch.zeros_like(X[Feats.TIMES])

    def _sample_send_times(
        send_counts: torch.Tensor,
        times: torch.Tensor,
        decay_times: torch.Tensor,
    ):
        send_times_l = []
        for c in range(1, send_counts.max().int() + 1):
            mask = send_counts == c

            start_times = times[mask]
            # (L, c)
            send_times = torch.rand(
                (start_times.shape[0], c), device=times.device
            ) * decay_times[mask].unsqueeze(1) + start_times.unsqueeze(1)
            send_times_l.append(send_times.flatten())

        send_times = torch.cat(send_times_l, dim=0)
        return send_times

    def _build_tensors(send_times_list: list[torch.Tensor], dir_: int):
        max_len = max(len(t_) for t_ in send_times_list)
        times_tensor = torch.zeros((len(send_times_list), max_len), device=times.device)
        dirs_tensor = torch.zeros((len(send_times_list), max_len), device=times.device)
        padding_tensor = torch.zeros(
            (len(send_times_list), max_len), device=times.device
        )

        for i in range(len(send_times_list)):
            send_times_ = send_times_list[i]
            times_tensor[i, : len(send_times_)] = send_times_
            dirs_tensor[i, : len(send_times_)] = dir_
            padding_tensor[i, : len(send_times_)] = 1.0

        return times_tensor, dirs_tensor, padding_tensor

    send_times_up_l = []
    send_times_down_l = []
    for i in range(times.shape[0]):
        times_ = times[i]
        sup_c = send_up_c[i]
        sdown_c = send_down_c[i]
        t_up = times_up[i]
        t_down = times_down[i]

        send_times_up = _sample_send_times(
            send_counts=sup_c, times=times_, decay_times=t_up
        )

        send_times_down = _sample_send_times(
            send_counts=sdown_c, times=times_, decay_times=t_down
        )

        send_times_up_l.append(send_times_up)
        send_times_down_l.append(send_times_down)

    for send_times, dir_ in zip(
        [send_times_up_l, send_times_down_l], [UPLOAD, DOWNLOAD]
    ):
        times_, dirs_, padding_ = _build_tensors(send_times, dir_)
        X[Feats.TIMES] = torch.cat([X[Feats.TIMES], times_], dim=1)
        X[Feats.DIRS] = torch.cat([X[Feats.DIRS], dirs_], dim=1)
        X[Feats.PADDING] = torch.cat([X[Feats.PADDING], padding_], dim=1)

    X = _sort_feature_dict(X)

    return X
