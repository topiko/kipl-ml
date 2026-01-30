from __future__ import annotations

import torch
import time

from kipl_ml.data.utils import DOWNLOAD, UPLOAD
from kipl_ml.logging.logger import get_logger
from kipl_ml.rl.enums import Actions
from kipl_ml.rl.utils import _fill_after_seq_end, _flush_left
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

    # The times are zero padded in the end. Fix this here.
    # ====================================
    if sorted_times.isnan().any():
        raise NotImplementedError("NaN times not supported in sorting yet.")

    rows, cols = torch.where(sorted_times.diff(dim=1) < 0)

    if rows.unique().numel() != len(rows):
        raise ValueError("Invalid times detected")

    for idx_r, idx_c in zip(rows, cols):
        sorted_times[idx_r, idx_c:] = sorted_times[idx_r, idx_c]
    # ====================================

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
    timing: dict[str, float] | None = None,
) -> dict[Feats, torch.Tensor]:
    t0 = time.perf_counter()
    X = {k: v.clone() for k, v in X.items()}

    # TODO: improve this by removing the batch dim loops...
    send_up_c = actions[Actions.SEND_COUNT_UP]
    times_up = actions[Actions.SEND_TIME_UP]
    send_down_c = actions[Actions.SEND_COUNT_DOWN]
    times_down = actions[Actions.SEND_TIME_DOWN]

    if timing is not None:
        timing["send_exec_prep_ms"] = (time.perf_counter() - t0) * 1000

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

        if send_times_l:
            send_times = torch.cat(send_times_l, dim=0)
        else:
            send_times = torch.zeros(0, device=send_counts.device)

        return send_times

    def _build_tensors(send_times_list: list[torch.Tensor], dir_: int):
        max_len = max(len(t_) for t_ in send_times_list)
        times_tensor = torch.zeros((len(send_times_list), max_len), device=times.device)
        dirs_tensor = torch.zeros((len(send_times_list), max_len), device=times.device)
        padding_tensor = torch.zeros(
            (len(send_times_list), max_len), device=times.device
        )

        for i, v in enumerate(send_times_list):
            send_times_ = v
            times_tensor[i, : len(send_times_)] = send_times_
            dirs_tensor[i, : len(send_times_)] = dir_
            padding_tensor[i, : len(send_times_)] = 1.0

        return times_tensor, dirs_tensor, padding_tensor

    send_times_up_l = []
    send_times_down_l = []

    t1 = time.perf_counter()
    for i in range(times.shape[0]):
        # NAN time signals seq has ended.
        mask = times[i].isfinite()
        times_ = times[i][mask]
        # Protection against sending at zero time.
        times_[times_ == 0] += 1e-9

        # Send counts
        sup_c = send_up_c[i][mask]
        sdown_c = send_down_c[i][mask]

        # Decay times
        dec_t_up = times_up[i][mask]
        dec_t_down = times_down[i][mask]

        send_times_up = _sample_send_times(
            send_counts=sup_c, times=times_, decay_times=dec_t_up
        )

        send_times_down = _sample_send_times(
            send_counts=sdown_c, times=times_, decay_times=dec_t_down
        )

        send_times_up_l.append(send_times_up)
        send_times_down_l.append(send_times_down)

    if timing is not None:
        timing["send_exec_sample_loop_ms"] = (time.perf_counter() - t1) * 1000

    t2 = time.perf_counter()
    for send_times, dir_ in zip(
        [send_times_up_l, send_times_down_l], [UPLOAD, DOWNLOAD]
    ):
        times_, dirs_, padding_ = _build_tensors(send_times, dir_)
        X[Feats.TIMES] = torch.cat([X[Feats.TIMES], times_], dim=1)
        X[Feats.DIRS] = torch.cat([X[Feats.DIRS], dirs_], dim=1)
        X[Feats.PADDING] = torch.cat([X[Feats.PADDING], padding_], dim=1)

    if timing is not None:
        timing["send_exec_build_cat_ms"] = (time.perf_counter() - t2) * 1000

    if set(X.keys()) != {Feats.TIMES, Feats.DIRS, Feats.PADDING}:
        raise ValueError("Invalid set of features detected")

    t3 = time.perf_counter()
    # In the above cat, the max in up/down padding does not
    # coinside on the same row -> the tensor becomes zero padded.
    mask = X[Feats.DIRS] != 0
    X = {k: _flush_left(v, mask, pad_val=0) for k, v in X.items()}

    if ((X[Feats.DIRS] != 0).diff(dim=1).sum(dim=1) > 1).any():
        raise ValueError("Non contiguous send actions detected")

    max_l = (X[Feats.DIRS] != 0).sum(dim=1).max()
    X = {k: v[:, :max_l] for k, v in X.items()}

    # The times are not sorted as of now, we pad w. max val.
    X[Feats.TIMES] = _fill_after_seq_end(X[Feats.TIMES], pad_val=0, fill_val="max")

    X = _sort_feature_dict(X)

    if timing is not None:
        timing["send_exec_flush_sort_ms"] = (time.perf_counter() - t3) * 1000
        timing["send_exec_total_ms"] = (time.perf_counter() - t0) * 1000

    if (X[Feats.TIMES].diff(dim=1) < 0).any():
        raise ValueError("Unsorted times detected after send_exec")

    return X
