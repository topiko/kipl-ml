from __future__ import annotations

from dataclasses import dataclass

import torch

from kipl_ml.data.utils import DOWNLOAD, UPLOAD
from kipl_ml.logging.logger import get_logger
from kipl_ml.rl.enums import Actions
from kipl_ml.rl.utils import _fill_after_seq_end, _flush_left
from kipl_ml.trace.enums import Feats

logger = get_logger(__name__)


@dataclass
class TraceExecState:
    """Accumulate stepwise actions and finalize once.

    This avoids re-sorting and re-padding the trace for every step.
    """

    X_base: dict[Feats, torch.Tensor]

    def __post_init__(self) -> None:
        if set(self.X_base.keys()) != {Feats.TIMES, Feats.DIRS, Feats.PADDING}:
            raise ValueError("X_base must contain TIMES/DIRS/PADDING")
        self.device = self.X_base[Feats.TIMES].device
        self.dtype_t = self.X_base[Feats.TIMES].dtype
        self.dtype_d = self.X_base[Feats.DIRS].dtype

        self._app_times: list[torch.Tensor] = []
        self._app_dirs: list[torch.Tensor] = []
        self._app_pad: list[torch.Tensor] = []
        self._app_trace_idx: list[torch.Tensor] = []

        self._delay_t0: list[torch.Tensor] = []
        self._delay_d: list[torch.Tensor] = []
        self._delay_trace_idx: list[torch.Tensor] = []

        # Fixed-mode send schedules (expanded only at finalize).
        self._fixed_send_time: list[torch.Tensor] = []
        self._fixed_send_count: list[torch.Tensor] = []
        self._fixed_send_dir: list[torch.Tensor] = []
        self._fixed_send_trace_idx: list[torch.Tensor] = []

    def step(
        self,
        trace_idx: torch.Tensor,
        times: torch.Tensor,
        actions: dict[Actions, torch.Tensor],
    ) -> None:
        """Record one step's actions.

        trace_idx: (N,) indices into batch.
        times: (N, 1) act times.
        actions: (N, 1) tensors.
        """

        if trace_idx.ndim != 1:
            raise ValueError("trace_idx must be (N,)")
        if times.ndim != 2 or times.shape[1] != 1:
            raise ValueError("times must be (N,1)")
        if trace_idx.shape[0] != times.shape[0]:
            raise ValueError("trace_idx and times batch mismatch")

        N = int(trace_idx.shape[0])
        if N == 0:
            return

        # Delay events (exclusive).
        if Actions.DELAY in actions:
            delay_s = actions[Actions.DELAY]
            if delay_s.shape != times.shape:
                raise ValueError("DELAY must match times shape")
            mask = (delay_s > 0).squeeze(1)
            if mask.any():
                self._delay_trace_idx.append(trace_idx[mask].detach().clone())
                self._delay_t0.append(times[mask].detach().clone().squeeze(1))
                self._delay_d.append(delay_s[mask].detach().clone().squeeze(1))

        # Enforce delay exclusivity for step execution.
        delay_mask = None
        if Actions.DELAY in actions:
            delay_mask = (actions[Actions.DELAY] > 0)

        # Padding sends.
        if Actions.SEND_COUNT_UP not in actions or Actions.SEND_COUNT_DOWN not in actions:
            raise ValueError("Missing SEND_COUNT actions")

        start_times = times.squeeze(1)
        start_times = torch.where(start_times == 0, start_times + 1e-9, start_times)

        def _record_dir(direction: str, dir_val: int) -> None:
            send_counts = (
                actions[Actions.SEND_COUNT_UP]
                if direction == "up"
                else actions[Actions.SEND_COUNT_DOWN]
            ).squeeze(1)

            send_counts = send_counts.to(dtype=torch.long)

            if delay_mask is not None and delay_mask.any():
                send_counts = torch.where(delay_mask.squeeze(1), 0, send_counts)

            if send_counts.max().item() <= 0:
                return

            decay_times, send_mode = _get_times_and_mode(actions, direction=direction)
            decay_times = decay_times.squeeze(1)

            if send_mode == "fixed":
                m = send_counts > 0
                if not m.any():
                    return
                t_send = (start_times + decay_times).to(dtype=self.dtype_t)
                self._fixed_send_trace_idx.append(trace_idx[m].detach().clone())
                self._fixed_send_time.append(t_send[m].detach().clone())
                self._fixed_send_count.append(send_counts[m].detach().clone())
                self._fixed_send_dir.append(
                    torch.full(
                        (int(m.sum().item()),),
                        float(dir_val),
                        device=self.device,
                        dtype=self.dtype_d,
                    )
                )
                return

            # Flatten into per-packet times + owning trace idx.
            out_times_l: list[torch.Tensor] = []
            out_idx_l: list[torch.Tensor] = []

            max_c = int(send_counts.max().item())
            for c in range(1, max_c + 1):
                m = send_counts == c
                if not m.any():
                    continue
                st = start_times[m]
                dt = decay_times[m]

                if send_mode != "spread":
                    raise ValueError(f"Invalid send mode: {send_mode}")
                send_times = (
                    torch.rand((st.shape[0], c), device=self.device) * dt.unsqueeze(1)
                    + st.unsqueeze(1)
                )

                out_times_l.append(send_times.flatten().to(dtype=self.dtype_t))
                out_idx_l.append(trace_idx[m].repeat_interleave(c))

            if not out_times_l:
                return

            out_times = torch.cat(out_times_l, dim=0)
            out_idx = torch.cat(out_idx_l, dim=0)
            out_dirs = torch.full_like(out_times, float(dir_val), dtype=self.dtype_d)
            out_pad = torch.ones_like(out_times, dtype=self.dtype_t)

            self._app_times.append(out_times)
            self._app_dirs.append(out_dirs)
            self._app_pad.append(out_pad)
            self._app_trace_idx.append(out_idx)

        _record_dir("up", UPLOAD)
        _record_dir("down", DOWNLOAD)

    def finalize(self) -> dict[Feats, torch.Tensor]:
        """Build a finalized trace dict like send_exec would produce."""

        B = int(self.X_base[Feats.TIMES].shape[0])

        # Expand fixed-mode schedules.
        if self._fixed_send_time:
            st = torch.cat(self._fixed_send_time, dim=0)
            sc = torch.cat(self._fixed_send_count, dim=0)
            sd = torch.cat(self._fixed_send_dir, dim=0)
            si = torch.cat(self._fixed_send_trace_idx, dim=0)

            # (n_events,) -> (n_packets,)
            rep = sc.to(dtype=torch.long)
            fixed_idx = si.repeat_interleave(rep)
            fixed_times = st.repeat_interleave(rep)
            fixed_dirs = sd.repeat_interleave(rep)
            fixed_pad = torch.ones_like(fixed_times, dtype=self.dtype_t)

            self._app_trace_idx.append(fixed_idx)
            self._app_times.append(fixed_times)
            self._app_dirs.append(fixed_dirs)
            self._app_pad.append(fixed_pad)

        if self._app_times:
            app_times = torch.cat(self._app_times, dim=0)
            app_dirs = torch.cat(self._app_dirs, dim=0)
            app_pad = torch.cat(self._app_pad, dim=0)
            app_idx = torch.cat(self._app_trace_idx, dim=0)
        else:
            app_times = torch.zeros((0,), device=self.device, dtype=self.dtype_t)
            app_dirs = torch.zeros((0,), device=self.device, dtype=self.dtype_d)
            app_pad = torch.zeros((0,), device=self.device, dtype=self.dtype_t)
            app_idx = torch.zeros((0,), device=self.device, dtype=torch.long)

        # Delay events
        if self._delay_t0:
            delay_t0 = torch.cat(self._delay_t0, dim=0)
            delay_d = torch.cat(self._delay_d, dim=0)
            delay_idx = torch.cat(self._delay_trace_idx, dim=0)
        else:
            delay_t0 = torch.zeros((0,), device=self.device, dtype=self.dtype_t)
            delay_d = torch.zeros((0,), device=self.device, dtype=self.dtype_t)
            delay_idx = torch.zeros((0,), device=self.device, dtype=torch.long)

        out_times_l: list[torch.Tensor] = []
        out_dirs_l: list[torch.Tensor] = []
        out_pad_l: list[torch.Tensor] = []
        max_len = 0

        base_times = self.X_base[Feats.TIMES]
        base_dirs = self.X_base[Feats.DIRS]
        base_pad = self.X_base[Feats.PADDING]

        for i in range(B):
            m_base = base_dirs[i] != 0
            t = base_times[i][m_base]
            d = base_dirs[i][m_base]
            p = base_pad[i][m_base]

            m_app = app_idx == i
            if m_app.any():
                t = torch.cat([t, app_times[m_app]], dim=0)
                d = torch.cat([d, app_dirs[m_app]], dim=0)
                p = torch.cat([p, app_pad[m_app]], dim=0)

            # Apply delay as cumulative shifts based on event time.
            m_del = delay_idx == i
            if m_del.any():
                t0 = delay_t0[m_del].to(dtype=self.dtype_t)
                dd = delay_d[m_del].to(dtype=self.dtype_t)
                order = torch.argsort(t0)
                t0 = t0[order]
                dd = dd[order]
                cum = torch.cumsum(dd, dim=0)
                # idx in [0..len(t0)] where to insert; shift uses right=True.
                ins = torch.searchsorted(t0, t, right=True) - 1
                shift = torch.where(ins >= 0, cum[ins.clamp(min=0)], torch.zeros_like(t))
                t = t + shift

            # Sort by time; tie-break by (dir, padding) for stability.
            if t.numel() > 0:
                # Round for stable tie-breaking when many identical timestamps.
                tr = torch.round(t * 1e6) / 1e6
                # lexsort equivalent: sort by (tr, d, p)
                key = tr * 10 + d * 1 + p * 0.1
                order = torch.argsort(key)
                t = t[order]
                d = d[order]
                p = p[order]

            max_len = max(max_len, int(t.numel()))
            out_times_l.append(t)
            out_dirs_l.append(d)
            out_pad_l.append(p)

        if max_len == 0:
            max_len = 1

        times_out = torch.zeros((B, max_len), device=self.device, dtype=self.dtype_t)
        dirs_out = torch.zeros((B, max_len), device=self.device, dtype=self.dtype_d)
        pad_out = torch.zeros((B, max_len), device=self.device, dtype=self.dtype_t)

        for i in range(B):
            n = int(out_times_l[i].numel())
            if n == 0:
                continue
            times_out[i, :n] = out_times_l[i]
            dirs_out[i, :n] = out_dirs_l[i]
            pad_out[i, :n] = out_pad_l[i]

        mask = dirs_out != 0
        times_out = _fill_after_seq_end(times_out, pad_val=0, fill_val="max")
        X = {Feats.TIMES: times_out, Feats.DIRS: dirs_out, Feats.PADDING: pad_out}
        X = {k: _flush_left(v, mask, pad_val=0) for k, v in X.items()}
        X = _sort_feature_dict(X)

        max_l = (X[Feats.DIRS] != 0).sum(dim=1).max()
        X = {k: v[:, :max_l] for k, v in X.items()}
        return X


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


def _get_times_and_mode(
    actions: dict[Actions, torch.Tensor], direction: str
) -> tuple[torch.Tensor, str]:
    if direction not in {"up", "down"}:
        raise ValueError("Invalid direction detected")
    spread_time_key = (
        Actions.SPREAD_TIME_UP if direction == "up" else Actions.SPREAD_TIME_DOWN
    )
    send_after_time_key = (
        Actions.SEND_UP_AFTER_TIME
        if direction == "up"
        else Actions.SEND_DOWN_AFTER_TIME
    )

    if spread_time_key in actions:
        times = actions[spread_time_key]
        mode = "spread"
    elif send_after_time_key in actions:
        times = actions[send_after_time_key]
        mode = "fixed"
    else:
        raise ValueError("Invalid send mode detected")

    return times, mode


def send_exec(
    X: dict[Feats, torch.Tensor],
    times: torch.Tensor,
    actions: dict[Actions, torch.Tensor],
) -> dict[Feats, torch.Tensor]:
    X = {k: v.clone() for k, v in X.items()}
    actions = {k: v.clone() for k, v in actions.items()}

    # TODO: improve this by removing the batch dim loops...

    if Actions.DELAY in actions:
        delay_s = actions[Actions.DELAY]
        if delay_s.shape != times.shape:
            raise ValueError("DELAY tensor must match times shape")

        delay_mask = delay_s > 0

        # Delay is only supported for stepwise execution (T=1) when it is
        # actually used. The DELAY tensor may be present (all zeros) in
        # multi-step runs.
        if delay_mask.any() and times.shape[1] != 1:
            raise NotImplementedError("DELAY only supported for stepwise send_exec (T=1)")

        if delay_mask.any():
            # Apply delay: shift all packets at/after the action time.
            for i in range(times.shape[0]):
                if not bool(delay_mask[i, 0].item()):
                    continue
                t0 = float(times[i, 0].item())
                d = float(delay_s[i, 0].item())
                X[Feats.TIMES][i] = torch.where(
                    X[Feats.TIMES][i] >= t0, X[Feats.TIMES][i] + d, X[Feats.TIMES][i]
                )

            # Delay is exclusive; enforce no padding sends in this step.
            for k in (
                Actions.SEND_COUNT_UP,
                Actions.SEND_COUNT_DOWN,
                Actions.SPREAD_TIME_UP,
                Actions.SPREAD_TIME_DOWN,
                Actions.SEND_UP_AFTER_TIME,
                Actions.SEND_DOWN_AFTER_TIME,
            ):
                if k in actions:
                    actions[k] = torch.where(delay_mask, 0, actions[k])
    send_up_c = actions[Actions.SEND_COUNT_UP]
    times_up, send_mode_up = _get_times_and_mode(actions, direction="up")

    send_down_c = actions[Actions.SEND_COUNT_DOWN]
    times_down, send_mode_down = _get_times_and_mode(actions, direction="down")

    if Feats.PADDING not in X:
        X[Feats.PADDING] = torch.zeros_like(X[Feats.TIMES])

    def _sample_send_times(
        send_counts: torch.Tensor,
        times: torch.Tensor,
        decay_times: torch.Tensor,
        send_mode: str,
    ):
        send_times_l = []
        for c in range(1, send_counts.max().int() + 1):
            mask = send_counts == c

            start_times = times[mask]

            # (L, c)
            if send_mode == "spread":
                send_times = torch.rand(
                    (start_times.shape[0], c), device=times.device
                ) * decay_times[mask].unsqueeze(1) + start_times.unsqueeze(1)
            elif send_mode == "fixed":
                send_times = torch.ones(
                    (start_times.shape[0], c), device=times.device
                ) * decay_times[mask].unsqueeze(1) + start_times.unsqueeze(1)
            else:
                raise ValueError(f"Invalid send mode: {send_mode}")

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
            send_counts=sup_c,
            times=times_,
            decay_times=dec_t_up,
            send_mode=send_mode_up,
        )

        send_times_down = _sample_send_times(
            send_counts=sdown_c,
            times=times_,
            decay_times=dec_t_down,
            send_mode=send_mode_down,
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

    if set(X.keys()) != {Feats.TIMES, Feats.DIRS, Feats.PADDING}:
        raise ValueError("Invalid set of features detected")

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

    if (X[Feats.TIMES].diff(dim=1) < 0).any():
        raise ValueError("Unsorted times detected after send_exec")

    return X
