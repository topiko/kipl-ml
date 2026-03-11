from __future__ import annotations

from dataclasses import dataclass
from math import gcd

import torch

from kipl_ml.data.utils import DOWNLOAD, UPLOAD
from kipl_ml.logging.logger import get_logger
from kipl_ml.rl.enums import Actions
from kipl_ml.rl.utils import _flush_left, fill_after_seq_end
from kipl_ml.trace.enums import Feats

logger = get_logger(__name__)


def _time_to_bin_idx(times: torch.Tensor, dt: float) -> torch.Tensor:
    if dt <= 0:
        raise ValueError(f"dt must be > 0, got {dt}")

    dt_us = max(1, int(round(float(dt) * 1e6)))
    t_us = torch.round(times * 1e6).to(torch.long)
    return torch.div(t_us, dt_us, rounding_mode="floor")


def _boundary_time_to_bin_idx(times: torch.Tensor, dt: float) -> torch.Tensor:
    if dt <= 0:
        raise ValueError(f"dt must be > 0, got {dt}")
    return torch.round(times.to(torch.float64) / float(dt)).to(torch.long)


def _duration_to_bin_offsets(durations: torch.Tensor, dt: float) -> torch.Tensor:
    if dt <= 0:
        raise ValueError(f"dt must be > 0, got {dt}")

    dt_us = max(1, int(round(float(dt) * 1e6)))
    d_us = torch.round(durations * 1e6).to(torch.long)
    return torch.div(d_us + (dt_us // 2), dt_us, rounding_mode="floor")


def _apply_delay_clamp_bins_inplace(
    t: torch.Tensor,
    start_bins: torch.Tensor,
    shift_bins: torch.Tensor,
    dt_s: float,
) -> torch.Tensor:
    if start_bins.numel() == 0:
        return t

    order = torch.argsort(start_bins)
    starts = start_bins[order].to(torch.long)
    shifts = shift_bins[order].to(torch.long)

    bins = _time_to_bin_idx(t, dt_s)
    for s, sh in zip(starts, shifts):
        sh_i = int(sh.item())
        if sh_i <= 0:
            continue
        s_i = int(s.item())
        e_i = s_i + sh_i
        m = (bins >= s_i) & (bins < e_i)
        if bool(m.any().item()):
            bins = torch.where(m, torch.full_like(bins, e_i), bins)
            t = torch.where(m, torch.full_like(t, float(e_i) * dt_s), t)
    return t


def _apply_delay_clamp_inplace(
    t: torch.Tensor,
    delay_starts: torch.Tensor,
    delay_durations: torch.Tensor,
) -> torch.Tensor:
    """Clamp times in each delay window [t0, t0+d) to t0+d."""
    if delay_starts.numel() == 0:
        return t

    order = torch.argsort(delay_starts)
    t0s = delay_starts[order]
    ds = delay_durations[order]

    # Prefer bin-space clamping when delay duration is constant.
    pos = ds[ds > 0]
    if pos.numel() > 0:
        d0 = float(pos[0].item())
        if torch.allclose(pos, torch.full_like(pos, d0), atol=1e-6, rtol=0.0):
            dt_q = round(d0 * 1e6)
            dt_s = dt_q / 1e6
            if dt_s > 0:
                start_bins = _boundary_time_to_bin_idx(t0s, dt_s)
                shift_bins = _duration_to_bin_offsets(ds, dt_s)
                return _apply_delay_clamp_bins_inplace(
                    t,
                    start_bins=start_bins,
                    shift_bins=shift_bins,
                    dt_s=dt_s,
                )

    # Fallback: float interval clamping.
    for t0, d in zip(t0s, ds):
        t1 = t0 + d
        t = torch.where((t >= t0) & (t < t1), t1, t)
    return t


@dataclass
class TraceExecState:
    """Accumulate stepwise actions and finalize once.

    This avoids re-sorting and re-padding the trace for every step.
    """

    X_base: dict[Feats, torch.Tensor]
    time_step_s: float

    def __post_init__(self) -> None:
        if set(self.X_base.keys()) != {Feats.TIMES, Feats.DIRS, Feats.PADDING}:
            raise ValueError("X_base must contain TIMES/DIRS/PADDING")
        if self.time_step_s <= 0:
            raise ValueError(f"time_step_s must be > 0, got {self.time_step_s}")
        self.device = self.X_base[Feats.TIMES].device
        self.dtype_t = self.X_base[Feats.TIMES].dtype
        self.dtype_d = self.X_base[Feats.DIRS].dtype

        self._app_times: list[torch.Tensor] = []
        self._app_dirs: list[torch.Tensor] = []
        self._app_pad: list[torch.Tensor] = []
        self._app_trace_idx: list[torch.Tensor] = []

        self._delay_start_bin: list[torch.Tensor] = []
        self._delay_shift_bin: list[torch.Tensor] = []
        self._delay_trace_idx: list[torch.Tensor] = []

        # Fixed-mode send schedules (expanded only at finalize).
        self._fixed_send_bin: list[torch.Tensor] = []
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
        act_bins = _boundary_time_to_bin_idx(times.squeeze(1), self.time_step_s)

        if Actions.DELAY in actions:
            delay_s = actions[Actions.DELAY]
            if delay_s.shape != times.shape:
                raise ValueError("DELAY must match times shape")
            delay_bins = _duration_to_bin_offsets(delay_s.squeeze(1), self.time_step_s)
            mask = delay_bins > 0
            if mask.any():
                self._delay_trace_idx.append(trace_idx[mask].detach().clone())
                self._delay_start_bin.append(act_bins[mask].detach().clone())
                self._delay_shift_bin.append(delay_bins[mask].detach().clone())

        # Enforce delay exclusivity for step execution.
        delay_mask = None
        if Actions.DELAY in actions:
            delay_mask = (actions[Actions.DELAY] > 0)

        # Padding sends.
        if Actions.SEND_COUNT_UP not in actions or Actions.SEND_COUNT_DOWN not in actions:
            raise ValueError("Missing SEND_COUNT actions")

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
            if send_mode != "fixed":
                raise NotImplementedError("send_mode='spread' is deprecated; use fixed")

            decay_bins = _duration_to_bin_offsets(
                decay_times.squeeze(1), self.time_step_s
            )

            m = send_counts > 0
            if not m.any():
                return

            send_bins = act_bins + decay_bins
            self._fixed_send_trace_idx.append(trace_idx[m].detach().clone())
            self._fixed_send_bin.append(send_bins[m].detach().clone())
            self._fixed_send_count.append(send_counts[m].detach().clone())
            self._fixed_send_dir.append(
                torch.full(
                    (int(m.sum().item()),),
                    float(dir_val),
                    device=self.device,
                    dtype=self.dtype_d,
                )
            )

        _record_dir("up", UPLOAD)
        _record_dir("down", DOWNLOAD)

    def finalize(self) -> dict[Feats, torch.Tensor]:
        """Build a finalized trace dict like send_exec would produce."""

        B = int(self.X_base[Feats.TIMES].shape[0])

        # Expand fixed-mode schedules.
        if self._fixed_send_bin:
            st = torch.cat(self._fixed_send_bin, dim=0)
            sc = torch.cat(self._fixed_send_count, dim=0)
            sd = torch.cat(self._fixed_send_dir, dim=0)
            si = torch.cat(self._fixed_send_trace_idx, dim=0)

            # (n_events,) -> (n_packets,)
            rep = sc.to(dtype=torch.long)
            fixed_idx = si.repeat_interleave(rep)
            fixed_bins = st.repeat_interleave(rep)
            fixed_times = fixed_bins.to(self.dtype_t) * float(self.time_step_s)
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
        if self._delay_start_bin:
            delay_start_bin = torch.cat(self._delay_start_bin, dim=0).to(torch.long)
            delay_shift_bin = torch.cat(self._delay_shift_bin, dim=0).to(torch.long)
            delay_idx = torch.cat(self._delay_trace_idx, dim=0)
        else:
            delay_start_bin = torch.zeros((0,), device=self.device, dtype=torch.long)
            delay_shift_bin = torch.zeros((0,), device=self.device, dtype=torch.long)
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

            # Apply delay by clamping within each delay window.
            m_del = delay_idx == i
            if m_del.any():
                start_bins = delay_start_bin[m_del]
                shift_bins = delay_shift_bin[m_del]
                t = _apply_delay_clamp_bins_inplace(
                    t,
                    start_bins=start_bins,
                    shift_bins=shift_bins,
                    dt_s=self.time_step_s,
                )

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
        times_out = fill_after_seq_end(times_out, mask, fill_val="max")
        X = {Feats.TIMES: times_out, Feats.DIRS: dirs_out, Feats.PADDING: pad_out}
        X = {k: _flush_left(v, mask, pad_val=0) for k, v in X.items()}
        X = _sort_feature_dict(X)

        max_l = (X[Feats.DIRS] != 0).sum(dim=1).max()
        X = {k: v[:, :max_l] for k, v in X.items()}
        return X


def _sort_feature_dict(
    feature_dict: dict[Feats, torch.Tensor],
) -> dict[Feats, torch.Tensor]:
    times = feature_dict[Feats.TIMES]
    try:
        # Stable sort preserves relative order for equal timestamps.
        indices = torch.argsort(times, dim=1, stable=True)
    except TypeError:
        # Fallback for older torch: stable tie-break via column index.
        B, L = times.shape
        tie = torch.arange(L, device=times.device, dtype=torch.long).unsqueeze(0).expand(B, -1)
        time_key = torch.round(times.to(torch.float64) * 1e6).to(torch.long)
        key = time_key * (L + 1) + tie
        indices = torch.argsort(key, dim=1)

    sorted_times = times.gather(1, indices)

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
        raise NotImplementedError("send_mode='spread' is deprecated; use fixed")
    if send_after_time_key in actions:
        times = actions[send_after_time_key]
        mode = "fixed"
    else:
        raise ValueError("Invalid send mode detected")

    return times, mode


def _infer_time_step_s(
    times: torch.Tensor,
    actions: dict[Actions, torch.Tensor],
) -> float:
    cand_us: list[int] = []

    if Actions.DELAY in actions:
        d = actions[Actions.DELAY]
        pos = d[torch.isfinite(d) & (d > 0)]
        if pos.numel() > 0:
            cand_us.extend(
                [int(v) for v in torch.round(pos * 1e6).to(torch.long).detach().cpu().tolist()]
            )

    for k in (Actions.SEND_UP_AFTER_TIME, Actions.SEND_DOWN_AFTER_TIME):
        if k in actions:
            a = actions[k]
            pos = a[torch.isfinite(a) & (a > 0)]
            if pos.numel() > 0:
                cand_us.extend(
                    [
                        int(v)
                        for v in torch.round(pos * 1e6).to(torch.long).detach().cpu().tolist()
                    ]
                )

    if times.ndim != 2:
        raise ValueError("times must be (B,T) to infer time_step_s")
    dt = times[:, 1:] - times[:, :-1]
    m = torch.isfinite(times[:, 1:]) & torch.isfinite(times[:, :-1]) & (dt > 0)
    pos = dt[m]
    if pos.numel() > 0:
        cand_us.extend(
            [int(v) for v in torch.round(pos * 1e6).to(torch.long).detach().cpu().tolist()]
        )

    cand_us = [v for v in cand_us if v > 0]
    if not cand_us:
        return 1e-6

    dt_us = cand_us[0]
    for v in cand_us[1:]:
        dt_us = gcd(dt_us, v)
        if dt_us == 1:
            break

    dt_q = max(1, dt_us) / 1e6
    if dt_q <= 0:
        raise ValueError(f"Inferred non-positive time_step_s: {dt_q}")
    return dt_q


def send_exec(
    X: dict[Feats, torch.Tensor],
    times: torch.Tensor,
    actions: dict[Actions, torch.Tensor],
    *,
    time_step_s: float | None = None,
) -> dict[Feats, torch.Tensor]:
    if set(X.keys()) - {Feats.TIMES, Feats.DIRS, Feats.PADDING}:
        raise ValueError("Invalid set of features detected")

    X = {k: v.clone() for k, v in X.items()}
    actions = {k: v.clone() for k, v in actions.items()}

    if Feats.PADDING not in X:
        X[Feats.PADDING] = torch.zeros_like(X[Feats.TIMES])

    if times.ndim != 2:
        raise ValueError("times must be (B,T)")

    for k, v in actions.items():
        if v.shape != times.shape:
            raise ValueError(f"Action tensor {k} must match times shape")

    if time_step_s is None:
        time_step_s = _infer_time_step_s(times, actions)

    exec_state = TraceExecState(
        {
            Feats.TIMES: X[Feats.TIMES],
            Feats.DIRS: X[Feats.DIRS],
            Feats.PADDING: X[Feats.PADDING],
        },
        time_step_s=float(time_step_s),
    )

    _, T = times.shape
    for t_i in range(T):
        active = times[:, t_i].isfinite()
        if not bool(active.any().item()):
            continue

        trace_idx = torch.where(active)[0]
        step_times = times[active, t_i : t_i + 1]
        step_actions = {k: v[active, t_i : t_i + 1] for k, v in actions.items()}
        exec_state.step(
            trace_idx=trace_idx,
            times=step_times,
            actions=step_actions,
        )

    return exec_state.finalize()
