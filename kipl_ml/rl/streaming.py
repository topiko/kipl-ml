from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from kipl_ml.data.utils import DOWNLOAD, UPLOAD
from kipl_ml.logging.logger import get_logger
from kipl_ml.rl.enums import ActDelay, Actions, NoAction, StepAction, StepActions
from kipl_ml.trace.enums import Feats

logger = get_logger(__name__)


@dataclass
class StepRes:
    aidx: int
    is_done: bool
    w_times: torch.Tensor
    w_dirs: torch.Tensor
    w_decoy: torch.Tensor
    current_bin: int
    stepped_bins: int


class SendBuffer:
    def __init__(
        self,
        flush_bin: int,
        dt: float,
        times: torch.Tensor,
        dir: torch.Tensor,
        decoy: bool = False,
        replace: bool = False,
        bypass: bool = False,
    ):
        self.times = times
        self.dir = dir
        self._flush_bin = flush_bin
        self.dt = dt
        self.replace = replace
        self.bypass = bypass
        self.decoy = decoy
        self.flushed = False
        self._up_delay_applied = False
        self._down_delay_applied = False

    @property
    def flush_bin(self) -> int:
        return self._flush_bin

    @property
    def up_delay_applied(self) -> bool:
        return self._up_delay_applied

    @property
    def down_delay_applied(self) -> bool:
        return self._down_delay_applied

    def reset_delay(self) -> None:
        self._up_delay_applied = False
        self._down_delay_applied = False

    def _delay(self, direction: int, time_bin: int) -> None:
        """Delay packets in the given direction by duration_s seconds."""
        if direction != self.dir:
            return
        if self.flushed:
            raise RuntimeError("Cannot delay flushed buffer")

        if self._flush_bin == time_bin:
            self._flush_bin = time_bin + 1
            self.times += self.dt
        elif self._flush_bin < time_bin:
            raise RuntimeError(
                f"Cannot delay buffer with flush_bin {self._flush_bin} in the past"
            )
        else:
            pass

    def delay_up(self, time_bin: int) -> None:
        if self.up_delay_applied:
            raise RuntimeError("Up delay already applied")
        self._up_delay_applied = True
        self._delay(UPLOAD, time_bin)

    def delay_down(self, time_bin: int) -> None:
        if self.down_delay_applied:
            raise RuntimeError("Down delay already applied")
        self._down_delay_applied = True
        self._delay(DOWNLOAD, time_bin)

    def subtract(self, dirs: torch.Tensor) -> None:
        if not self.replace:
            return
        if not self.decoy:
            return
        if len(self.times) == 0:
            return

        mydir = self.dir

        # Decoy buffer with replace enabled -> subtract matching queued normal packets.
        npackets = len(self.times)
        if (to_sub := min(npackets, (dirs == mydir).sum())) <= 0:
            return

        self.times = self.times[:-to_sub]

    def flush(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Flush the buffer and return (times, dirs)."""
        if self.flushed:
            raise RuntimeError("Buffer already flushed")
        self.flushed = True
        decoy = torch.full_like(self.times, self.decoy)
        dirs = torch.full_like(self.times, self.dir)
        return self.times, dirs, decoy


@dataclass
class DelayState:
    """Active delay window for one direction."""

    steps_left: int = 0
    steps_since_start: int = 0
    bypass: bool = False
    replace: bool = False
    rtt: int = 1

    @property
    def active(self) -> bool:
        return self.steps_left > 0

    def update(self, action: ActDelay) -> None:
        """Start or extend the active delay window.

        `replace=True` overwrites the current state. If a delay is already
        active and `replace=False`, the active delay must remain unchanged.
        """

        if self.active and not action.replace:
            return

        if self.active and not self.replace:
            return

        if self.active and self.replace and action.replace:
            # We cannot update self.steps_since_start. The delay continue.
            pass

        if not self.active:
            self.steps_since_start = 0

        self.steps_left = max(0, int(action.steps))
        self.bypass = action.bypass
        self.replace = action.replace

    def step(self, buffers: list[SendBuffer], time_bin: int, direction: int) -> None:
        """Apply one delay step to eligible send buffers."""

        if not self.active:
            return

        for buf in buffers:
            # If both the active delay and the buffer are bypass-enabled, leave it
            # untouched. If self.bypass is False, the delay applies to everything.
            if buf.bypass and self.bypass:
                continue
            if direction == UPLOAD:
                if (buf.dir == UPLOAD) and (not buf.up_delay_applied):
                    buf.delay_up(time_bin)

                # The delay applies to packets in the opposite direction after the
                # RTT has passed.
                if (
                    (buf.dir == DOWNLOAD)
                    and (self.steps_since_start >= self.rtt)
                    and (not buf.down_delay_applied)
                ):
                    buf.delay_down(time_bin)

            elif direction == DOWNLOAD:
                if (buf.dir == DOWNLOAD) and (not buf.down_delay_applied):
                    buf.delay_down(time_bin)

                # The delay applies to packets in the opposite direction after the
                # RTT has passed.
                if (
                    (buf.dir == UPLOAD)
                    and (self.steps_since_start >= self.rtt)
                    and (not buf.up_delay_applied)
                ):
                    buf.delay_up(time_bin)
            else:
                raise KeyError(f"Invalid direction {direction}")

        self.steps_left -= 1
        self.steps_since_start += 1


def _get_from_interval(
    times: torch.Tensor,
    dirs: torch.Tensor,
    start_s: float,
    end_s: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Get packets in the interval [start_s, end_s)."""
    mask = (times >= start_s) & (times < end_s) & (dirs != 0)
    return times[mask], dirs[mask]


class TraceStateCursor:
    """Per-trace cursor producing the same bin sequence as get_window_feature_dict.

    It iterates over packet bins and inserts silence bins in gaps using the same
    K-step rule as _add_actions_to_silence_periods().
    """

    def __init__(
        self,
        times: torch.Tensor,
        dirs: torch.Tensor,
        decoy: torch.Tensor,
        dt: float,
        rtt_bins: int = 0,
        add_tail_s: float | None = None,
    ):
        self.dt = dt
        self.times = times
        self.dirs = dirs
        self.decoy = decoy
        # `send_buffers` collect packets scheduled during the current cursor walk.
        # They hold both original packets from the active window and decoy packets
        # injected by the current action, then flush once their target bin is reached.
        self.send_buffers: list[SendBuffer] = []
        self.delay_up = DelayState(rtt=rtt_bins)
        self.delay_down = DelayState(rtt=rtt_bins)
        self.cursor_time_bin: int = 0
        self.prev_time_bin: int = 0
        if add_tail_s is None:
            self.terminate_after_s = times.max().item() + dt
        elif isinstance(add_tail_s, int | float):
            self.terminate_after_s = times.max().item() + add_tail_s
        self.device = times.device

    def step(
        self, actions: StepAction
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
        if self.prev_time_bin * self.dt >= self.terminate_after_s:
            raise StopIteration

        assert actions.time_bin == self.cursor_time_bin, (
            f"Expected action time {self.cursor_time_bin}, got {actions.time_bin}"
        )

        times, dirs = _get_from_interval(
            self.times,
            self.dirs,
            float(self.cursor_time_bin) * self.dt,
            float(self.cursor_time_bin + 1) * self.dt,
        )
        # We add one buffer for both directions for better delay control.
        for dir_ in (UPLOAD, DOWNLOAD):
            mask = dirs == dir_
            if not mask.any():
                continue

            self.send_buffers.append(
                SendBuffer(
                    flush_bin=self.cursor_time_bin,
                    dt=self.dt,
                    times=times[mask],
                    dir=dir_,
                    decoy=False,
                )
            )
        # Action logic here.
        # DelayTraffic semantics to mirror later implementation:
        # - delay is framework-scoped and affects all outgoing traffic
        # - bypass=True means only bypass-enabled decoy traffic may skip the delay
        # - replace=True replaces the active delay; otherwise keep the strongest
        #   active delay (max duration and max packet count independently)
        # - when a delay is started or adjusted, its bypassable status is replaced too
        # DECOY follows the packet through the buffers so the final reconstructed
        # trace can distinguish original packets from injected decoys.
        act_time_bin = actions.time_bin
        dt_bins = self.cursor_time_bin - self.prev_time_bin

        # Send down actions:
        if Actions.SEND_DOWN in actions:
            act_s_d = actions[Actions.SEND_DOWN]
            times = torch.full(
                (act_s_d.count,),
                (act_time_bin + act_s_d.after_steps + 0.5) * self.dt,
                device=self.device,
            )
            sdb = SendBuffer(
                act_time_bin + act_s_d.after_steps,
                self.dt,
                times,
                DOWNLOAD,
                True,
                replace=act_s_d.replace,
                bypass=act_s_d.bypass,
            )

            self.send_buffers.append(sdb)

        # Send up actions:
        if Actions.SEND_UP in actions:
            act_s_u = actions[Actions.SEND_UP]
            times = torch.full(
                (act_s_u.count,),
                (act_time_bin + act_s_u.after_steps + 0.5) * self.dt,
                device=self.device,
            )
            sdb = SendBuffer(
                act_time_bin + act_s_u.after_steps,
                self.dt,
                times,
                UPLOAD,
                True,
                replace=act_s_u.replace,
                bypass=act_s_u.bypass,
            )

            self.send_buffers.append(sdb)

        if Actions.DELAY_DOWN in actions:
            self.delay_down.update(actions[Actions.DELAY_DOWN])

        if Actions.DELAY_UP in actions:
            self.delay_up.update(actions[Actions.DELAY_UP])

        # Apply delays:
        self.delay_down.step(self.send_buffers, self.cursor_time_bin, DOWNLOAD)
        self.delay_up.step(self.send_buffers, self.cursor_time_bin, UPLOAD)

        # Then flush buffers and collect.
        times_l: list[torch.Tensor] = []
        dirs_l: list[torch.Tensor] = []
        decoy_l: list[torch.Tensor] = []

        # First apply the normal packets
        for buf in self.send_buffers:
            # Restore delay bookkeeping.
            buf.reset_delay()

            if buf.decoy:
                continue

            if buf.flush_bin < self.cursor_time_bin:
                raise RuntimeError(
                    f"Buffer flush_bin {buf.flush_bin} is in the past (cursor_time_bin={self.cursor_time_bin})"
                )
            elif buf.flush_bin == self.cursor_time_bin:
                times, dirs, decoy = buf.flush()
                times_l.append(times)
                dirs_l.append(dirs)
                decoy_l.append(decoy)

                # For all scheduled future decoy buffers with replace=True,
                # subtract the flushed packets if possible.
                for future_buf in self.send_buffers:
                    if future_buf.decoy and future_buf.replace:
                        future_buf.subtract(dirs)

        # Then the decoy ones:
        for buf in self.send_buffers:
            if not buf.decoy:
                continue

            if buf.flush_bin < self.cursor_time_bin:
                raise RuntimeError(
                    f"Buffer flush_bin {buf.flush_bin} is in the past (cursor_time_bin={self.cursor_time_bin})"
                )
            if buf.flush_bin == self.cursor_time_bin:
                times, dirs, decoy = buf.flush()
                times_l.append(times)
                dirs_l.append(dirs)
                decoy_l.append(decoy)

        self.send_buffers = [buf for buf in self.send_buffers if not buf.flushed]

        self.prev_time_bin = self.cursor_time_bin
        self.cursor_time_bin += 1

        times = torch.cat(times_l) if times_l else torch.Tensor([])
        dirs = torch.cat(dirs_l) if dirs_l else torch.Tensor([])
        decoy = torch.cat(decoy_l) if decoy_l else torch.Tensor([])

        return times, dirs, decoy, self.cursor_time_bin, dt_bins


class WindowFeatureStreamer:
    """Stream action-window features (B, 1) step-by-step.

    This matches get_window_feature_dict() for the feature set used by RNNDefenceAgent.
    """

    def __init__(
        self,
        X: dict[Feats, torch.Tensor],
        dt: float,
        max_silence_s: float,
        features: list[Feats],
        rtt_bins: int = 0,
        add_tail_s: torch.Tensor | float | None = None,
    ):
        if dt <= 0:
            raise ValueError(f"dt must be > 0, got {dt}")

        ratio = max_silence_s / dt
        # K in bin index space.
        self.max_silence_bins = int(ratio)
        if abs(ratio - round(ratio)) > 1e-8:
            raise ValueError(
                f"max_silence_s must be divisible by dt "
                f"(max_silence_s={max_silence_s}, dt={dt})."
            )

        if set(X.keys()) > {Feats.DECOY, Feats.DIRS, Feats.TIMES}:
            raise ValueError("Invalid set of feats")

        self.features = features
        self.device = X[Feats.TIMES].device
        self.bs = X[Feats.TIMES].shape[0]
        self.dt = float(dt)
        self.X = {k: v.clone() for k, v in X.items()}

        self._cursors: list[TraceStateCursor] = []

        if add_tail_s is None:
            pass
        elif isinstance(add_tail_s, (int, float)):
            add_tail_s = np.ones(self.bs, dtype=np.float64) * add_tail_s
        elif isinstance(add_tail_s, torch.Tensor):
            if add_tail_s.shape != (self.bs,):
                raise ValueError(
                    f"cut_off_time_s must have shape ({self.bs},), got {add_tail_s.shape}"
                )
        else:
            raise ValueError(f"Invalid type for cut_off_time_s: {type(add_tail_s)}")

        for i in range(self.bs):
            self._cursors.append(
                TraceStateCursor(
                    times=self.X[Feats.TIMES][i],
                    dirs=self.X[Feats.DIRS][i],
                    decoy=self.X[Feats.DECOY][i],
                    dt=dt,
                    rtt_bins=rtt_bins,
                    add_tail_s=add_tail_s[i].item() if add_tail_s is not None else None,
                )
            )

        self.done = np.zeros(self.bs, dtype=bool)

    def active_mask(self) -> torch.Tensor:
        return ~self.done

    def step(
        self,
        actions: StepActions,
    ) -> tuple[
        dict[Feats, torch.Tensor], dict[Feats, list[torch.Tensor]], torch.Tensor
    ]:
        bs = self.bs
        device = self.device

        # Emit int bins (not float times)
        time_bins = torch.full(
            (bs, 1), -1, device=device, dtype=torch.long
        )  # -1 = invalid
        up = torch.zeros((bs, 1), device=device, dtype=torch.long)
        down = torch.zeros((bs, 1), device=device, dtype=torch.long)
        dt_bins = torch.full((bs, 1), 0, device=device, dtype=torch.long)  # bin count

        times_l: list[torch.Tensor] = []
        dirs_l: list[torch.Tensor] = []
        decoy_l: list[torch.Tensor] = []

        active_idxs = np.arange(bs)[self.active_mask()]

        def _step_one(aidx: int, cur_action: StepAction) -> StepRes:
            cursor = self._cursors[aidx]
            stepped_bins = 1
            current_bin = cursor.cursor_time_bin
            while True:
                try:
                    w_times, w_dirs, w_decoy, _, _ = cursor.step(cur_action)
                except StopIteration:
                    empty = torch.Tensor([])
                    return StepRes(
                        aidx=aidx,
                        is_done=True,
                        w_times=empty,
                        w_dirs=empty,
                        w_decoy=empty,
                        current_bin=current_bin,
                        stepped_bins=stepped_bins,
                    )

                if w_times.numel() > 0 or stepped_bins >= self.max_silence_bins:
                    return StepRes(
                        aidx=aidx,
                        is_done=False,
                        w_times=w_times,
                        w_dirs=w_dirs,
                        w_decoy=w_decoy,
                        current_bin=current_bin,
                        stepped_bins=stepped_bins,
                    )

                stepped_bins += 1
                cur_action = NoAction(time=cursor.cursor_time_bin)

        # Do the stepping on the active traces. TODO: parallelize??
        results = [_step_one(idx_, a) for idx_, a in zip(active_idxs, actions)]

        for res in results:
            if res.is_done:
                self.done[res.aidx] = True
                continue

            times_l.append(res.w_times)
            dirs_l.append(res.w_dirs)
            decoy_l.append(res.w_decoy)

            up[res.aidx, 0] = ((res.w_dirs == UPLOAD) & (res.w_decoy == 0)).sum()
            down[res.aidx, 0] = ((res.w_dirs == DOWNLOAD) & (res.w_decoy == 0)).sum()
            time_bins[res.aidx, 0] = res.current_bin
            dt_bins[res.aidx, 0] = res.stepped_bins

        fd: dict[Feats, torch.Tensor] = {
            Feats.UP_COUNT: up,
            Feats.DOWN_COUNT: down,
            Feats.Dt_BINS: dt_bins,
            Feats.Dt: dt_bins.to(torch.float64) * self.dt,
            Feats.TIME_BINS: time_bins,
        }

        if Feats.SILENCE_FLAG in self.features:
            fd[Feats.SILENCE_FLAG] = ((up == 0) & (down == 0)).float()

        if not all(f in fd for f in self.features):
            raise ValueError("Some requested features are missing!")

        # If time_bins < 0, this row is done and the values are invalid.
        # Mask out just in case.
        terminated = (time_bins < 0).squeeze(1)
        return (
            {f: fd[f] for f in self.features},
            {Feats.TIMES: times_l, Feats.DIRS: dirs_l, Feats.DECOY: decoy_l},
            ~terminated,
        )
