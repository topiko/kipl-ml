from __future__ import annotations

import torch

from kipl_ml.data.utils import DOWNLOAD, UPLOAD
from kipl_ml.logging.logger import get_logger
from kipl_ml.rl.enums import Actions
from kipl_ml.rl.utils import _append_values, _push_left
from kipl_ml.trace.enums import Feats

logger = get_logger(__name__)


class DelayActionExec:
    def __init__(self, default_delay: float):
        self.default_delay: float = default_delay
        self._delay_times: torch.Tensor
        self._update_times: torch.Tensor
        self._counts: torch.Tensor
        self.t: float = 0.0

    def step(self, dt: float, feat_dict: dict[Feats, torch.Tensor]):
        self._feature_dict: dict[Feats, torch.Tensor] = feat_dict
        self.t += dt

    def update(self, counts: torch.Tensor, delays: torch.Tensor | None = None):
        if self.t == 0:
            self._init_tensors(counts)

        mask = counts != 0

        self._counts[mask] = counts[mask]
        self._update_times[mask] = self.t

        if delays is not None:
            self._delay_times[mask] = delays[mask]
        else:
            self._delay_times[mask] = self.default_delay

    @property
    def feature_dict(self) -> dict[Feats, torch.Tensor]:
        return self._feature_dict

    def _init_tensors(self, mask: torch.Tensor):
        device = mask.device
        self._counts = torch.zeros_like(mask, dtype=torch.float, device=device)
        self._update_times = torch.zeros_like(mask, dtype=torch.float, device=device)
        self._delay_times = torch.zeros_like(mask, dtype=torch.float, device=device)


class SendActionExec:
    def __init__(self, dir_: int, default_dur: float):
        self.dir_ = dir_
        self.default_duration: float = default_dur
        self.t = 0.0

        self._counts: torch.Tensor
        self._update_times: torch.Tensor
        self._decaytimes: torch.Tensor
        self._X = torch.Tensor
        self._feature_dict: dict[Feats, torch.Tensor]

    def step(self, dt: float, curXobs: dict[Feats, torch.Tensor]):
        if self._counts is None:
            raise ValueError("No counts set. Call update() before step().")

        # (B, ) we sample from uniform..
        density_p_send = torch.where(
            self._counts != 0, self._counts / self._decaytimes, 0.0
        )

        resolution = 1e-6  # microsecond

        # Total number of time steps [unitless]
        T = dt / resolution
        # (B, T)
        times = (
            torch.arange(0, T, step=1.0, device=self._counts.device)
            .unsqueeze(0)
            .expand(self._counts.shape[0], -1)
        )

        # (B, T)
        mask = torch.rand_like(times) < resolution * density_p_send.unsqueeze(1)

        # Padding and normal
        # (B, )
        send_normal = (curXobs[Feats.DIRS] == self.dir_).sum(dim=1)

        # (1, )
        npackets = (mask.sum(dim=1) + send_normal).max()
        npad_packets = mask.sum(dim=1).max()

        # (B, npackets)
        send_times = torch.zeros(
            (self._counts.shape[0], npackets), device=self._counts.device
        )
        packets = torch.zeros_like(send_times)
        padding = torch.zeros_like(send_times)

        if npackets > 0:
            if npad_packets > 0:
                # (B, npackets)
                send_times = _push_left(times, mask)[:, :npackets] * resolution
                print(send_times)
                breakpoint()

                # (B, npackets) padding mask
                padding[send_times != 0] = 1.0

                # (B, npackets)
                packets[send_times != 0] = self.dir_

            # (B, npackets) observed packets to send
            send_times = _append_values(
                send_times,
                _push_left(
                    curXobs[Feats.TIMES],
                    curXobs[Feats.DIRS] == self.dir_,
                ),
                on_short_base="cat",
            )[:, :npackets]

            # (B, npackets) sampled packets to send
            packets = _append_values(
                packets,
                _push_left(curXobs[Feats.DIRS], curXobs[Feats.DIRS] == self.dir_),
                on_short_base="cat",
            )[:, :npackets]

            # We still need to sort everything according to time:
            send_times, sort_idx = torch.sort(send_times, dim=1)
            packets = torch.gather(packets, dim=1, index=sort_idx)
            padding = torch.gather(padding, dim=1, index=sort_idx)
            print()
            print("self t", self.t)
            print("send_times", send_times)
            print(curXobs[Feats.TIMES])
            breakpoint()

        self._feature_dict = {
            Feats.DIRS: packets[:, :npackets],
            Feats.TIMES: send_times[:, :npackets],
            Feats.PADDING: padding[:, :npackets],
        }
        self.t += dt

    @property
    def feature_dict(self) -> dict[Feats, torch.Tensor]:
        return self._feature_dict

    def update(
        self,
        counts: torch.Tensor,
        decaytimes: torch.Tensor | None = None,
    ):
        if self.t == 0:
            self._init_tensors(counts)

        mask = counts != 0
        self._counts[mask] = counts[mask]
        self._update_times[mask] = self.t

        if decaytimes is not None:
            self._decaytimes[mask] = decaytimes[mask]
        else:
            self._decaytimes[mask] = self.default_duration

        # (B, )
        time_passed = self.t - self._update_times

        # (B, ) reset counts where time passed > decay_time
        reset_mask = time_passed > self._decaytimes

        self._counts[reset_mask] = 0
        self._decaytimes[reset_mask] = self.default_duration
        self._update_times[mask] = self.t

    def _init_tensors(self, counts: torch.Tensor):
        device = counts.device
        self._counts = torch.zeros_like(counts, dtype=torch.float, device=device)
        self._update_times = torch.zeros_like(counts, dtype=torch.float, device=device)
        self._decaytimes = torch.zeros_like(counts, dtype=torch.float, device=device)


class ActionsExec:
    def __init__(self):
        self.send_ackts_up = SendActionExec(dir_=UPLOAD, default_dur=1.0)
        self.send_ackts_down = SendActionExec(dir_=DOWNLOAD, default_dur=1.0)
        self.delay_ackts = DelayActionExec(default_delay=1.0)

    def step(
        self, dt: float, actions: torch.Tensor, curXobs: dict[Feats, torch.Tensor]
    ):
        # Update the action executors
        self.send_ackts_down.update(
            counts=actions[Actions.SEND_COUNT_DOWN],
            decaytimes=actions[Actions.SEND_TIME_DOWN],
        )
        self.send_ackts_up.update(
            counts=actions[Actions.SEND_COUNT_UP],
            decaytimes=actions[Actions.SEND_TIME_UP],
        )

        self.send_ackts_up.step(dt, curXobs)
        self.send_ackts_down.step(dt, curXobs)

        fd_u = self.send_ackts_up.feature_dict
        fd_d = self.send_ackts_down.feature_dict

        breakpoint()
        feature_dict: dict[Feats, torch.Tensor] = {
            k: torch.cat([fd_u[k], fd_d[k]], dim=1) for k in fd_u
        }

        sorted_times, indices = torch.sort(feature_dict[Feats.TIMES], dim=1)
        feature_dict[Feats.TIMES] = sorted_times
        feature_dict[Feats.DIRS] = feature_dict[Feats.DIRS][indices]
        feature_dict[Feats.PADDING] = feature_dict[Feats.PADDING][indices]

        # self.delay_ackts.update()
        self.delay_ackts.step(dt, feature_dict)

        return self.delay_ackts.feature_dict
