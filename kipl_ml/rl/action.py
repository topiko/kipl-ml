from __future__ import annotations

import torch

from kipl_ml.data.utils import DOWNLOAD, UPLOAD
from kipl_ml.logging.logger import get_logger
from kipl_ml.rl.enums import Actions
from kipl_ml.rl.utils import _append_values, _flush_left
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


class DelayActionExec:
    def __init__(self, dt: float):
        self._delay_times: torch.Tensor
        self._update_times: torch.Tensor
        self._counts: torch.Tensor
        self.dt: float = dt
        self.t: float = 0.0
        logger.warning("DelayActionExec has dummy implementation.")

    def step(self, curXobs: dict[Feats, torch.Tensor]):
        self._feature_dict: dict[Feats, torch.Tensor] = curXobs
        self.t += self.dt

    def update(self, counts: torch.Tensor, delays: torch.Tensor):
        if self.t == 0:
            self._init_tensors(counts)

        mask = counts != 0

        self._counts[mask] = counts[mask]
        self._update_times[mask] = self.t
        self._delay_times[mask] = delays[mask]

    @property
    def feature_dict(self) -> dict[Feats, torch.Tensor]:
        return self._feature_dict

    def _init_tensors(self, mask: torch.Tensor):
        device = mask.device
        self._counts = torch.zeros_like(mask, dtype=torch.float, device=device)
        self._update_times = torch.zeros_like(mask, dtype=torch.float, device=device)
        self._delay_times = torch.zeros_like(mask, dtype=torch.float, device=device)


class SendActionExec:
    def __init__(self, dir_: int, dt: float):
        self.dir_ = dir_
        self.dt = dt
        self.t = 0.0

        self._counts: torch.Tensor
        self._update_times: torch.Tensor
        self._decaytimes: torch.Tensor
        self._X = torch.Tensor
        self._unsorted_feature_dict: dict[Feats, torch.Tensor]

    def step(self, curXobs: dict[Feats, torch.Tensor]):
        if self._counts is None:
            raise ValueError("No counts set. Call update() before step().")

        # (B, ) we sample from uniform..
        density_p_send = torch.where(
            self._counts != 0, self._counts / self._decaytimes, 0.0
        )

        resolution = 1e-6  # microsecond

        # Total number of time steps [unitless]
        T = self.dt / resolution
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
                send_times = _flush_left(times, mask)[:, :npackets]
                # Return to absolute time
                send_times[send_times != 0] *= resolution
                # Move to current time frame
                send_times[send_times != 0] += self.t

                # (B, npackets) padding mask
                padding[send_times != 0] = 1.0

                # (B, npackets)
                packets[send_times != 0] = self.dir_

            # (B, npackets) observed packets to send
            send_times = _append_values(
                send_times,
                _flush_left(
                    curXobs[Feats.TIMES],
                    curXobs[Feats.DIRS] == self.dir_,
                ),
                on_short_base="cat",
            )[:, :npackets]

            # (B, npackets) sampled packets to send
            packets = _append_values(
                packets,
                _flush_left(curXobs[Feats.DIRS], curXobs[Feats.DIRS] == self.dir_),
                on_short_base="cat",
            )[:, :npackets]

            # We still need to sort everything according to time:
            if (curXobs[Feats.TIMES] > (self.t + self.dt)).any():
                raise ValueError("Trying to send packets beyond current time + dt.")
            if (curXobs[Feats.TIMES][curXobs[Feats.DIRS] != 0] < self.t).any():
                print(self.t)
                print(curXobs[Feats.TIMES])
                breakpoint()
                raise ValueError("Trying to send packets before current time.")

        self._unsorted_feature_dict = {
            Feats.DIRS: packets[:, :npackets],
            Feats.TIMES: send_times[:, :npackets],
            Feats.PADDING: padding[:, :npackets],
        }
        self.t += self.dt

    @property
    def unsorted_feature_dict(self) -> dict[Feats, torch.Tensor]:
        return self._unsorted_feature_dict

    def update(
        self,
        counts: torch.Tensor,
        decaytimes: torch.Tensor,
        update_mask: torch.Tensor,
    ):
        if self.t == 0:
            self._init_tensors(counts)

        mask = counts != 0 & update_mask
        self._counts[mask] = counts[mask]
        self._update_times[mask] = self.t
        self._decaytimes[mask] = decaytimes[mask]

        # (B, )
        time_passed = self.t - self._update_times

        # (B, ) reset counts where time passed > decay_time
        reset_mask = time_passed > self._decaytimes

        self._counts[reset_mask] = 0
        self._decaytimes[reset_mask] = 0
        self._update_times[mask] = self.t

    def _init_tensors(self, counts: torch.Tensor):
        device = counts.device
        self._counts = torch.zeros_like(counts, dtype=torch.float, device=device)
        self._update_times = torch.zeros_like(counts, dtype=torch.float, device=device)
        self._decaytimes = torch.zeros_like(counts, dtype=torch.float, device=device)


class ActionsExec:
    def __init__(self, dt: float):
        self.send_ackts_up = SendActionExec(dir_=UPLOAD, dt=dt)
        self.send_ackts_down = SendActionExec(dir_=DOWNLOAD, dt=dt)
        self.delay_ackts = DelayActionExec(dt=dt)

    def step(
        self, actions: dict[Actions, torch.Tensor], curXobs: dict[Feats, torch.Tensor]
    ):
        # Update the action executors
        self.send_ackts_down.update(
            counts=actions[Actions.SEND_COUNT_DOWN],
            decaytimes=actions[Actions.SEND_TIME_DOWN],
            update_mask=~actions[Actions.WAIT],
        )
        self.send_ackts_up.update(
            counts=actions[Actions.SEND_COUNT_UP],
            decaytimes=actions[Actions.SEND_TIME_UP],
            update_mask=~actions[Actions.WAIT],
        )

        self.send_ackts_up.step(curXobs)
        self.send_ackts_down.step(curXobs)

        fd_u = self.send_ackts_up.unsorted_feature_dict
        fd_d = self.send_ackts_down.unsorted_feature_dict

        unsorted_feature_dict: dict[Feats, torch.Tensor] = {
            k: torch.cat([fd_u[k], fd_d[k]], dim=1) for k in fd_u
        }

        feature_dict = _sort_feature_dict(unsorted_feature_dict)

        # self.delay_ackts.update()
        self.delay_ackts.step(feature_dict)

        return self.delay_ackts.feature_dict
