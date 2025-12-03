from __future__ import annotations

import torch

from kipl_ml.logging.logger import get_logger
from kipl_ml.trace.enums import Feats

logger = get_logger(__name__)


class DiscTraceObservation:
    def __init__(self, B: int, L: int, device: torch.DeviceObjType):
        self.L = L
        self.X = torch.zeros((B, L, 3), device=device)  # dirs, times, padding
        self._obs_list: list[torch.Tensor] = []

    @property
    def times(self) -> torch.Tensor:
        return self.X[..., 1]

    @property
    def dirs(self) -> torch.Tensor:
        return self.X[..., 0]

    @property
    def count_send(self) -> torch.Tensor:
        return (self.X[..., 0] != 0).sum(dim=1)

    @property
    def count_padding(self) -> torch.Tensor:
        return (self.X[..., 2] == 1).sum(dim=1)

    @property
    def is_waiting(self) -> torch.Tensor:
        return self.X[:, 0, 0] == 0

    @property
    def feature_dict(self) -> dict[Feats, torch.Tensor]:
        return {
            Feats.DIRS: self.X[..., 0],
            Feats.TIMES: self.X[..., 1],
            Feats.PADDING: self.X[..., 2],
        }

    def reset(self):
        self.X[...] = 0

    def append(self):
        self._obs_list.append(self.X.clone().detach())

    @property
    def history(self) -> dict[Feats, torch.Tensor]:
        Xobs = torch.cat(self._obs_list, dim=1)
        device = Xobs.device
        bs = Xobs.shape[0]

        mask = Xobs[..., 0] != 0
        Lmax = mask.sum(dim=1).max()

        Xobsd = {}
        for f, i in zip((Feats.DIRS, Feats.TIMES, Feats.PADDING), range(3)):
            lens = mask.sum(dim=1)

            idxs = torch.arange(Lmax, device=device).unsqueeze(0).expand(bs, -1)

            new_mask = idxs < lens.unsqueeze(1)

            # (B, Lmax, 3)
            Xobs_ = torch.zeros((bs, Lmax), device=device)
            Xobs_[new_mask] = Xobs[mask, i]

            Xobsd[f] = Xobs_

        Xobsd[Feats.PADDING] = Xobsd[Feats.PADDING] == 1

        return Xobsd


class BaseTraceObservation:
    def __init__(self, X: dict[Feats, torch.Tensor]):
        self._X = X
        self.t = 0.0

        self.curX: torch.Tensor
        self._obs_list: list[torch.Tensor] = []

    def step(self, dt: float):
        times = self._X[Feats.TIMES]

        mask = (self.t <= times) & (times < self.t + dt)

        max_l = mask.sum(dim=1).max()

        self.curX = torch.zeros((times.shape[0], max_l, 3), device=times.device)

        # (B, M), M = times.shape[1]
        row_idxs = (
            torch.arange(times.shape[0], device=times.device)
            .unsqueeze(1)
            .expand(-1, times.shape[1])
        )
        # (B, M)
        col_idxs = mask.cumsum(dim=1) - 1

        # (mask.sum(), )
        row_idxs = row_idxs[mask]
        col_idxs = col_idxs[mask]

        self.curX[row_idxs, col_idxs, 0] = self._X[Feats.DIRS][mask]
        self.curX[row_idxs, col_idxs, 1] = self._X[Feats.TIMES][mask]
        self.curX[row_idxs, col_idxs, 2] = 0.0  # padding

        self.t += dt

    @property
    def feature_dict(self) -> dict[Feats, torch.Tensor]:
        return {
            Feats.DIRS: self.curX[..., 0],
            Feats.TIMES: self.curX[..., 1],
            Feats.PADDING: self.curX[..., 2],
        }

    def reset(self):
        self.X[...] = 0

    def append(self):
        self._obs_list.append(self.curX.clone().detach())

    @property
    def history(self) -> dict[Feats, torch.Tensor]:
        Xobs = torch.cat(self._obs_list, dim=1)
        device = Xobs.device
        bs = Xobs.shape[0]

        mask = Xobs[..., 0] != 0
        Lmax = mask.sum(dim=1).max()

        Xobsd = {}
        for f, i in zip((Feats.DIRS, Feats.TIMES, Feats.PADDING), range(3)):
            lens = mask.sum(dim=1)

            idxs = torch.arange(Lmax, device=device).unsqueeze(0).expand(bs, -1)

            new_mask = idxs < lens.unsqueeze(1)

            # (B, Lmax, 3)
            Xobs_ = torch.zeros((bs, Lmax), device=device)
            Xobs_[new_mask] = Xobs[mask, i]

            Xobsd[f] = Xobs_

        Xobsd[Feats.PADDING] = Xobsd[Feats.PADDING] == 1

        return Xobsd
