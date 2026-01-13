from __future__ import annotations

import multiprocessing

import torch
from torch import nn
from torch.utils.data import DataLoader

from kipl_ml.data.utils import DOWNLOAD, UPLOAD
from kipl_ml.data.wf_dataset import WFDataset
from kipl_ml.trace.features import Feats


def collate_fn_(
    batch: list[tuple[dict[Feats, torch.Tensor], torch.Tensor]], seq_len: int = 300
) -> tuple[dict[Feats, torch.Tensor], torch.Tensor]:
    bs = len(batch)
    features = batch[0][0].keys()
    X = {f: torch.zeros((bs, seq_len), dtype=torch.float) for f in features}
    y = torch.zeros((bs,), dtype=torch.long)
    for i, (x_, y_) in enumerate(batch):
        for f in features:
            xtmp = x_[f][: seq_len + 1]
            if f == Feats.BURST_DIRS:
                xtmp += 1

            if len(xtmp) < seq_len + 1:
                lenx = len(xtmp)
            else:
                lenx = len(xtmp) - 1

            X[f][i, :lenx] = xtmp[:lenx]
        y[i] = y_

    return X, y


def dl_(
    ds: WFDataset,
    bs: int,
    collate_fn: callable | None,
    shuffle: bool = False,
    nworkers: int | None = None,
    **kwargs,
) -> DataLoader:
    if nworkers is None:
        nworkers = multiprocessing.cpu_count() // 8 * 7

    return DataLoader(
        ds,
        batch_size=bs,
        shuffle=shuffle,
        num_workers=nworkers,
        collate_fn=collate_fn,
        **kwargs,
    )


class Burst2Packets:
    def __init__(self, obs: nn.Module | None, trace_len: int):
        self.obs = obs
        self.trace_len = trace_len

    def to(self, device: torch.DeviceObjType):
        if self.obs is None:
            return
        self.obs.to(device)

    def __call__(self, X: dict[Feats, torch.Tensor]) -> dict[Feats, torch.Tensor]:
        if self.obs is not None:
            X = self.obs(X)

        # (B, L = nbursts)
        blens = X[Feats.BURST_LENS].round().long()
        B, BL = blens.shape

        # (B, BL + 1)
        bedges = torch.cat(
            (
                torch.zeros((len(blens), 1), device=blens.device).long(),
                blens.cumsum(dim=1),
            ),
            dim=1,
        )

        # Trace len
        TL = max(bedges[:, -1].max(), self.trace_len)

        # (B, TL)
        dirs = torch.zeros((B, TL), device=blens.device)

        # (B, BL)
        starts = bedges[:, :-1]
        ends = bedges[:, 1:]

        # (1, TL)
        idxs = torch.arange(TL, device=blens.device).view(1, TL)

        for bidx in range(BL):
            # (B, 1)
            starts_ = starts[:, bidx].view(B, 1)
            ends_ = ends[:, bidx].view(B, 1)

            dir_ = DOWNLOAD if (bidx % 2) == 0 else UPLOAD

            # (B, TL)
            mask = (idxs >= starts_) & (idxs < ends_)

            # (B, TL)
            dirs[mask] = dir_

        dirs = dirs[:, : self.trace_len]

        return {Feats.DIRS: dirs}
