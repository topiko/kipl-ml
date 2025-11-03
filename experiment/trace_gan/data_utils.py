from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from kipl_ml.data.wf_dataset import WFDataset
from kipl_ml.trace.features import Feats


def collate_fn_(
    batch: list[tuple[dict[Feats, torch.Tensor], torch.Tensor]], seq_len: int = 300
) -> tuple[dict[Feats, torch.Tensor], torch.Tensor]:
    bs = len(batch)
    features = batch[0][0].keys()
    X = {f: torch.zeros((bs, seq_len), dtype=torch.float) for f in features}
    y = torch.zeros((bs,), dtype=torch.long)
    start_idx = torch.randint(0, 5, (bs,))
    for i, (x_, y_) in enumerate(batch):
        sidx = start_idx[i]
        for f in features:
            xtmp = x_[f][sidx : sidx + seq_len + 1]
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
    ds: WFDataset, bs: int, collate_fn: callable, shuffle: bool = False
) -> DataLoader:
    return DataLoader(
        ds,
        batch_size=bs,
        shuffle=shuffle,
        num_workers=30,
        collate_fn=collate_fn,
    )
