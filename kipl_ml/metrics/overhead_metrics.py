from typing import ClassVar

import torch

from kipl_ml.metrics.metrics import GeneralMetric, Objective
from kipl_ml.trace.enums import Feats


class _BurstOverheadLoss(GeneralMetric):
    OBJECTIVE: ClassVar[str] = Objective.MIN

    def __init__(self, feat: Feats, reduction: str = "mean"):
        self.feat = feat

        if reduction not in {"mean", "sum"}:
            raise KeyError("'reduction' can only be 'mean' or 'sum'.")

        self.reduction = reduction

    def __call__(
        self, xobs: dict[Feats, torch.Tensor], x: dict[Feats, torch.Tensor]
    ) -> torch.Tensor:
        # absolute amount after obsfuscation:
        # (N, L)
        abs_extra = xobs[self.feat] - x[self.feat]

        # for each trace:
        abs_extra = abs_extra.sum(dim=1)
        orig = x[self.feat].sum(dim=1)

        loss = abs_extra / orig
        if self.reduction == "mean":
            return loss.mean()

        return loss.sum()


class BurstLenOverhead(_BurstOverheadLoss):
    def __init__(self, reduction: str = "mean"):
        super().__init__(feat=Feats.BURST_LENS, reduction=reduction)


class BurstDurOverhead(_BurstOverheadLoss):
    def __init__(self, reduction: str = "mean"):
        super().__init__(feat=Feats.BURST_DURS, reduction=reduction)


class BurstRelDurOverhead(_BurstOverheadLoss):
    def __init__(self, reduction: str = "mean"):
        super().__init__(feat=Feats.BURST_RELDURS, reduction=reduction)
