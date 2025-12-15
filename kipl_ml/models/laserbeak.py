"""
Wrappers for laserbeak models.
"""

import torch
from laserbeak.cls_cvt import ConvolutionalVisionTransformer
from laserbeak.transdfnet import DFNet

from kipl_ml.logging.logger import get_logger
from kipl_ml.models.utils import unsqueeze_batch
from kipl_ml.trace.enums import Feats

logger = get_logger(__name__)


class WrapDFNet(DFNet):
    def example_input(self, X: dict[Feats, torch.Tensor]) -> dict[Feats, torch.Tensor]:
        return unsqueeze_batch(X)

    def forward(
        self,
        x: dict[Feats, torch.Tensor],
        sample_sizes=None,
        return_feats=False,
        *args,
        **kwargs,
    ):
        x_ = torch.cat([x_.unsqueeze(1) for x_ in x.values()], dim=1)

        return super().forward(x_, sample_sizes, return_feats, *args, **kwargs)

    def predict(
        self,
        x: dict[Feats, torch.Tensor],
        sample_sizes=None,
        return_feats=False,
        *args,
        **kwargs,
    ):
        x_ = torch.cat([x_.unsqueeze(1) for x_ in x.values()], dim=1)

        logits = self.forward(x_, sample_sizes, return_feats, *args, **kwargs)

        preds = torch.argmax(logits, dim=-1)
        return preds


class CNNVisTransformer(ConvolutionalVisionTransformer):
    def example_input(self, X: dict[Feats, torch.Tensor]) -> dict[Feats, torch.Tensor]:
        return unsqueeze_batch(X)

    def forward(self, x: torch.Tensor):
        x = torch.cat([x_.unsqueeze(1) for x_ in x.values()], dim=1)
        return super().forward(x)
