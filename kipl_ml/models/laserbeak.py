"""
Wrappers for laserbeak models.
"""

import torch
from src.cls_cvt import ConvolutionalVisionTransformer
from src.transdfnet import DFNet


class WrapDFNet(DFNet):
    name = "lasereak_dfnet"

    def forward(
        self,
        x: dict[str, torch.Tensor],
        sample_sizes=None,
        return_feats=False,
        *args,
        **kwargs
    ):
        x = torch.cat([x_.unsqueeze(1) for x_ in x.values()], dim=1)

        return super().forward(x, sample_sizes, return_feats, *args, **kwargs)


class CNNVisTransformer(ConvolutionalVisionTransformer):
    name = "lasereak_cvt"

    def forward(self, x: torch.Tensor):
        x = torch.cat([x_.unsqueeze(1) for x_ in x.values()], dim=1)
        return super().forward(x)
