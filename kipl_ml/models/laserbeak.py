"""
Wrappers for laserbeak models.
"""

import torch
from src.cls_cvt import ConvolutionalVisionTransformer
from src.transdfnet import DFNet
from torch import nn


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


def get_model(
    model_name: str, n_classes: int, inputs: dict[str, dict[str, int]]
) -> nn.Module:

    input_lens: set[int] = set()
    for input_dict in inputs.values():
        input_lens = input_lens.union(set(input_dict.values()))

    if len(input_lens) != 1:
        raise ValueError("All inputs must have the same size.")

    input_size = next(iter(input_lens))

    if model_name == "dfnet":
        return WrapDFNet(
            num_classes=n_classes, input_channels=len(inputs), input_size=input_size
        )
    if model_name == "cvt":
        return CNNVisTransformer(
            num_classes=n_classes, in_chans=len(inputs), input_size=input_size
        )

    raise NotImplementedError("Model not implemented yet.")
