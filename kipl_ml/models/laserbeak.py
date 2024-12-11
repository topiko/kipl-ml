"""
Wrappers for laserbeak models.
"""

import torch
from kipl_ml.data.wf_dataset import WFDataset
from kipl_ml.logging.logger import get_logger
from kipl_ml.models.utils import count_parameters
from mlflow.models import infer_signature
from mlflow.models.signature import ModelSignature
from src.cls_cvt import ConvolutionalVisionTransformer
from src.transdfnet import DFNet
from torch import nn

logger = get_logger(__name__)


class WrapDFNet(DFNet):

    def example_input(self, X: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return get_example_input(X)

    def forward(
        self,
        x: dict[str, torch.Tensor],
        sample_sizes=None,
        return_feats=False,
        *args,
        **kwargs,
    ):
        x_ = torch.cat([x_.unsqueeze(1) for x_ in x.values()], dim=1)

        return super().forward(x_, sample_sizes, return_feats, *args, **kwargs)


class CNNVisTransformer(ConvolutionalVisionTransformer):

    def example_input(self, X: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return get_example_input(X)

    def forward(self, x: torch.Tensor):
        x = torch.cat([x_.unsqueeze(1) for x_ in x.values()], dim=1)
        return super().forward(x)


def get_model(
    model_name: str,
    n_classes: int,
    inputs: dict[str, dict[str, int]],
    model_config: dict,
) -> nn.Module:

    input_lens: set[int] = set()
    for input_dict in inputs.values():
        input_lens = input_lens.union(set(input_dict.values()))

    if len(input_lens) != 1:
        raise ValueError("All inputs must have the same size.")

    input_size = next(iter(input_lens))

    if model_config["input_size"] != input_size:
        raise ValueError("Input size mismatch.")

    logger.info(f"Creating model {model_name}...")
    logger.info(f"\tConfig:")
    for k, v in model_config.items():
        logger.info(f"{k:>30}: {v}")

    if model_name.startswith("df") or model_name.startswith("laserbeak"):
        net = WrapDFNet(
            num_classes=n_classes, input_channels=len(inputs), **model_config
        )
        net.name = model_name
        logger.info(f"\t-->{count_parameters(net)} parameters.")

        return net

    if model_name == "cvt":
        net = CNNVisTransformer(
            num_classes=n_classes, in_chans=len(inputs), input_size=input_size
        )
        net.name = model_name
        return net

    raise NotImplementedError("Model not implemented yet.")


def get_example_input(X: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    X_ = {k: v.unsqueeze(0) for k, v in X.items()}
    return X_


def get_signature(
    model: WrapDFNet | CNNVisTransformer, ds: WFDataset
) -> ModelSignature:

    model.eval()
    with torch.no_grad():
        X_ = model.example_input(ds[0][0])
        y_ = model(X_).numpy()

    signature = infer_signature({k: v.numpy() for k, v in X_.items()}, y_)

    return signature
