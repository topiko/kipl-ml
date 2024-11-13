import argparse
import os

import hydra
import src
import torch
from kipl_ml.data.wf_dataset import WFDataset
from kipl_ml.flow.transforms import FeatureTrs
from kipl_ml.logging.logger import get_logger
from kipl_ml.metrics.clf_metrics import Accuracy, CrossEntropyLoss
from kipl_ml.train.loops import train_model
from omegaconf import DictConfig
from src.cls_cvt import ConvolutionalVisionTransformer
from src.transdfnet import DFNet
from torch import nn
from torch.utils.data import DataLoader

logger = get_logger(__name__)

WORKING_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_DIR_PATH = os.path.join(WORKING_DIR, "config")


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


def get_model(model_name: str, n_classes: int, inputs: dict[str, int]) -> nn.Module:

    if len(set(inputs.values())) != 1:
        raise ValueError("All inputs must have the same size.")

    input_size = inputs[list(inputs.keys())[0]]

    if model_name == "dfnet":
        return WrapDFNet(
            num_classes=n_classes, input_channels=len(inputs), input_size=input_size
        )
    if model_name == "cvt":
        return CNNVisTransformer(
            num_classes=n_classes, in_chans=len(inputs), input_size=input_size
        )

    raise NotImplementedError("Model not implemented yet.")


@hydra.main(config_path=CONFIG_DIR_PATH, config_name="config", version_base=None)
def main(cfg: DictConfig):
    dataset = cfg.dataset.name
    model = cfg.model.name
    n_packets = cfg.flow.n_packets

    feature_trs = FeatureTrs(feature_names=cfg.features)

    trainset = WFDataset(dataset, feature_trs=feature_trs, n_packets=n_packets)

    model = get_model(
        model,
        n_classes=trainset.n_classes,
        inputs=trainset.outputs,
    )

    train_loader = DataLoader(trainset, batch_size=cfg.train.batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters())
    loss_fn = torch.nn.CrossEntropyLoss()
    metrics = [Accuracy(), CrossEntropyLoss()]
    early_stop_metric = "accuracy"
    patience = 2

    train_model(
        model=model,
        train_loader=train_loader,
        valid_loader=train_loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        metrics=metrics,
        early_stop_metric=early_stop_metric,
        patience=patience,
    )


if __name__ == "__main__":
    argp = argparse.ArgumentParser()

    argp.add_argument("--model", type=str, help="Model name")
    args = argp.parse_args()

    main()
