import argparse
import os

import hydra
import src
import torch
from kipl_ml.data.wf_dataset import WFDataset
from kipl_ml.flow.transforms import parse_transforms
from kipl_ml.logging.logger import get_logger
from kipl_ml.metrics.clf_metrics import Accuracy, CrossEntropyLoss
from kipl_ml.train.loops import train_model
from omegaconf import DictConfig
from torch import nn
from torch.utils.data import DataLoader

logger = get_logger(__name__)

WORKING_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_DIR_PATH = os.path.join(WORKING_DIR, "config")


def get_model(
    model_name: str, n_classes: int, inputs: list[tuple[str, int]]
) -> nn.Module:
    if model_name == "dfnet":
        from src.transdfnet import DFNet

        if len({i[1] for i in inputs}) != 1:
            raise ValueError("All inputs must have the same size.")

        input_size = inputs[0][1]

        return DFNet(n_classes, len(inputs), input_size=input_size)
    else:
        raise NotImplementedError("Model not implemented yet.")


@hydra.main(config_path=CONFIG_DIR_PATH, config_name="config", version_base=None)
def main(cfg: DictConfig):
    dataset = cfg.dataset.name
    model = cfg.model.name
    transform = cfg.transform.list
    n_packets = cfg.flow.n_packets

    transforms = parse_transforms(transform)

    trainset = WFDataset(dataset, n_packets=n_packets, transform=transforms)

    model = get_model(
        model,
        n_classes=trainset.n_classes,
        inputs=trainset.outputs,
    )

    train_loader = DataLoader(trainset, batch_size=cfg.train.batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters())
    loss_fn = torch.nn.CrossEntropyLoss()
    metrics = [Accuracy(), CrossEntropyLoss()]
    early_stopping = "loss"
    patience = 2

    train_model(
        model=model,
        train_loader=train_loader,
        valid_loader=train_loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        metrics=metrics,
        early_stopping=early_stopping,
        patience=patience,
    )


if __name__ == "__main__":
    argp = argparse.ArgumentParser()

    argp.add_argument("--model", type=str, help="Model name")
    args = argp.parse_args()

    main()
