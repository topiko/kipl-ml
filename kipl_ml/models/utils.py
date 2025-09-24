import json
import os

import laserbeak.model_configs as lasereak_configs
import torch
from mlflow.models import ModelSignature, infer_signature
from torch import nn

from kipl_ml.data.wf_dataset import WFDataset
from kipl_ml.logging.logger import get_logger

logger = get_logger(__name__)


def count_parameters(model) -> str:
    np = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    return f"{np:.02f} M"


def unsqueeze_batch(x: dict[str, torch.tensor]) -> dict[str, torch.tensor]:

    X_ = {k: v.unsqueeze(0) for k, v in x.items()}
    return X_


def get_laserbeak_model_config(model_name: str) -> dict:
    if model_name == "laserbeak_wo_attention":
        model_name = "tuned-multi"
        logger.info("Mapped name 'laserbeak_wo_attention' into 'tuned-multi'")

    config_path = os.path.join(list(lasereak_configs.__path__)[0], model_name + ".json")
    with open(config_path, "r") as fi:
        model_config = json.load(fi)

    return model_config


def get_signature(model: nn.Module, ds: WFDataset) -> ModelSignature:

    model.eval()
    with torch.no_grad():
        X_ = model.example_input(ds[0][0])
        y_ = model(X_).detach().cpu().numpy()

    inputs = {k: v.detach().cpu().numpy() for k, v in X_.items()}
    signature = infer_signature(inputs, y_)

    return signature
