import torch
from kipl_ml.logging.logger import get_logger

logger = get_logger(__name__)


def get_device() -> torch.DeviceObjType:
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    return device
