import torch

from kipl_ml.logging.logger import get_logger

logger = get_logger(__name__)


def tensor_footprint(t: torch.Tensor, include_grad: bool = True) -> str:
    """Calculate the memory footprint of a tensor in bytes.

    Args:
        tensor (torch.Tensor): The tensor to calculate the footprint for.
        include_grad (bool, optional): Whether to include the gradient tensor's
            footprint if it exists. Defaults to True.

    Returns:
        str: The memory footprint in a human-readable format.
    """

    bytes_ = t.element_size() * t.nelement()

    if include_grad and t.grad is not None:
        bytes_ += t.grad.element_size() * t.grad.nelement()

    if bytes_ < 1024:
        return f"{bytes_} B"
    if bytes_ < 1024**2:
        return f"{bytes_ / 1024:.2f} KB"
    if bytes_ < 1024**3:
        return f"{bytes_ / 1024**2:.2f} MB"
    return f"{bytes_ / 1024**3:.2f} GB"


def get_device() -> torch.DeviceObjType:
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    return device
