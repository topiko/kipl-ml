import torch
from torch import nn

from kipl_ml.logging.logger import get_logger
from kipl_ml.models.df import DF
from kipl_ml.models.laserbeak import WrapDFNet
from kipl_ml.models.march import March
from kipl_ml.models.rf import RF
from kipl_ml.models.utils import count_parameters
from kipl_ml.trace.enums import Feats

logger = get_logger(__name__)


class _WrapPacketProbsNet(nn.Module):
    def __init__(self, net: nn.Module, learn_kernel: bool = True):
        super().__init__()
        self.net = net
        self.net.features[self.net.features.index(Feats.DIR_PROBS)] = Feats.DIRS
        self.conv = nn.Conv1d(3, 1, kernel_size=1, bias=False)

        self.conv.weight = nn.Parameter(
            torch.tensor([[[1.0, 0.0, -1.0]]]).reshape(1, 3, 1),
            requires_grad=learn_kernel,
        )
        logger.info(f"Model {self.net.name} wrapped to use packet probabilities.")

    def _map(self, x: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        # (B, 3, L)
        dirps = x[Feats.DIR_PROBS].permute(0, 2, 1)
        x[Feats.DIRS] = self.conv(dirps).squeeze(1)

        return x

    def forward(self, x: dict[str, torch.Tensor], *args, **kwargs):
        return self.net(self._map(x), *args, **kwargs)

    def predict(self, x: dict[str, torch.Tensor], *args, **kwargs):
        return self.net.predict(self._map(x), *args, **kwargs)


def get_model(
    source: str,
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

    input_len = input_lens.pop()

    logger.info(f"Creating model {model_name}...")
    logger.info("\tConfig:")
    for k, v in model_config.items():
        logger.info(f"{k:>30}: {v}")

    def _get_lb_models():
        if model_name.startswith("df") or model_name.startswith("laserbeak"):
            net = WrapDFNet(
                num_classes=n_classes, input_channels=len(inputs), **model_config
            )
            net.name = model_name

            return net

        raise NotImplementedError(f"Model '{model_name}' not implemented yet.")

    def _get_local_models():
        match model_name:
            case "df":
                if input_len == 5000:
                    return DF(n_classes, large_input=False)
                if input_len == 10_000:
                    return DF(n_classes, large_input=True)

                raise ValueError("Invalid input len for DF")
            case "rf" | "rf*":
                return RF(n_classes)
            case "march":
                return March(
                    n_classes=n_classes, in_channels=len(inputs), **model_config
                )
            case _:
                raise NotImplementedError(f"Model '{model_name}'")

    if source == "lb":
        model = _get_lb_models()
    elif source == "local":
        model = _get_local_models()
    else:
        raise KeyError("Invalid source for model.")

    logger.info(f"\t-->{count_parameters(model)} parameters.")

    return model
