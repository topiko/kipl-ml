from torch import nn

from kipl_ml.logging.logger import get_logger
from kipl_ml.models.df import DF
from kipl_ml.models.laserbeak import WrapDFNet
from kipl_ml.models.march import March
from kipl_ml.models.rf import RF
from kipl_ml.models.utils import count_parameters

logger = get_logger(__name__)


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

        raise NotImplementedError("Model not implemented yet.")

    def _get_local_models():
        match model_name:
            case "df":
                if input_len == 5000:
                    return DF(n_classes, large_input=False)
                if input_len == 10_000:
                    return DF(n_classes, large_input=True)

                raise ValueError("Invalid input len for DF")
            case "rf":
                return RF(n_classes)
            case "march":
                return March(
                    n_classes=n_classes, in_channels=len(inputs), **model_config
                )
            case _:
                raise NotImplementedError()

    if source == "lb":
        model = _get_lb_models()
    elif source == "local":
        model = _get_local_models()
    else:
        raise KeyError("Invalid source for model.")

    logger.info(f"\t-->{count_parameters(model)} parameters.")

    return model
