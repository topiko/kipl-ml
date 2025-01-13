import json
import os

import configs as lasereak_configs


def count_parameters(model) -> str:
    np = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    return f"{np:.02f} M"


def get_laserbeak_model_config(model_name: str) -> dict:
    config_path = os.path.join(list(lasereak_configs.__path__)[0], model_name + ".json")
    with open(config_path, "r") as fi:
        model_config = json.load(fi)
    return model_config
