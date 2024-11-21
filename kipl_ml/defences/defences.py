from typing import Any

import torch
from kipl_ml.defences.base import _Def
from kipl_ml.defences.naive import Chi2Delays, RandomPadding
from kipl_ml.logging.logger import get_logger

logger = get_logger(__name__)


def forge_defence(**kwargs) -> _Def:
    defence_name = kwargs.pop("name")

    match defence_name:
        case "random_padding":
            return RandomPadding(**kwargs)
        case "chi2delays":
            return Chi2Delays(**kwargs)
        case _:
            raise ValueError(f"Invalid defence: {defence_name}")


class Defences(_Def):
    def __init__(self, defences: list[_Def] | list[dict[str, Any]]):
        if all(isinstance(defence, _Def) for defence in defences):
            _defences = defences
        elif all(isinstance(defence, dict) for defence in defences):
            _defences: list[_Def] = []
            for def_dict in defences:
                assert isinstance(def_dict, dict), "Defences must be a list of dicts"
                _defences.append(forge_defence(**def_dict))

        else:
            raise ValueError("Invalid defences")

        self.defences = _defences

    def report(self, to_log: bool = True) -> str:
        str_ = "Defences...\n"
        for defence in self.defences:
            str_ += defence.report(to_log=False) + "\n"

        if to_log:
            for line in str_.split("\n"):
                logger.info(line)

        return str_

    def __call__(self, trace: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        for defence in self.defences:
            trace = defence(trace)

        return trace


class NoDefence(_Def):

    def report(self, to_log: bool = True) -> str:
        str_ = "No defence applied"
        if to_log:
            logger.info(str_)
        return str_

    def __call__(self, trace: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return trace
