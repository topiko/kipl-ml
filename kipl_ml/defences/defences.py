import os
from typing import Any

import kipl_ml.data.assets as assets
import pandas as pd
import torch
from kipl_ml.data.utils import (
    get_std_trace_array,
    get_std_trace_dict,
    load_dataset_meta_df,
    preserve_class_frac_sample,
)
from kipl_ml.defences.base import _Def
from kipl_ml.defences.maybenot import Maybenot
from kipl_ml.defences.naive import Chi2Delays, RandomPadding
from kipl_ml.logging.logger import get_logger
from kipl_ml.logging.utils import key_val_fmt
from kipl_ml.trace.features import FeatureTrs
from kipl_ml.trace.transforms import _TR
from torch.utils.data import Dataset

logger = get_logger(__name__)


def forge_defence(**kwargs) -> _Def:
    defence_name = kwargs.pop("name")

    match defence_name:
        case "random_padding":
            return RandomPadding(**kwargs)
        case "chi2delays":
            return Chi2Delays(**kwargs)
        case "maybenot":
            return Maybenot(**kwargs)
        case "no_defence":
            return NoDefence()
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

    def sim_defence(self, trace_path: os.PathLike) -> dict[str, torch.Tensor]:

        if isinstance(self.defences[0], Maybenot) and len(self.defences) == 1:
            return self.defences[0].sim_defence(trace_path)
        if any(isinstance(defence, Maybenot) for defence in self.defences):
            raise NotImplementedError("Combining maybenot w. others not implemented.")

        trace = get_std_trace_dict(trace_path)

        for defence in self.defences:
            trace = defence(trace)

        return trace

    def __call__(self, trace: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        raise NotImplementedError("You need to implement this method")


class NoDefence(_Def):

    def report(self, to_log: bool = True) -> str:
        str_ = "No defence applied"
        if to_log:
            logger.info(str_)
        return str_

    def __call__(self, trace: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return trace
