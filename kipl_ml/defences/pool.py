from __future__ import annotations

import os
from collections.abc import Sequence

import torch

from kipl_ml.defences.base import DEFENCE_TYPE_KW, _Def
from kipl_ml.logging.utils import log_multiline
from kipl_ml.network.network import NetworkContextIntDict
from kipl_ml.tools.rng_samplers import MachineRng
from kipl_ml.trace.enums import Feats


class DefencePool(_Def):
    """Sample one child defence per trace from a pool of existing defences."""

    def __init__(
        self,
        defences: Sequence[_Def],
        seed: int | None = 42,
        fixed_per_trace: bool = False,
        name: str | None = None,
        member_ids: Sequence[str] | None = None,
    ):
        if not defences:
            raise ValueError("DefencePool requires at least one defence")
        if fixed_per_trace:
            raise ValueError("DefencePool does not support legacy fixed_per_trace mode")

        super().__init__(seed=seed, fixed_per_trace=fixed_per_trace)
        self.defences = list(defences)
        self.defence_rng = MachineRng(len(self.defences), seed=seed)
        self.pool_name = name
        self.member_ids = (
            tuple(member_ids)
            if member_ids is not None
            else tuple(str(index) for index in range(len(defences)))
        )
        if (
            len(self.member_ids) != len(defences)
            or len(set(self.member_ids)) != len(defences)
            or any(
                not isinstance(member, str) or not member for member in self.member_ids
            )
        ):
            raise ValueError(
                "Pool member IDs must be unique nonempty strings, one per defence"
            )

    @property
    def name(self) -> str:
        return self.pool_name or self.__class__.__name__

    def _select_defence_idx(self) -> int:
        return int(self.defence_rng())

    def _simulate(
        self,
        trace_path: os.PathLike,
        machine_idx: int | None = None,
        trim_raw: int = 0,
        network_context: NetworkContextIntDict | None = None,
    ) -> dict[Feats, torch.Tensor]:
        trace, _ = self.simulate_with_metadata(
            trace_path,
            machine_idx=machine_idx,
            trim_raw=trim_raw,
            network_context=network_context,
        )
        return trace

    def simulate_with_metadata(
        self,
        trace_path: os.PathLike,
        machine_idx: int | None = None,
        trim_raw: int = 0,
        network_context: NetworkContextIntDict | None = None,
    ) -> tuple[dict[Feats, torch.Tensor], dict[str, str]]:
        if not isinstance(trace_path, os.PathLike):
            raise TypeError(
                f"Expected trace to be os.PathLike, got: {type(trace_path)}"
            )
        if machine_idx is not None:
            raise ValueError(
                "DefencePool does not support legacy machine_idx selection"
            )

        defence_idx = self._select_defence_idx()
        defence = self.defences[defence_idx]

        trace, metadata = defence.simulate_with_metadata(
            trace_path,
            machine_idx=None,
            trim_raw=trim_raw,
            network_context=network_context,
        )
        return trace, {**metadata, "defence_id": self.member_ids[defence_idx]}

    def report(self, to_log: bool = False) -> str:
        str_ = "DefencePool:\n"
        if self.pool_name is not None:
            str_ += f"\tName: {self.pool_name}\n"
        str_ += f"\tN defences: {len(self.defences)}\n"
        str_ += f"\tFixed per trace: {self.FIXED_PER_TRACE}\n"
        str_ += "\tMembers:\n"
        for idx, defence in enumerate(self.defences):
            member_report = defence.report(to_log=False).rstrip()
            member_report = member_report.replace("\n", "\n\t\t")
            str_ += f"\t\t{idx}: {member_report}\n"

        if to_log:
            log_multiline(str_)

        return str_

    def _mlflow_log_params(self) -> dict[str, str]:
        member_types = [d.__class__.__name__.lower() for d in self.defences]
        d = {
            DEFENCE_TYPE_KW: self.__class__.__name__.lower(),
            "fixed_per_trace": str(self.FIXED_PER_TRACE),
            "n_defences": str(len(self.defences)),
            "member_types": ",".join(member_types),
        }
        if self.pool_name is not None:
            d["pool_name"] = self.pool_name
        return d
