from __future__ import annotations

import subprocess

from kipl_ml.defences.fixed_machines import _FixedMachine
from kipl_ml.logging.logger import get_logger

logger = get_logger(__name__)


class Regulator(_FixedMachine):
    def __init__(
        self,
        network_delay_millis: tuple[int, int],
        network_pps: tuple[int, int],
        U: float,
        C: float,
        R: float,
        D: float,
        T: int,
        padding_budget: int,
        seed: int = 42,
        fixed_per_trace: bool = False,
    ):

        self.machination_kwargs = {
            "U": U,
            "C": C,
            "R": R,
            "D": D,
            "T": T,
            "padding_budget": padding_budget,
            "seed": seed,
        }
        super().__init__(
            network_delay_millis=network_delay_millis,
            network_pps=network_pps,
            machination_kwargs=self.machination_kwargs,
            fixed_per_trace=fixed_per_trace,
        )

    def _machination(
        self,
        tmpfile_: str,
        U: float,
        C: float,
        R: float,
        D: float,
        T: float,
        padding_budget: int,
        seed: int,
    ) -> None:

        run = subprocess.run(
            [
                self._rust_machination,
                "fixed",
                "-c",
                f"regulator_client {U} {C}",
                "-s",
                f"regulator_server {R} {D} {T} {padding_budget}",
                "-o",
                tmpfile_,
                "--seed",
                str(seed),
            ],
            check=False,
            capture_output=True,
        )

        if run.returncode != 0:
            raise RuntimeError(
                f"{self.__class__.__name__} machination failed!! --> {run.stderr!r}"
            )
