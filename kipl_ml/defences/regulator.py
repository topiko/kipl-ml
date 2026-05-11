from __future__ import annotations

import subprocess

from kipl_ml.defences.fixed_machines import _FixedMachine
from kipl_ml.logging.logger import get_logger

logger = get_logger(__name__)


class Regulator(_FixedMachine):
    def __init__(
        self,
        U: float,
        C: float,
        R: float,
        D: float,
        T: float,
        padding_budget: int,
        B: int,
        seed: int = 42,
        simul_kwargs: dict | None = None,
    ):
        self._U = U
        self._C = C
        self._R = R
        self._D = D
        self._T = T
        self._padding_budget = padding_budget
        self._B = B
        super().__init__(seed=seed, simul_kwargs=simul_kwargs)

    def _machination(self, tmpfile_: str, seed: int) -> None:
        client_machine = f"regulator_client {self._U} {self._C}"
        server_machine = f"regulator_server {self._R} {self._D} {self._T} {self._padding_budget} {self._B}"

        run = subprocess.run(
            [
                self._rust_maybenot,
                "fixed",
                "--client",
                client_machine,
                "--server",
                server_machine,
                "--output",
                tmpfile_,
                "--seed",
                str(seed),
            ],
            check=False,
            capture_output=True,
        )
        self.machination_args = run.args

        if run.returncode != 0:
            raise RuntimeError(
                f"{self.__class__.__name__} maybenot failed!! --> {run.stderr!r}"
            )
