from __future__ import annotations

import subprocess

from kipl_ml.defences.fixed_machines import _FixedMachine
from kipl_ml.logging.logger import get_logger

logger = get_logger(__name__)


class Tamaraw(_FixedMachine):
    def __init__(
        self,
        network_delay_millis: tuple[int, int],
        network_pps: tuple[int, int],
        pc: float,
        ps: float,
        window_val: int,
        seed: int = 42,
        fixed_per_trace: bool = False,
        simul_kwargs: dict | None = None,
    ):
        self._pc = pc
        self._ps = ps
        self._window_val = window_val
        super().__init__(
            network_delay_millis=network_delay_millis,
            network_pps=network_pps,
            seed=seed,
            fixed_per_trace=fixed_per_trace,
            simul_kwargs=simul_kwargs,
        )

    def _machination(self, tmpfile_: str, seed: int) -> None:
        client_machine = f"tamaraw {self._pc} {self._window_val}"
        server_machine = f"tamaraw {self._ps} {self._window_val}"

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
