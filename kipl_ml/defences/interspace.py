from __future__ import annotations

import subprocess

from kipl_ml.defences.fixed_machines import _FixedMachine
from kipl_ml.logging.logger import get_logger

logger = get_logger(__name__)


class Interspace(_FixedMachine):
    def __init__(
        self,
        n_machines: int,
        seed: int = 42,
        fixed_per_trace: bool = True,
        simul_kwargs: dict | None = None,
    ):
        self._client_machine = "InterspaceClient"
        self._server_machine = "InterspaceServer"
        self._n_machines = n_machines
        super().__init__(seed=seed, fixed_per_trace=fixed_per_trace, simul_kwargs=simul_kwargs)

    def _machination(self, tmpfile_: str, seed: int) -> None:
        run = subprocess.run(
            [
                self._rust_maybenot,
                "fixed",
                "--client",
                "interspace_client",
                "--server",
                "interspace_server",
                "--output",
                tmpfile_,
                "--seed",
                str(seed),
                "--n",
                str(self._n_machines),
            ],
            check=False,
            capture_output=True,
        )
        self.machination_args = run.args

        if run.returncode != 0:
            raise RuntimeError(f"Interspace maybenot failed!! --> {run.stderr!r}")


if __name__ == "__main__":
    inter = Interspace(n_machines=10000)
    print(inter.report())
