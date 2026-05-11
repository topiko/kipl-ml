from __future__ import annotations

import subprocess

from kipl_ml.defences.fixed_machines import _FixedMachine
from kipl_ml.logging.logger import get_logger

logger = get_logger(__name__)


class Breakpad(_FixedMachine):
    def __init__(
        self,
        seed: int = 42,
        simul_kwargs: dict | None = None,
    ):
        self._client_machine = "break_pad_client"
        self._server_machine = "break_pad_server"
        super().__init__(seed=seed, simul_kwargs=simul_kwargs)

    def _machination(self, tmpfile_: str, seed: int) -> None:
        run = subprocess.run(
            [
                self._rust_maybenot,
                "fixed",
                "--client",
                self._client_machine,
                "--server",
                self._server_machine,
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
            raise RuntimeError(f"Breakpad maybenot failed!! --> {run.stderr!r}")


if __name__ == "__main__":
    breakpad = Breakpad()
    print(breakpad.report())
