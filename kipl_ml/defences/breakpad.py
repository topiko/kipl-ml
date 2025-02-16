from __future__ import annotations

import subprocess
from collections.abc import Callable

from kipl_ml.defences.fixed_machines import _FixedMachine
from kipl_ml.logging.logger import get_logger

logger = get_logger(__name__)


class Breakpad(_FixedMachine):
    def __init__(
        self,
        network_delay_millis: int | tuple[int, int] | Callable[[], int],
        seed: int = 0,
    ):

        self.machination_kwargs: dict[str, int | float] = {"seed": seed}
        super().__init__(
            network_delay_millis=network_delay_millis,
            machination_kwargs=self.machination_kwargs,
        )

    def _machination(self, tmpfile_: str, seed: int = 0) -> None:
        run = subprocess.run(
            [
                self._rust_machination,
                "fixed",
                "-c",
                "break_pad_client",
                "-s",
                "break_pad_server",
                "-o",
                tmpfile_,
            ],
            check=False,
            capture_output=True,
        )

        if run.returncode != 0:
            raise RuntimeError(f"Breakpad machination failed!! --> {run.stderr!r}")


if __name__ == "__main__":
    breakpad = Breakpad(network_delay_millis=50)
    print(breakpad.report())
