from __future__ import annotations

import subprocess
from collections.abc import Callable

from kipl_ml.defences.fixed_machines import _FixedMachine
from kipl_ml.logging.logger import get_logger

logger = get_logger(__name__)


class Interspace(_FixedMachine):
    def __init__(
        self,
        network_delay_millis: int | tuple[int, int] | Callable[[], int],
        n_machines: int,
        seed: int = 42,
    ):
        self.machination_kwargs: dict[str, int | float] = {
            "n_machines": n_machines,
            "seed": seed,
        }
        super().__init__(
            network_delay_millis=network_delay_millis,
            machination_kwargs=self.machination_kwargs,
        )

    def _machination(self, tmpfile_: str, n_machines: int, seed: int) -> None:
        run = subprocess.run(
            [
                self._rust_machination,
                "fixed",
                "-c",
                "interspace_client",
                "-s",
                "interspace_server",
                "-n",
                str(n_machines),
                "-o",
                tmpfile_,
                "--seed",
                str(seed),
            ],
            check=False,
            capture_output=True,
        )

        if run.returncode != 0:
            raise RuntimeError(f"Interspace machination failed!! --> {run.stderr!r}")


if __name__ == "__main__":
    inter = Interspace(network_delay_millis=50, n=10000)
    print(inter.report())
