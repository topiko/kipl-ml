from __future__ import annotations

import subprocess
from collections.abc import Callable

from kipl_ml.defences.fixed_machines import _FixedMachine
from kipl_ml.logging.logger import get_logger
from rustbindings import sim_trace_from_file_advanced

logger = get_logger(__name__)


class Interspace(_FixedMachine):
    def __init__(
        self,
        network_delay_millis: int | tuple[int, int] | Callable[[], int],
        n: int = 10000,
        seed: int = 0,
    ):

        logger.warning(
            "Interspace produces plenty of machines, I do not know ow to use them during runtime. EXPERIMENTAL!!!"
        )

        self.machination_kwargs: dict[str, int | float] = {
            "n": n,
            "seed": seed,
        }
        super().__init__(
            network_delay_millis=network_delay_millis,
            machination_kwargs=self.machination_kwargs,
        )

    def _machination(self, tmpfile_: str, n: int, seed: int = 0) -> None:
        run = subprocess.run(
            [
                self._rust_machination,
                "fixed",
                "-c",
                "interspace_client",
                "-s",
                "interspace_server",
                "-n",
                str(n),
                "-o",
                tmpfile_,
            ],
            check=False,
            capture_output=True,
        )

        if run.returncode != 0:
            raise RuntimeError(f"Interspace machination failed!! --> {run.stderr!r}")


if __name__ == "__main__":
    inter = Interspace(network_delay_millis=50, n=10000)
    print(inter.report())
