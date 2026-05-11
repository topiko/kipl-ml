from __future__ import annotations

import subprocess

from kipl_ml.defences.fixed_machines import _FixedMachine
from kipl_ml.logging.logger import get_logger

logger = get_logger(__name__)


class FRONT(_FixedMachine):
    def __init__(
        self,
        padding_budget_max_client: int,
        padding_budget_max_server: int,
        window_min_client: float,
        window_min_server: float,
        window_max_client: float,
        window_max_server: float,
        num_states_client: int,
        num_states_server: int,
        n_machines: int,
        seed: int = 42,
        fixed_per_trace: bool = True,
        simul_kwargs: dict | None = None,
    ):
        self._padding_budget_max_client = padding_budget_max_client
        self._padding_budget_max_server = padding_budget_max_server
        self._window_min_client = window_min_client
        self._window_min_server = window_min_server
        self._window_max_client = window_max_client
        self._window_max_server = window_max_server
        self._num_states_client = num_states_client
        self._num_states_server = num_states_server
        self._n_machines = n_machines
        super().__init__(
            seed=seed,
            fixed_per_trace=fixed_per_trace,
            simul_kwargs=simul_kwargs,
        )

    def _machination(self, tmpfile_: str, seed: int) -> None:
        client_machine = f"front {self._padding_budget_max_client} {self._window_min_client} {self._window_max_client} {self._num_states_client}"
        server_machine = f"front {self._padding_budget_max_server} {self._window_min_server} {self._window_max_server} {self._num_states_server}"

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
                "--n",
                str(self._n_machines),
            ],
            check=False,
            capture_output=True,
        )
        self.machination_args = run.args

        if run.returncode != 0:
            raise RuntimeError(f"FRONT maybenot failed!! --> {run.stderr!r}")


if __name__ == "__main__":
    front = FRONT(
        padding_budget_max_client=1,
        padding_budget_max_server=1,
        window_min_client=0.1,
        window_min_server=0.1,
        window_max_client=0.2,
        window_max_server=0.2,
        num_states_client=1,
        num_states_server=1,
        n_machines=1,
    )
    print(front.report())
