from __future__ import annotations

import os
import sys


def _ensure_experiment_name(argv: list[str]) -> list[str]:
    if ("-en" in argv) or ("--experiment-name" in argv):
        return argv
    return ["-en", "Ethan", *argv]


def main() -> None:
    # Make outputs land in experiment/Ethan/tables regardless of where called from.
    this_dir = os.path.dirname(os.path.abspath(__file__))
    ethan_dir = os.path.dirname(this_dir)
    os.chdir(ethan_dir)

    os.makedirs("tables", exist_ok=True)

    argv = _ensure_experiment_name(sys.argv[1:])
    sys.argv = [sys.argv[0], *argv]

    from experiment.ephemeral_defences.scripts.overview_table import main as _main

    _main()


if __name__ == "__main__":
    main()
