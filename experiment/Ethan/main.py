from __future__ import annotations

import os

import dotenv
import hydra
from omegaconf import DictConfig


WORKING_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(WORKING_DIR))
CONFIG_DIR_PATH = os.path.join(WORKING_DIR, "config")


dotenv.load_dotenv(os.path.join(REPO_ROOT, ".env"))


@hydra.main(config_path=CONFIG_DIR_PATH, config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    from experiment.ephemeral_defences.main import run

    run(cfg)


if __name__ == "__main__":
    main()
