import numpy as np
from omegaconf import OmegaConf

from kipl_ml.defences.base import NoDefence
from kipl_ml.defences.breakpad import Breakpad
from kipl_ml.defences.front import FRONT
from kipl_ml.defences.interspace import Interspace
from kipl_ml.defences.maybenot import Deck, DeckStats, Maybenot
from kipl_ml.logging.logger import get_logger

logger = get_logger(__name__)


def _parse_netwk(cfg: OmegaConf) -> tuple[tuple[int, int], tuple[int, int]]:
    delay = (cfg.network.delay_millis.min, cfg.network.delay_millis.max)
    pps = (cfg.network.pps.min, cfg.network.pps.max)

    return delay, pps


def no_def(cfg: OmegaConf) -> dict[str, NoDefence]:

    netwk_delay, netwk_pps = _parse_netwk(cfg)
    no_defence = NoDefence(
        network_delay_millis=netwk_delay, network_pps=netwk_pps, seed=cfg.seed
    )

    return {
        "defence_train": no_defence,
        "defence_valid": no_defence,
        "defence_test": no_defence,
    }


def breakpad(cfg: OmegaConf) -> dict[str, Breakpad]:
    netwk_delay, netwk_pps = _parse_netwk(cfg)
    defence = Breakpad(
        network_delay_millis=netwk_delay, network_pps=netwk_pps, seed=cfg.seed
    )

    return {
        "defence_train": defence,
        "defence_valid": defence,
        "defence_test": defence,
    }


def interspace(cfg: OmegaConf) -> dict[str, Interspace]:
    d = dict(cfg.defence)
    d.pop("type")
    seed = cfg.seed
    n_train_machines = d.pop("n_train_machines")
    n_valid_machines = d.pop("n_valid_machines")
    n_test_machines = d.pop("n_test_machines")

    netwk_delay, netwk_pps = _parse_netwk(cfg)

    def _interspace(n_machines: int, seed: int):
        return Interspace(
            network_delay_millis=netwk_delay,
            network_pps=netwk_pps,
            **d,
            n_machines=n_machines,
            seed=seed,
        )

    return {
        "defence_train": _interspace(n_train_machines, seed=seed + 1),
        "defence_valid": _interspace(n_valid_machines, seed=seed + 2),
        "defence_test": _interspace(n_test_machines, seed=seed + 3),
    }


def front(cfg: OmegaConf) -> dict[str, FRONT]:
    d = dict(cfg.defence)
    d.pop("type")
    seed = cfg.seed
    n_train_machines = d.pop("n_train_machines")
    n_valid_machines = d.pop("n_valid_machines")
    n_test_machines = d.pop("n_test_machines")

    netwk_delay, netwk_pps = _parse_netwk(cfg)

    def _front(n_machines: int, seed: int):
        return FRONT(
            network_delay_millis=netwk_delay,
            network_pps=netwk_pps,
            **d,
            n_machines=n_machines,
            seed=seed,
        )

    return {
        "defence_train": _front(n_train_machines, seed + 1),
        "defence_valid": _front(n_valid_machines, seed + 2),
        "defence_test": _front(n_test_machines, seed + 3),
    }


def maybenot(cfg: OmegaConf) -> dict[str, Maybenot]:

    maybenot_config = dict(cfg.defence)
    deck_path = maybenot_config.pop("deck_path")
    maybenot_config.pop("type")
    n_machines = maybenot_config.pop("n_machines")

    rng = np.random.default_rng(seed=cfg.seed)

    deck_stats = DeckStats.load(deck_path)
    machine_idxs = list(
        rng.choice(deck_stats.n_machines, size=n_machines, replace=False)
    )

    maybenot_config["deck"] = Deck(deck_stats, machine_idxs)

    seed = cfg.seed
    netwk_delay, netwk_pps = _parse_netwk(cfg)
    maybenot_config["network_delay_millis"] = netwk_delay
    maybenot_config["network_pps"] = netwk_pps

    defence_train = Maybenot(**maybenot_config, seed=seed)

    if n_machines > 16000:
        logger.info("Setting separate state machines for train and test.")

        test_machines = list(
            set(range(deck_stats.n_machines)).difference(set(machine_idxs))
        )

        if len(test_machines) < 2000:
            logger.warning("Small amount of machines available fro testing")

        maybenot_config["deck"] = Deck(deck_stats, test_machines)
        defence_valid = Maybenot(**maybenot_config, seed=seed + 1)
        defence_test = Maybenot(**maybenot_config, seed=seed + 2)
    else:
        defence_valid = Maybenot(**maybenot_config, seed=seed + 2)
        defence_test = Maybenot(**maybenot_config, seed=seed + 3)

    return {
        "defence_train": defence_train,
        "defence_valid": defence_valid,
        "defence_test": defence_test,
    }
