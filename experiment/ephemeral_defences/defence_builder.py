import numpy as np
from kipl_ml.defences.base import NoDefence
from kipl_ml.defences.maybenot import Deck, DeckStats, Maybenot
from kipl_ml.logging.logger import get_logger
from omegaconf import OmegaConf

logger = get_logger(__name__)


def no_def(netwk_delay: tuple[int, int]) -> dict[str, NoDefence]:

    defence_train = NoDefence(network_delay_millis=netwk_delay)
    defence_valid = defence_train
    defence_test = defence_train

    return {
        "defence_train": defence_train,
        "defence_valid": defence_valid,
        "defence_test": defence_test,
    }


def maybenot(cfg: OmegaConf, netwk_delay: tuple[int, int]) -> dict[str, Maybenot]:

    maybenot_config = dict(cfg.defence)
    maybenot_config["network_delay_millis"] = netwk_delay
    deck_path = maybenot_config.pop("deck_path")
    maybenot_config.pop("type")
    n_machines = maybenot_config.pop("n_machines")

    rng = np.random.default_rng(seed=cfg.seed)

    deck_stats = DeckStats.load(deck_path)
    machine_idxs = list(
        rng.choice(deck_stats.n_machines, size=n_machines, replace=False)
    )

    maybenot_config["deck"] = Deck(deck_stats, machine_idxs)

    defence_train = Maybenot(**maybenot_config)

    if n_machines > 16000:
        logger.info("Setting separate state machines for train and test.")

        test_machines = list(
            set(range(deck_stats.n_machines)).difference(set(machine_idxs))
        )

        if len(test_machines) < 2000:
            logger.warning("Small amount of machines available fro testing")

        maybenot_config["deck"] = Deck(deck_stats, test_machines)
        defence_valid = Maybenot(**maybenot_config)
        defence_test = Maybenot(**maybenot_config)
    else:
        defence_valid = defence_train
        defence_test = defence_train

    return {
        "defence_train": defence_train,
        "defence_valid": defence_valid,
        "defence_test": defence_test,
    }
