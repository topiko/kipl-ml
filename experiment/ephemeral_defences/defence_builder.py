from omegaconf import OmegaConf

from kipl_ml.defences.base import NoDefence
from kipl_ml.defences.breakpad import Breakpad
from kipl_ml.defences.front import FRONT
from kipl_ml.defences.interspace import Interspace
from kipl_ml.defences.maybenot import Maybenot
from kipl_ml.defences.regulator import Regulator
from kipl_ml.defences.tamaraw import Tamaraw
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

    seed = cfg.seed

    def _breakpad(seed: int):
        return Breakpad(
            network_delay_millis=netwk_delay, network_pps=netwk_pps, seed=seed
        )

    return {
        "defence_train": _breakpad(seed + 1),
        "defence_valid": _breakpad(seed + 2),
        "defence_test": _breakpad(seed + 3),
    }


def tamaraw(cfg: OmegaConf) -> dict[str, Breakpad]:
    netwk_delay, netwk_pps = _parse_netwk(cfg)

    seed = cfg.seed

    def _tamaraw(seed: int):
        return Tamaraw(
            network_delay_millis=netwk_delay,
            network_pps=netwk_pps,
            pc=cfg.defence.pc,
            ps=cfg.defence.ps,
            window_val=cfg.defence.window_val,
            seed=seed,
        )

    return {
        "defence_train": _tamaraw(seed + 1),
        "defence_valid": _tamaraw(seed + 2),
        "defence_test": _tamaraw(seed + 3),
    }


def regulator(cfg: OmegaConf) -> dict[str, Breakpad]:
    netwk_delay, netwk_pps = _parse_netwk(cfg)

    seed = cfg.seed
    def_params = cfg.defence

    def _regulator(seed: int):
        return Regulator(
            network_delay_millis=netwk_delay,
            network_pps=netwk_pps,
            U=def_params.U,
            C=def_params.C,
            R=def_params.R,
            D=def_params.D,
            T=def_params.T,
            padding_budget=def_params.padding_budget,
            seed=seed,
        )

    return {
        "defence_train": _regulator(seed + 1),
        "defence_valid": _regulator(seed + 2),
        "defence_test": _regulator(seed + 3),
    }


def interspace(cfg: OmegaConf) -> dict[str, Interspace]:
    d = dict(cfg.defence)
    d.pop("type")
    seed = cfg.seed
    n_train_machines = d.pop("n_train_machines")
    n_valid_machines = d.pop("n_valid_machines")
    n_test_machines = d.pop("n_test_machines")

    netwk_delay, netwk_pps = _parse_netwk(cfg)

    def _interspace(n_machines: int, fixed_per_trace: bool, seed: int):
        return Interspace(
            network_delay_millis=netwk_delay,
            network_pps=netwk_pps,
            **d,
            n_machines=n_machines,
            seed=seed,
            fixed_per_trace=fixed_per_trace,
        )

    fixed_per_trace = d.pop("fixed_per_trace")
    return {
        "defence_train": _interspace(n_train_machines, fixed_per_trace, seed=seed + 1),
        "defence_valid": _interspace(n_valid_machines, True, seed=seed + 2),
        "defence_test": _interspace(n_test_machines, True, seed=seed + 3),
    }


def front(cfg: OmegaConf) -> dict[str, FRONT]:
    d = dict(cfg.defence)
    d.pop("type")
    seed = cfg.seed
    n_train_machines = d.pop("n_train_machines")
    n_valid_machines = d.pop("n_valid_machines")
    n_test_machines = d.pop("n_test_machines")

    netwk_delay, netwk_pps = _parse_netwk(cfg)

    def _front(n_machines: int, fixed_per_trace: bool, seed: int):
        return FRONT(
            network_delay_millis=netwk_delay,
            network_pps=netwk_pps,
            **d,
            n_machines=n_machines,
            seed=seed,
            fixed_per_trace=fixed_per_trace,
        )

    fixed_per_trace = d.pop("fixed_per_trace")

    return {
        "defence_train": _front(n_train_machines, fixed_per_trace, seed + 1),
        "defence_valid": _front(n_valid_machines, True, seed + 2),
        "defence_test": _front(n_test_machines, True, seed + 3),
    }


def maybenot(cfg: OmegaConf) -> dict[str, Maybenot]:

    mbnt_conf = dict(cfg.defence)
    mbnt_conf.pop("type")

    seed = cfg.seed
    netwk_delay, netwk_pps = _parse_netwk(cfg)
    mbnt_conf["network_delay_millis"] = netwk_delay
    mbnt_conf["network_pps"] = netwk_pps

    mbnt_conf["deck_path"] = (
        "/home/topiko/Playground/KIPL/.maybenot-decks/" + mbnt_conf.pop("deck_name")
    )
    keys = (
        "client_padding_budget",
        "client_blocking_budget",
        "client_padding_frac",
        "client_blocking_frac",
        "server_padding_budget",
        "server_blocking_budget",
        "server_padding_frac",
        "server_blocking_frac",
    )
    for key in keys:
        mbnt_conf[key] = tuple(mbnt_conf[key])

    def _maybenot(n_machines: int, fixed_per_trace: bool, seed: int) -> Maybenot:
        mbnt_conf["n_machines"] = n_machines
        return Maybenot(**mbnt_conf, seed=seed, fixed_per_trace=fixed_per_trace)

    n_train_machines = mbnt_conf.pop("n_train_machines")
    n_valid_machines = mbnt_conf.pop("n_valid_machines")
    n_test_machines = mbnt_conf.pop("n_test_machines")

    fixed_per_trace = mbnt_conf.pop("fixed_per_trace")

    return {
        "defence_train": _maybenot(n_train_machines, fixed_per_trace, seed + 1),
        "defence_valid": _maybenot(n_valid_machines, True, seed + 2),
        "defence_test": _maybenot(n_test_machines, True, seed + 3),
    }
