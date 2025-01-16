from kipl_ml.defences.base import NoDefence, _Def
from kipl_ml.defences.maybenot import Maybenot
from kipl_ml.defences.naive import Chi2Delays, RandomPadding
from kipl_ml.logging.logger import get_logger

logger = get_logger(__name__)


def _forge_single_defence(**kwargs) -> _Def:
    defence_name = kwargs.pop("name")

    match defence_name:
        case "random_padding":
            return RandomPadding(**kwargs)
        case "chi2delays":
            return Chi2Delays(**kwargs)
        case "maybenot":
            return Maybenot(**kwargs)
        case "no_defence":
            return NoDefence()
        case _:
            raise ValueError(f"Invalid defence: {defence_name}")
