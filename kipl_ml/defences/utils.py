from kipl_ml.defences.base import NoDefence, _Def
from kipl_ml.defences.maybenot import Maybenot
from kipl_ml.logging.logger import get_logger

logger = get_logger(__name__)


def _forge_single_defence(**kwargs) -> _Def:
    defence_name = kwargs.pop("name")

    match defence_name:
        case "maybenot":
            return Maybenot(**kwargs)
        case "no_defence":
            return NoDefence(**kwargs)
        case _:
            raise ValueError(f"Invalid defence: {defence_name}")
