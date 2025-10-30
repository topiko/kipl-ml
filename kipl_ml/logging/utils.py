from typing import Any

from kipl_ml.logging.logger import get_logger

KEY_LEN = 25

logger = get_logger(__name__)


def log_multiline(log_str: str) -> None:
    for line in log_str.split("\n"):
        logger.info(line)


def key_val_fmt(key: str, val: Any, key_len: int = KEY_LEN, suffix: str = "\n") -> str:
    return f"{key:>{key_len}}: {val}{suffix}"


def log_dict(dict_: dict[str, float | int], key_len: int = KEY_LEN):
    for k, v in dict_.items():
        if isinstance(v, int):
            v_ = str(v)
        elif isinstance(v, float):
            v_ = f"{v:.03f}"

        logger.info(key_val_fmt(k, v_, key_len=key_len, suffix=""))
