from typing import Any

KEY_LEN = 25


def key_val_fmt(key: str, val: Any, key_len: int = KEY_LEN) -> str:
    return f"{key:>{key_len}}: {val}"
