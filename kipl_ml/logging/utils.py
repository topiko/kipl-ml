from typing import Any

KEY_LEN = 25


def key_val_fmt(key: str, val: Any) -> str:
    return f"{key:>{KEY_LEN}}: {val}"
