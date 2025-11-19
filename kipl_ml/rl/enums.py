from enum import IntEnum


class Actions(IntEnum):
    WAIT = 0
    SEND_BUFFER = 1
    SEND_PADDING_UP = 2
    SEND_PADDING_DOWN = 3
