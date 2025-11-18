from enum import IntEnum


class Actions(IntEnum):
    SEND_PADDING_UP = 0
    SEND_PADDING_DOWN = 1
    SEND_BUFFER = 2
    WAIT = 3
