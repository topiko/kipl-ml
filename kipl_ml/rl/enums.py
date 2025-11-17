from enum import StrEnum


class Actions(StrEnum):
    SEND_PADDING_UP = "send_up"
    SEND_PADDING_DOWN = "send_down"
    SEND_BUFFER = "send_buffer"
    WAIT = "wait"
