from enum import StrEnum


class Actions(StrEnum):
    WAIT = "wait"
    SEND_BUFFER = "send_from_buffer"
    SEND_PADDING_UP = "send_padding_up"
    SEND_PADDING_DOWN = "send_padding_down"

    COUNT = "count_to_send"
