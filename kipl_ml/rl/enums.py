from enum import StrEnum


class Actions(StrEnum):
    WAIT = "wait"
    SEND_COUNT_UP = "send_up"
    SEND_COUNT_DOWN = "send_down"
    SEND_TIME_UP = "send_time_up"
    SEND_TIME_DOWN = "send_time_down"

    SELECTOR = "selector"
