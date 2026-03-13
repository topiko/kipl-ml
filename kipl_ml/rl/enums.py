from enum import StrEnum


class Actions(StrEnum):
    DO_NOTHING = "do_nothing"
    # Backward compat alias.
    WAIT = "do_nothing"
    SEND_COUNT_UP = "send_up"
    SEND_COUNT_DOWN = "send_down"

    # Internal keys used during forward pass (renamed to *_BINS after).
    SEND_TIME_UP = "send_time_up"
    SEND_TIME_DOWN = "send_time_down"

    # Exclusive action: block all packets for a duration (in bins).
    DELAY_BINS = "delay_bins"

    SEND_UP_AFTER_BINS = "send_up_after_bins"
    SEND_DOWN_AFTER_BINS = "send_down_after_bins"

    SELECTOR = "selector"
