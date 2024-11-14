from dataclasses import dataclass


@dataclass
class assets:
    DIR: str = "dir"
    SIZE: str = "size"
    TIME: str = "time"
    LABEL: str = "label"
    FLOW_ID: str = "flow_id"

    TIME_IDX: int = 0
    DIR_IDX: int = 1
    SIZE_IDX: int = 2
