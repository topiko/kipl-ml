DIRS: str = "dirs"
SIZES: str = "sizes"
TIMES: str = "times"
LABEL: str = "label"
PRED: str = "pred"
PRED_CLS_PROB: str = "prob"
PADDING: str = "padding"
TRACE_ID: str = "trace_id"

ORIG_PACKETS: str = "orig_packets"
TIMES_IDX: int = 0
DIRS_IDX: int = 1
SIZES_IDX: int = 2


def XV_SPLIT(n_splits: int) -> str:
    return f"xv_splits-{n_splits}"
