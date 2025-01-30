DIRS: str = "dirs"
SIZES: str = "sizes"
TIMES: str = "times"
PAGE_LABEL: str = "page_label"
SUB_PAGE_LABEL: str = "sub_page_label"
PRED: str = "pred"
PRED_CLS_PROB: str = "prob"
PADDING: str = "padding"
TRACE_ID: str = "trace_id"
SAMPLE_ID: str = "sample_id"

ORIG_PACKETS: str = "orig_packets"
TIMES_IDX: int = 0
DIRS_IDX: int = 1
SIZES_IDX: int = 2


def XV_SPLIT(n_splits: int, label: str) -> str:
    return f"xv_splits-{label}-{n_splits}"
