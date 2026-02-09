from enum import StrEnum

from kipl_ml.data import assets


class Feats(StrEnum):
    DIRS = assets.DIRS
    SIZES = assets.SIZES
    TIMES = assets.TIMES
    PADDING = assets.PADDING
    TIMES_NORMALIZED = f"normalized_{TIMES}"
    TIMES_MAX_NORMALIZED = f"max_normalized_{TIMES}"
    CUM_TIMES = f"cum_{TIMES}"
    CUM_SIZES = f"cum_{SIZES}"
    CUM_SIZES_NORMALIZED = f"normalized_{CUM_SIZES}"
    LABEL = "label"
    IATS = "iats"
    IATS_NORMALIZED = f"normalized_{IATS}"
    IATS_MAX_NORMALIZED = f"max_normalized_{IATS}"
    LOG1P_IATS = f"log1p_{IATS}"
    UP_IATS = f"up_{IATS}"
    UP_IATS_NORMALIZED = f"up_{IATS_NORMALIZED}"
    DOWN_IATS = f"down_{IATS}"
    DOWN_IATS_NORMALIZED = f"down_{IATS_NORMALIZED}"
    UP_PACKETS = "up_packets"
    DOWN_PACKETS = "down_packets"
    TIME_DIRS = f"{TIMES}_dirs"
    IAT_DIRS = f"{IATS}_dirs"
    # FLOW_IAT_DIRS = f"flow_{IAT_DIRS}"
    IAT_DIRS_NORMALIZED = f"{IATS_NORMALIZED}_dirs"
    CUM_SIZES_MAX_NORMALIZED = f"max_normalized_{CUM_SIZES}"
    BURST_EDGES = "burst_edges"
    BURST_LENS = "burst_lens"
    BURST_DURS = "burst_durs"
    BURST_DIRS = "burst_dirs"
    BURST_RELDURS = "burst_reldurs"
    FLOW_IATS = "flow_iats"
    FLOW_IATS_NORMALIZED = f"normalized_{FLOW_IATS}"
    LOG_INV_FLOW_IATS = f"log_inv_{FLOW_IATS}"
    LOG_INV_FLOW_IATS_NORMALIZED = f"log_inv_{FLOW_IATS_NORMALIZED}"
    LOG_INV_FLOW_IATS_NORMALIZED_DIRS = f"log_inv_{FLOW_IATS_NORMALIZED}_dirs"
    LOG_INV_FLOW_IAT_DIRS = f"{LOG_INV_FLOW_IATS}_dirs"
    RUNNING_RATE_SIZES = f"running_rate_{SIZES}"
    RUNNING_RATE_SIZES_MAX_NORMALIZED = f"max_normalized_running_rate_{SIZES}"
    SIZE_DIRS = f"{SIZES}_dirs"
    CUM_SIZE_DIRS = f"cum_{SIZE_DIRS}"
    CUM_SIZE_DIRS_MAX_NORMALIZED = f"max_normalized_{CUM_SIZE_DIRS}"
    TAM_UP_COUNTS = "tam-upload"
    TAM_UP_TIMES = "tam-upload-times"
    TAM_UP_COUNTS_MAX_NORMALIZED = f"{TAM_UP_COUNTS}_max_normalized"
    TAM_DOWN_COUNTS = "tam-download"
    TAM_DOWN_TIMES = "tam-download-times"
    TAM_DOWN_COUNTS_MAX_NORMALIZED = f"{TAM_DOWN_COUNTS}_max_normalized"

    # Helpers:
    DIR_PROBS = "dir_probs"
    UP_BUFFER = "up_buffer"
    UP_COUNT = "up_count"
    DOWN_BUFFER = "down_buffer"
    DOWN_COUNT = "down_count"
    ACTION_LOGITS = "action_logits"
    STATE_VALUE = "state_value"
    SEQ_LENS = "seq_lens"
    Dt = "Delta time"
    SILENCE_FLAG = "silence_flag"
    DISC_ID = "disc_id"

    def __str__(self) -> str:
        return self.value

    def __repr__(self) -> str:
        return self.value
