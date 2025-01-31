import pandas as pd
import torch
from kipl_ml.data.utils import get_std_trace_dict, load_dataset_meta_df
from kipl_ml.logging.logger import get_logger
from kipl_ml.trace.features import FEAT_NAME_MAP, Feats, get_feature_tr
from src.data import ToProcessed
from src.processor import DataProcessor
from tqdm import tqdm

logger = get_logger(__name__)

LASERBEAK_FEATURES = [
    "iat_dirs",
    "size_dirs",
    "flow_iats",
    "cumul_norm",
    "cumul",
    "time_dirs",
    "times_norm",
    "running_rates",
    "inv_iat_log_dirs",
    "inv_iat_logs",
]


def main():
    meta_df = load_dataset_meta_df("bigenough")
    for feature in LASERBEAK_FEATURES:
        test_feature(feature, meta_df)


def test_feature(feature: str, meta_df: pd.DataFrame):
    feature_list = [feature]
    processor = DataProcessor(feature_list)
    on_load_transforms = ToProcessed(processor)

    my_trs = get_feature_tr(feature_name=FEAT_NAME_MAP[feature], n_packets=7000)
    N = 100

    beg_end_issues = False
    with tqdm(
        meta_df.loc[:, "orig_path"].sample(N),
        desc=f"Testing for feature {feature}",
        ncols=88,
    ) as pbar:
        for i, path in enumerate(pbar):

            x_d = get_std_trace_dict(path, network_delay_millis=0)

            if i == 0:
                my_trs.get_shapes(x_d)

            my_x_tr = my_trs(x_d)[my_trs.output]

            # laserbeak standard:
            # time, size, dir
            # note that the x_d[*] got extended to correct len in the previous step.
            x = torch.vstack((x_d[Feats.TIMES], x_d[Feats.SIZES], x_d[Feats.DIRS])).T
            # apply processing to sample
            x_tr = on_load_transforms(x).flatten()

            if not torch.isclose(x_tr, my_x_tr).all():
                if feature in ("flow_iats", "inv_iat_logs", "inv_iat_log_dirs"):
                    if (~torch.isclose(x_tr, my_x_tr).bool()).sum() < 3:
                        beg_end_issues = True
                        continue

                print(f"Failed for {path}")
                print(x_tr[:20])
                print(my_x_tr[:20])
                breakpoint()
                break

    if beg_end_issues:
        logger.warning(f"{feature} failed but only in max 2 points (beg and end)...")


if __name__ == "__main__":

    main()
