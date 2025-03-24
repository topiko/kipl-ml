import argparse

from kipl_ml.data.utils import get_std_trace_dict
from kipl_ml.trace.features import FEAT_NAME_MAP, FeatureTrs

PATH_TO_BE = "/home/topiko/Playground/KIPL/.data/wf-data/bigenough-95x10x20-standard-rngsubpages/"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", type=int, default=0)
    parser.add_argument("--npackets", type=int, default=1000)
    parser.add_argument("--trace", type=str, default="0000-0000-0001.log")

    args = parser.parse_args()

    # This returns a w. raw tensors as the trace:
    trace_d = get_std_trace_dict(
        path=PATH_TO_BE + f"{args.label}/{args.trace}",
        network_delay_millis=0,
        network_packets_per_second=0,
    )

    # It needs to be converted to float32
    trace_d_float32 = {k: v.float() for k, v in trace_d.items()}

    # Here are the lb features - they are mapped to local standard:
    lb_feats = [
        FEAT_NAME_MAP[f]
        for f in (
            "time_dirs",
            "times_norm",
            "cumul_norm",
            "iat_dirs",
            "inv_iat_log_dirs",
            "running_rates",
        )
    ] + ["times"]

    # Create a feature trasnformer object:
    feat_trs = FeatureTrs(feature_names=lb_feats, n_packets=args.npackets)
    # It needs to be fitted before using (use here for the fitting though):
    feat_trs.get_shapes(trace_d_float32)

    # Transform the trace:
    transformed_d = feat_trs(trace_d_float32)

    for k, v in transformed_d.items():
        print(f"{k:>25}: {v}")


if __name__ == "__main__":
    main()
