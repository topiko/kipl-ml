import numpy as np
from mbnt import load_trace_to_str
from src.data import load_full_dataset

from kipl_ml.data.utils import (
    Datasets,
    assets,
    get_std_trace_array,
    load_dataset_meta_df,
)


def main():

    meta_df = (
        load_dataset_meta_df(Datasets.BIGENOUGH)
        .loc[:, [assets.TRACE_PATH, assets.TRACE_ID, assets.PAGE_LABEL]]
        .sort_values(assets.TRACE_ID)
    )

    def get_my_trace(idx) -> np.ndarray:

        return get_std_trace_array(meta_df.loc[idx, assets.TRACE_PATH])

    def get_my_trace_str(idx) -> str:
        return load_trace_to_str(str(meta_df.loc[idx, assets.TRACE_PATH]))

    data, labels, IDs, class_names = load_full_dataset(
        data_dir="/home/topiko/Playground/KIPL/.data/laserbeak/wf-bigenough",
        include_unm=False,
        mon_sample_idx=np.arange(19000),
        mon_raw_data_name="undef-mon.pkl",
        class_divisor=10,
    )

    vals = []
    for label, val in labels.items():
        print(label, val)
        vals.append(val)

    pairs = []
    for key in data.keys():
        print(key, end=" ")
        meta_df_ = meta_df[meta_df.loc[:, assets.PAGE_LABEL] == int(key.split("-")[0])]
        dat = data[key][:, 0].flatten()
        for idx in meta_df_.index:

            test = get_my_trace(idx)[0].flatten() / 1e6
            if len(test) != len(dat):
                continue

            close = np.isclose(dat, test)

            if all(close):
                trace_id = meta_df.loc[idx, assets.TRACE_ID]
                print("->", meta_df.loc[idx, assets.TRACE_ID])

                pairs.append([key, trace_id])

                break

        pairs_ = np.array(pairs)
        np.savetxt("map.csv", pairs_, fmt="%s,%s")
    print(set(vals), len(set(vals)))

    print(data["0-0"])
    print(meta_df.head())

    breakpoint()


if __name__ == "__main__":
    main()
