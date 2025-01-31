import numpy as np
from src.data import load_data


def main():

    ids = []
    xvs = []
    for i in range(10):

        trainloader, valloader, testloader, classes, test_ids = load_data(
            "be",
            batch_size=64,
            tr_transforms=[],
            te_transforms=[],
            tr_augments=[],
            te_augments=[],
            root="/home/topiko/Playground/KIPL/.data/laserbeak/",
            val_perc=1.0 / 9,
            include_unm=False,
            multisample_count=1,
            tmp_directory="./tmp",
            tmp_subdir=None,
            keep_tmp=False,
            subpage_as_labels=False,
            te_chunk_no=i,
        )

        ids += test_ids

    test_ids = np.array(ids)
    xvs = np.array(xvs)

    test_ids = np.vstack((test_ids, xvs)).T

    np.savetxt("lb_xvs.csv", test_ids, fmt="%s,%s")

    print(
        len(np.unique(test_ids[:, 0])),
        "== 19 000 ? If not only subset is used for testing",
    )


if __name__ == "__main__":
    main()
