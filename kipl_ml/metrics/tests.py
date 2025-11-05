import unittest

import numpy as np
import torch

from kipl_ml.metrics.clf_metrics import Accuracy
from kipl_ml.metrics.overhead_metrics import BurstLenOverhead
from kipl_ml.trace.enums import Feats


class TestMets(unittest.TestCase):
    def test_accuracy(self):
        acc_m = Accuracy()

        pred = torch.tensor([1, 2, 3, 4, 5])
        target = torch.tensor([1, 2, 3, 4, 5])

        acc_ = acc_m(pred, target)
        self.assertTrue(np.isclose(acc_, 1.0))

        pred = torch.tensor([2, 1, 1, 4, 5])

        acc_ = acc_m(pred, target)
        self.assertTrue(np.isclose(acc_, 0.4))

    def test_burst_len_overhead(self):
        mblen = BurstLenOverhead()

        N = 10
        blens = torch.randint(N, 100, (1, 100)).float()

        extras = torch.rand_like(blens) * 10

        xobs = {Feats.BURST_LENS: blens + extras}
        x = {Feats.BURST_LENS: blens}

        mval = mblen(xobs, x)

        overhead = (extras.sum(dim=1) / blens.sum(dim=1)).mean()

        self.assertTrue(torch.allclose(mval, overhead))


if __name__ == "__main__":
    unittest.main()
