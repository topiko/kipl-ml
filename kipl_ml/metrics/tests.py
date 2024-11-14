import unittest

import numpy as np
import torch
from kipl_ml.metrics.clf_metrics import Accuracy


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


if __name__ == "__main__":
    unittest.main()
