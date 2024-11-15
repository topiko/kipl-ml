import unittest

import torch
from kipl_ml.data.assets import assets
from kipl_ml.trace.transforms import _TR, get_feature_tr


class TestTR(unittest.TestCase):
    N_PACKETS = 100

    def _get_trace(self, n_packets: int = 50):
        dirs = torch.ones(n_packets, dtype=torch.float32)
        dirs[::2] = -1

        times = torch.arange(n_packets, dtype=torch.float32)
        sizes = torch.ones_like(times)

        return {assets.DIR: dirs, assets.SIZE: sizes, assets.TIME: times}

    def _get_key(self, key: str, trace: dict[str, torch.Tensor]) -> _TR:
        tr = get_feature_tr(key)
        tr.get_shapes(trace)
        return tr

    def _shapes_test(self, tr: _TR, trace: dict[str, torch.Tensor]):
        self.assertEqual(tr.input_size, self.N_PACKETS)
        self.assertEqual(tr.output_size, self.N_PACKETS)

    def _simple_test(self, key: str):
        trace = self._get_trace(self.N_PACKETS)
        tr = self._get_key(key, trace)

        self._shapes_test(tr, trace)

        asset_key = {"dirs": assets.DIR, "sizes": assets.SIZE, "times": assets.TIME}[
            key
        ]

        self.assertTrue(torch.allclose(tr(trace), trace[asset_key]))

    def test_dirs(self):
        self._simple_test("dirs")

    def test_sizes(self):
        self._simple_test("sizes")

    def test_times(self):
        self._simple_test("times")

    def test_iats(self):
        trace = self._get_trace(self.N_PACKETS)
        tr = self._get_key("iats", trace)

        self._shapes_test(tr, trace)

        iats = torch.zeros_like(trace[assets.TIME])
        iats[1:] = torch.diff(trace[assets.TIME], dim=0)

        self.assertTrue(torch.allclose(tr(trace), iats))

    def test_ud_packets(self):
        trace = self._get_trace(self.N_PACKETS)
        tr_up = self._get_key("up_packets", trace)
        tr_down = self._get_key("down_packets", trace)

        self._shapes_test(tr_up, trace)
        self._shapes_test(tr_down, trace)

        self.assertTrue(torch.allclose(tr_up(trace), (trace[assets.DIR] == 1).float()))
        self.assertTrue(torch.allclose(tr_down(trace), (trace[assets.DIR] == -1).float()))

    def test_ud_iats(self):
        trace = self._get_trace(self.N_PACKETS)
        tr_up = self._get_key("up_iats", trace)
        tr_down = self._get_key("down_iats", trace)

        self._shapes_test(tr_up, trace)
        self._shapes_test(tr_down, trace)

        mask = trace[assets.DIR] == 1
        idxs = torch.where(mask)[0]
        iats = torch.zeros_like(trace[assets.TIME])
        iats[idxs[1:]] = torch.diff(trace[assets.TIME][mask], dim=0)

        self.assertTrue(torch.allclose(tr_up(trace), iats))

        mask = trace[assets.DIR] == -1
        idxs = torch.where(mask)[0]
        iats = torch.zeros_like(trace[assets.TIME])
        iats[idxs[1:]] = torch.diff(trace[assets.TIME][mask], dim=0)

        self.assertTrue(torch.allclose(tr_down(trace), iats))

    def test_normalize_iats(self):
        trace = self._get_trace(self.N_PACKETS)
        tr = self._get_key("iats_normalized", trace)

        self._shapes_test(tr, trace)

        iats = torch.zeros_like(trace[assets.TIME])
        iats[1:] = torch.diff(trace[assets.TIME], dim=0)

        iats -= iats.mean()
        iats /= iats.std()

        self.assertTrue(torch.allclose(tr(trace), iats))

    def test_normalized_times(self):
        trace = self._get_trace(self.N_PACKETS)
        tr = self._get_key("times_normalized", trace)

        self._shapes_test(tr, trace)

        times = trace[assets.TIME]
        times -= times.mean()
        times /= times.std()


if __name__ == "__main__":
    unittest.main()
