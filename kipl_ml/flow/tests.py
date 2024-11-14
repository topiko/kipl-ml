import unittest

import torch
from kipl_ml.data.assets import assets
from kipl_ml.flow.transforms import _TR, get_feature_tr


class TestTR(unittest.TestCase):
    N_PACKETS = 100

    def _get_flow(self, n_packets: int = 50):
        dirs = torch.ones(n_packets, dtype=torch.float32)
        dirs[::2] = -1

        times = torch.arange(n_packets, dtype=torch.float32)
        sizes = torch.ones_like(times)

        return {assets.DIR: dirs, assets.SIZE: sizes, assets.TIME: times}

    def _get_key(self, key: str, flow: dict[str, torch.Tensor]) -> _TR:
        tr = get_feature_tr(key)
        tr.get_shapes(flow)
        return tr

    def _shapes_test(self, tr: _TR, flow: dict[str, torch.Tensor]):
        self.assertEqual(tr.input_size, self.N_PACKETS)
        self.assertEqual(tr.output_size, self.N_PACKETS)

    def _simple_test(self, key: str):
        flow = self._get_flow(self.N_PACKETS)
        tr = self._get_key(key, flow)

        self._shapes_test(tr, flow)

        asset_key = {"dirs": assets.DIR, "sizes": assets.SIZE, "times": assets.TIME}[
            key
        ]

        self.assertTrue(torch.allclose(tr(flow), flow[asset_key]))

    def test_dirs(self):
        self._simple_test("dirs")

    def test_sizes(self):
        self._simple_test("sizes")

    def test_times(self):
        self._simple_test("times")

    def test_iats(self):
        flow = self._get_flow(self.N_PACKETS)
        tr = self._get_key("iats", flow)

        self._shapes_test(tr, flow)

        iats = torch.zeros_like(flow[assets.TIME])
        iats[1:] = torch.diff(flow[assets.TIME], dim=0)

        self.assertTrue(torch.allclose(tr(flow), iats))

    def test_ud_packets(self):
        flow = self._get_flow(self.N_PACKETS)
        tr_up = self._get_key("up_packets", flow)
        tr_down = self._get_key("down_packets", flow)

        self._shapes_test(tr_up, flow)
        self._shapes_test(tr_down, flow)

        self.assertTrue(torch.allclose(tr_up(flow), (flow[assets.DIR] == 1).float()))
        self.assertTrue(torch.allclose(tr_down(flow), (flow[assets.DIR] == -1).float()))

    def test_up_iats(self):
        flow = self._get_flow(self.N_PACKETS)
        tr = self._get_key("up_iats", flow)

        self._shapes_test(tr, flow)

        mask = flow[assets.DIR] > 1
        idxs = torch.where(mask)[0]
        iats = torch.zeros_like(flow[assets.TIME])
        iats[idxs[1:]] = torch.diff(flow[assets.TIME][mask], dim=0)

        self.assertTrue(torch.allclose(tr(flow), iats))


if __name__ == "__main__":
    unittest.main()
