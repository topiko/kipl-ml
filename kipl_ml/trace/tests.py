import unittest

import kipl_ml.data.assets as assets
import torch
from kipl_ml.trace.features import get_feature_tr
from kipl_ml.trace.transforms import _TR


class TestTR(unittest.TestCase):
    N_PACKETS = 100

    def _get_trace(self, n_packets: int = 50):
        n_packets = n_packets + 10
        dirs = torch.ones(n_packets, dtype=torch.float32)
        dirs[::2] = -1

        times = torch.arange(n_packets, dtype=torch.float32)
        sizes = torch.ones_like(times)

        return {assets.DIRS: dirs, assets.SIZES: sizes, assets.TIMES: times}

    def _get_key(self, key: str, trace: dict[str, torch.Tensor]) -> _TR:
        tr = get_feature_tr(key, self.N_PACKETS)
        tr.get_shapes(trace)
        return tr

    def _shapes_test(self, tr: _TR, trace: dict[str, torch.Tensor]):
        for k, v in tr.output_sizes.items():
            self.assertEqual(v, self.N_PACKETS)
            self.assertEqual(tr(trace)[k].shape[0], self.N_PACKETS)

    def _simple_test(self, key: str):
        trace = self._get_trace(self.N_PACKETS)
        tr = self._get_key(key, trace)

        self._shapes_test(tr, trace)

        for _, v in tr(trace).items():
            self.assertTrue(torch.allclose(v, trace[key][: self.N_PACKETS]))

    def test_dirs(self):
        self._simple_test(assets.DIRS)

    def test_sizes(self):
        self._simple_test(assets.SIZES)

    def test_times(self):
        self._simple_test(assets.TIMES)

    def test_iats(self):
        trace = self._get_trace(self.N_PACKETS)
        tr = self._get_key(assets.IATS, trace)

        self._shapes_test(tr, trace)

        iats = torch.zeros_like(trace[assets.TIMES])
        iats[1:] = torch.diff(trace[assets.TIMES], dim=0)

        self.assertTrue(torch.allclose(tr(trace)[assets.IATS], iats[: self.N_PACKETS]))

    def test_ud_packets(self):
        trace = self._get_trace(self.N_PACKETS)
        tr_up = self._get_key(assets.UP_PACKETS, trace)
        tr_down = self._get_key(assets.DOWN_PACKETS, trace)

        self._shapes_test(tr_up, trace)
        self._shapes_test(tr_down, trace)

        self.assertTrue(
            torch.allclose(
                tr_up(trace)[assets.UP_PACKETS],
                (trace[assets.DIRS][: self.N_PACKETS] == 1).float(),
            )
        )
        self.assertTrue(
            torch.allclose(
                tr_down(trace)[assets.DOWN_PACKETS],
                (trace[assets.DIRS][: self.N_PACKETS] == -1).float(),
            )
        )

    def test_ud_iats(self):
        trace = self._get_trace(self.N_PACKETS)
        tr_up = self._get_key(assets.UP_IATS, trace)
        tr_down = self._get_key(assets.DOWN_IATS, trace)

        self._shapes_test(tr_up, trace)
        self._shapes_test(tr_down, trace)

        mask = trace[assets.DIRS] == 1
        idxs = torch.where(mask)[0]
        iats = torch.zeros_like(trace[assets.TIMES])
        iats[idxs[1:]] = torch.diff(trace[assets.TIMES][mask], dim=0)
        iats = iats[: self.N_PACKETS]

        self.assertTrue(torch.allclose(tr_up(trace)[assets.UP_IATS], iats))

        mask = trace[assets.DIRS] == -1
        idxs = torch.where(mask)[0]
        iats = torch.zeros_like(trace[assets.TIMES])
        iats[idxs[1:]] = torch.diff(trace[assets.TIMES][mask], dim=0)
        iats = iats[: self.N_PACKETS]

        self.assertTrue(torch.allclose(tr_down(trace)[assets.DOWN_IATS], iats))

    def test_normalize_iats(self):
        trace = self._get_trace(self.N_PACKETS)
        tr = self._get_key(assets.IATS_NORMALIZED, trace)

        self._shapes_test(tr, trace)

        iats = torch.zeros_like(trace[assets.TIMES])[: self.N_PACKETS]
        iats[1:] = torch.diff(trace[assets.TIMES][: self.N_PACKETS], dim=0)

        iats -= iats.mean()
        iats /= iats.std()

        self.assertTrue(torch.allclose(tr(trace)[assets.IATS_NORMALIZED], iats))

    def test_normalized_times(self):
        trace = self._get_trace(self.N_PACKETS)
        tr = self._get_key(assets.TIMES_NORMALIZED, trace)

        self._shapes_test(tr, trace)

        times = trace[assets.TIMES][: self.N_PACKETS]
        times -= times.mean()
        times /= times.std()

        self.assertTrue(torch.allclose(tr(trace)[assets.TIMES_NORMALIZED], times))

    def test_time_dirs(self):
        trace = self._get_trace(self.N_PACKETS)
        tr = self._get_key(assets.TIME_DIRS, trace)

        self._shapes_test(tr, trace)

        time_dirs = (
            trace[assets.TIMES][: self.N_PACKETS] * trace[assets.DIRS][: self.N_PACKETS]
        )

        self.assertTrue(torch.allclose(tr(trace)[assets.TIME_DIRS], time_dirs))


if __name__ == "__main__":
    unittest.main()
