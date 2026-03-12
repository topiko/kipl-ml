import unittest

import torch

from kipl_ml.trace.features import Compose, Feats, LogInv, PadOrCutTrace, get_feature_tr
from kipl_ml.trace.params import DOWNLOAD, UPLOAD
from kipl_ml.trace.transforms import _TR


class TestTR(unittest.TestCase):
    N_PACKETS = 100

    def _get_trace(self, n_packets: int = 50):
        n_packets = n_packets + 10
        dirs = torch.ones(n_packets, dtype=torch.float32)
        dirs[::2] = -1

        times = torch.arange(n_packets, dtype=torch.float32)
        sizes = torch.ones_like(times)

        return {Feats.DIRS: dirs, Feats.SIZES: sizes, Feats.TIMES: times}

    def _get_key(self, key: str, trace: dict[Feats, torch.Tensor]) -> _TR:
        tr = get_feature_tr(key, self.N_PACKETS)
        tr.get_shapes(trace)
        return tr

    def _shapes_test(self, tr: _TR, trace: dict[Feats, torch.Tensor]):
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
        self._simple_test(Feats.DIRS)

    def test_sizes(self):
        self._simple_test(Feats.SIZES)

    def test_times(self):
        self._simple_test(Feats.TIMES)

    def test_iats(self):
        trace = self._get_trace(self.N_PACKETS)
        tr = self._get_key(Feats.IATS, trace)

        self._shapes_test(tr, trace)

        iats = torch.zeros_like(trace[Feats.TIMES])
        iats[1:] = torch.diff(trace[Feats.TIMES], dim=0)

        self.assertTrue(torch.allclose(tr(trace)[Feats.IATS], iats[: self.N_PACKETS]))

    def test_ud_packets(self):
        trace = self._get_trace(self.N_PACKETS)
        tr_up = self._get_key(Feats.UP_PACKETS, trace)
        tr_down = self._get_key(Feats.DOWN_PACKETS, trace)

        self._shapes_test(tr_up, trace)
        self._shapes_test(tr_down, trace)

        self.assertTrue(
            torch.allclose(
                tr_up(trace)[Feats.UP_PACKETS],
                (trace[Feats.DIRS][: self.N_PACKETS] == UPLOAD).float(),
            )
        )
        self.assertTrue(
            torch.allclose(
                tr_down(trace)[Feats.DOWN_PACKETS],
                (trace[Feats.DIRS][: self.N_PACKETS] == DOWNLOAD).float(),
            )
        )

    def test_ud_iats(self):
        trace = self._get_trace(self.N_PACKETS)
        tr_up = self._get_key(Feats.UP_IATS, trace)
        tr_down = self._get_key(Feats.DOWN_IATS, trace)

        self._shapes_test(tr_up, trace)
        self._shapes_test(tr_down, trace)

        mask = trace[Feats.DIRS] == UPLOAD
        idxs = torch.where(mask)[0]
        iats = torch.zeros_like(trace[Feats.TIMES])
        iats[idxs[1:]] = torch.diff(trace[Feats.TIMES][mask], dim=0)
        iats = iats[: self.N_PACKETS]

        self.assertTrue(torch.allclose(tr_up(trace)[Feats.UP_IATS], iats))

        mask = trace[Feats.DIRS] == DOWNLOAD
        idxs = torch.where(mask)[0]
        iats = torch.zeros_like(trace[Feats.TIMES])
        iats[idxs[1:]] = torch.diff(trace[Feats.TIMES][mask], dim=0)
        iats = iats[: self.N_PACKETS]

        self.assertTrue(torch.allclose(tr_down(trace)[Feats.DOWN_IATS], iats))

    def test_normalize_iats(self):
        trace = self._get_trace(self.N_PACKETS)
        tr = self._get_key(Feats.IATS_NORMALIZED, trace)

        self._shapes_test(tr, trace)

        iats = torch.zeros_like(trace[Feats.TIMES])[: self.N_PACKETS]
        iats[1:] = torch.diff(trace[Feats.TIMES][: self.N_PACKETS], dim=0)

        iats -= iats.mean()
        iats /= iats.std()

        self.assertTrue(torch.allclose(tr(trace)[Feats.IATS_NORMALIZED], iats))

    def test_normalized_times(self):
        trace = self._get_trace(self.N_PACKETS)
        tr = self._get_key(Feats.TIMES_NORMALIZED, trace)

        self._shapes_test(tr, trace)

        times = trace[Feats.TIMES][: self.N_PACKETS]
        times -= times.mean()
        times /= times.std()

        self.assertTrue(torch.allclose(tr(trace)[Feats.TIMES_NORMALIZED], times))

    def test_time_dirs(self):
        trace = self._get_trace(self.N_PACKETS)
        tr = self._get_key(Feats.TIME_DIRS, trace)

        self._shapes_test(tr, trace)

        time_dirs = (
            trace[Feats.TIMES][: self.N_PACKETS] * trace[Feats.DIRS][: self.N_PACKETS]
        )

        self.assertTrue(torch.allclose(tr(trace)[Feats.TIME_DIRS], time_dirs))

    def test_iat_dirs(self):
        trace = self._get_trace(self.N_PACKETS)
        tr = self._get_key(Feats.IAT_DIRS, trace)

        self._shapes_test(tr, trace)

        iats = torch.zeros_like(trace[Feats.TIMES])[: self.N_PACKETS]
        iats[1:] = torch.diff(trace[Feats.TIMES][: self.N_PACKETS], dim=0) + 1.0
        iats[0] = 1.0

        dirs = trace[Feats.DIRS][: self.N_PACKETS]

        iat_dirs = iats * dirs

        self.assertTrue(torch.allclose(tr(trace)[Feats.IAT_DIRS], iat_dirs))

    def test_normalized_iat_dirs(self):
        trace = self._get_trace(self.N_PACKETS)
        tr = self._get_key(Feats.IAT_DIRS_NORMALIZED, trace)

        self._shapes_test(tr, trace)

        iats = torch.zeros_like(trace[Feats.TIMES])[: self.N_PACKETS]
        iats[1:] = torch.diff(trace[Feats.TIMES][: self.N_PACKETS], dim=0)
        iats -= iats.mean()
        iats /= iats.std()

        iats += 1.0

        dirs = trace[Feats.DIRS][: self.N_PACKETS]

        iat_dirs = iats * dirs

        self.assertTrue(torch.allclose(tr(trace)[Feats.IAT_DIRS_NORMALIZED], iat_dirs))

    def test_max_normalized_iats(self):
        trace = self._get_trace(self.N_PACKETS)
        tr = self._get_key(Feats.IATS_MAX_NORMALIZED, trace)

        self._shapes_test(tr, trace)

        iats = torch.zeros_like(trace[Feats.TIMES])[: self.N_PACKETS]
        iats[1:] = torch.diff(trace[Feats.TIMES][: self.N_PACKETS], dim=0)

        iats -= iats.mean()
        iats /= torch.absolute(iats).max()

        self.assertTrue(torch.allclose(tr(trace)[Feats.IATS_MAX_NORMALIZED], iats))

    def test_cum_sizes(self):
        trace = self._get_trace(self.N_PACKETS)
        tr = self._get_key(Feats.CUM_SIZES, trace)

        self._shapes_test(tr, trace)

        sizes = trace[Feats.SIZES][: self.N_PACKETS]

        cum_sizes = torch.cumsum(sizes, dim=0)

        self.assertTrue(torch.allclose(tr(trace)[Feats.CUM_SIZES], cum_sizes))

    def test_max_normalized_cum_sizes(self):
        trace = self._get_trace(self.N_PACKETS)
        tr = self._get_key(Feats.CUM_SIZES_MAX_NORMALIZED, trace)

        self._shapes_test(tr, trace)

        sizes = trace[Feats.SIZES][: self.N_PACKETS]

        cum_sizes = torch.cumsum(sizes, dim=0)
        cum_sizes -= cum_sizes.mean()
        cum_sizes /= torch.absolute(cum_sizes).max()

        cum_sizes_tr = tr(trace)[Feats.CUM_SIZES_MAX_NORMALIZED]

        self.assertTrue(torch.allclose(cum_sizes, cum_sizes_tr))

    def test_burst_edges(self):
        trace = self._get_trace(self.N_PACKETS)
        tr = self._get_key(Feats.BURST_EDGES, trace)

        self._shapes_test(tr, trace)

        dirs = trace[Feats.DIRS][: self.N_PACKETS]
        edges = torch.diff(dirs, dim=0, prepend=torch.Tensor([0]))

        self.assertTrue(torch.allclose(tr(trace)[Feats.BURST_EDGES], edges))

    def test_burst_lens_dirs(self):
        burst_lens = torch.randint(1, 11, (21,))
        dirs = []
        for i, bl in enumerate(burst_lens):
            dirs.append(torch.ones(bl, dtype=torch.float32) * (-1) ** i)

        dirs = torch.cat(dirs)
        trace = {Feats.DIRS: dirs}

        tr = self._get_key(Feats.BURST_LENS, trace)

        burst_lens_ = tr(trace)[Feats.BURST_LENS].long()

        self.assertTrue(torch.allclose(burst_lens_, burst_lens))

        tr = self._get_key(Feats.BURST_DIRS, trace)

        burst_dirs = tr(trace)[Feats.BURST_DIRS]
        burst_dirs_ = torch.ones_like(burst_lens)
        burst_dirs_[1::2] = -1

        self.assertTrue(torch.all(burst_dirs.long() == burst_dirs_))

    def test_normalized_flow_iats(self):
        trace = self._get_trace(self.N_PACKETS)
        tr = self._get_key(Feats.FLOW_IATS_NORMALIZED, trace)

        self._shapes_test(tr, trace)

        flow_iats_tr = tr(trace)[Feats.FLOW_IATS_NORMALIZED]

        dirs = trace[Feats.DIRS][: self.N_PACKETS]
        up_idxs = torch.where(dirs == UPLOAD)[0]
        down_idxs = torch.where(dirs == DOWNLOAD)[0]

        flow_iats = torch.zeros_like(dirs)

        flow_iats[up_idxs[1:]] = torch.diff(trace[Feats.TIMES][up_idxs], dim=0)
        flow_iats[down_idxs[1:]] = torch.diff(trace[Feats.TIMES][down_idxs], dim=0)

        flow_iats -= flow_iats.mean()
        flow_iats /= flow_iats.std()

        self.assertTrue(torch.allclose(flow_iats_tr, flow_iats))

    def test_log_inv(self):
        trace = self._get_trace(self.N_PACKETS)

        tr = Compose(PadOrCutTrace(100), LogInv(Feats.TIMES))

        times = trace[Feats.TIMES][: self.N_PACKETS]

        log_times = torch.log(torch.nan_to_num(1 / times + 1, posinf=1e4))

        self.assertTrue(torch.allclose(tr(trace)[f"log_inv_{Feats.TIMES}"], log_times))

    def test_log_inv_iat_dirs(self):
        trace = self._get_trace(self.N_PACKETS)

        tr = self._get_key(Feats.LOG_INV_FLOW_IATS_NORMALIZED_DIRS, trace)

        tr(trace)[Feats.LOG_INV_FLOW_IATS_NORMALIZED_DIRS]

    def test_running_rate(self):
        trace = self._get_trace(self.N_PACKETS)
        tr = self._get_key(Feats.RUNNING_RATE_SIZES, trace)

        rsizes = tr(trace)[Feats.RUNNING_RATE_SIZES]

        sizes = trace[Feats.SIZES][: self.N_PACKETS]
        times = trace[Feats.TIMES][: self.N_PACKETS]
        sizes[0] = 0.0
        times[0] = 1.0

        rsizes_true = torch.cumsum(sizes, dim=0) / times

        self.assertTrue(torch.allclose(rsizes, rsizes_true))

    def test_size_dirs(self):
        trace = self._get_trace(self.N_PACKETS)
        tr = self._get_key(Feats.SIZE_DIRS, trace)

        self._shapes_test(tr, trace)

        sizes = trace[Feats.SIZES][: self.N_PACKETS]
        dirs = trace[Feats.DIRS][: self.N_PACKETS]

        size_dirs = sizes * dirs

        self.assertTrue(torch.allclose(tr(trace)[Feats.SIZE_DIRS], size_dirs))


if __name__ == "__main__":
    unittest.main()
