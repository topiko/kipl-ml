import unittest
import warnings

import torch

from kipl_ml.data.utils import DOWNLOAD, UPLOAD
from kipl_ml.rl.enums import Actions, ActSendDown, ActSendUp, NoAction, StepAction
from kipl_ml.rl.observation import (
    WindowFeatureStreamer,
    get_window_feature_dict,
)
from kipl_ml.rl.utils import fill_after_seq_end
from kipl_ml.trace.enums import Feats
from kipl_ml.utils.time import (
    _boundary_time_to_bin_idx,
    _duration_to_bin_offsets,
    _time_to_bin_idx,
)


class TestBinConversion(unittest.TestCase):
    DT = 0.02

    def test_time_to_bin_idx_basic(self):
        times = torch.tensor([0.0, 0.02, 0.04, 0.06, 0.08])
        bins = _time_to_bin_idx(times, self.DT)
        expected = torch.tensor([0, 1, 2, 3, 4])
        self.assertTrue(torch.equal(bins, expected))

    def test_time_to_bin_idx_fractional(self):
        times = torch.tensor([0.01, 0.019, 0.021, 0.039])
        bins = _time_to_bin_idx(times, self.DT)
        expected = torch.tensor([0, 0, 1, 1])
        self.assertTrue(torch.equal(bins, expected))

    def test_boundary_time_to_bin_idx(self):
        times = torch.tensor([0.0, 0.02, 0.04])
        bins = _boundary_time_to_bin_idx(times, self.DT)
        expected = torch.tensor([0, 1, 2])
        self.assertTrue(torch.equal(bins, expected))

    def test_duration_to_bin_offsets(self):
        durations = torch.tensor([0.0, 0.02, 0.04, 0.06])
        offsets = _duration_to_bin_offsets(durations, self.DT)
        expected = torch.tensor([0, 1, 2, 3])
        self.assertTrue(torch.equal(offsets, expected))

    def test_duration_to_bin_offsets_rounds(self):
        durations = torch.tensor([0.019, 0.021])
        offsets = _duration_to_bin_offsets(durations, self.DT)
        expected = torch.tensor([1, 1])
        self.assertTrue(torch.equal(offsets, expected))


class TestFillAfterSeqEnd(unittest.TestCase):
    def test_fill_nan(self):
        values = torch.tensor([[1.0, 2.0, 0.0, 0.0], [3.0, 0.0, 0.0, 0.0]])
        keep_mask = torch.tensor(
            [[True, True, False, False], [True, False, False, False]]
        )
        result = fill_after_seq_end(values.clone(), keep_mask, fill_val="nan")
        self.assertTrue(torch.isnan(result[0, 2:]).all())
        self.assertTrue(torch.isnan(result[1, 1:]).all())
        self.assertTrue(torch.equal(result[0, :2], torch.tensor([1.0, 2.0])))
        self.assertTrue(torch.equal(result[1, 0], torch.tensor(3.0)))

    def test_fill_last(self):
        values = torch.tensor([[1.0, 2.0, 0.0, 0.0]])
        keep_mask = torch.tensor([[True, True, False, False]])
        result = fill_after_seq_end(values.clone(), keep_mask, fill_val="last")
        expected = torch.tensor([[1.0, 2.0, 2.0, 2.0]])
        self.assertTrue(torch.equal(result, expected))

    def test_fill_max(self):
        values = torch.tensor([[1.0, 5.0, 0.0, 0.0]])
        keep_mask = torch.tensor([[True, True, False, False]])
        result = fill_after_seq_end(values.clone(), keep_mask, fill_val="max")
        expected = torch.tensor([[1.0, 5.0, 5.0, 5.0]])
        self.assertTrue(torch.equal(result, expected))


class TestDeprecatedWindowFeatureDict(unittest.TestCase):
    DT = 0.02
    MAX_SILENCE_S = 0.1

    def test_deprecated_helper_warns_and_returns_bins(self):
        times = torch.tensor([[0.0, 0.02, 0.04, 0.06, 0.08]])
        dirs = torch.tensor([[UPLOAD, DOWNLOAD, UPLOAD, DOWNLOAD, UPLOAD]])
        padding = torch.zeros_like(dirs)

        X = {Feats.TIMES: times, Feats.DIRS: dirs, Feats.PADDING: padding}
        features = [Feats.TIME_BINS, Feats.Dt_BINS, Feats.UP_COUNT, Feats.DOWN_COUNT]

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fd = get_window_feature_dict(X, self.DT, self.MAX_SILENCE_S, features)

        self.assertTrue(any(issubclass(w.category, DeprecationWarning) for w in caught))
        self.assertEqual(fd[Feats.TIME_BINS].dtype, torch.long)
        self.assertEqual(fd[Feats.Dt_BINS].dtype, torch.long)
        self.assertEqual(fd[Feats.UP_COUNT].dtype, torch.long)
        self.assertEqual(fd[Feats.DOWN_COUNT].dtype, torch.long)
        self.assertTrue((fd[Feats.TIME_BINS] >= 0).any())
        self.assertTrue((fd[Feats.Dt_BINS] >= 0).any())


class TestWindowFeatureStreamer(unittest.TestCase):
    DT = 0.02
    MAX_SILENCE_S = 0.1

    def test_basic_streaming(self):
        times = torch.tensor([[0.0, 0.02, 0.04, 0.06, 0.08]])
        dirs = torch.tensor([[UPLOAD, DOWNLOAD, UPLOAD, DOWNLOAD, UPLOAD]])
        padding = torch.zeros_like(dirs)

        X = {Feats.TIMES: times, Feats.DIRS: dirs, Feats.PADDING: padding}
        features = [Feats.TIME_BINS, Feats.Dt_BINS, Feats.UP_COUNT, Feats.DOWN_COUNT]

        streamer = WindowFeatureStreamer(X, self.DT, self.MAX_SILENCE_S, features)

        steps = []
        actions = [NoAction(time=0)]
        for _ in range(100):
            fd_t, _, active = streamer.step(actions)
            steps.append(fd_t)
            if active.sum() == 0:
                break

        self.assertTrue(streamer.done.all())
        self.assertTrue(len(steps) > 0)

        for fd_t in steps:
            self.assertEqual(fd_t[Feats.TIME_BINS].dtype, torch.long)
            self.assertEqual(fd_t[Feats.Dt_BINS].dtype, torch.long)
            self.assertEqual(fd_t[Feats.UP_COUNT].dtype, torch.long)
            self.assertEqual(fd_t[Feats.DOWN_COUNT].dtype, torch.long)

    def test_matches_streaming(self):
        times = torch.tensor([[0.0, 0.02, 0.04, 0.06, 0.08]])
        dirs = torch.tensor([[UPLOAD, DOWNLOAD, UPLOAD, DOWNLOAD, UPLOAD]])
        padding = torch.zeros_like(dirs)

        X = {Feats.TIMES: times, Feats.DIRS: dirs, Feats.PADDING: padding}
        features = [Feats.TIME_BINS, Feats.Dt_BINS, Feats.UP_COUNT, Feats.DOWN_COUNT]

        streamer = WindowFeatureStreamer(X, self.DT, self.MAX_SILENCE_S, features)

        steps = {f: [] for f in features}
        actions = [NoAction(time=0)]
        for _ in range(100):
            fd_t, _, active = streamer.step(actions)
            for f in features:
                steps[f].append(fd_t[f])
            if active.sum() == 0:
                break

        fd_stream = {f: torch.cat(steps[f], dim=1) for f in features}

        seq_lens_stream = (fd_stream[Feats.TIME_BINS] >= 0).sum(dim=1)
        self.assertEqual(seq_lens_stream.shape, (1,))
        self.assertTrue((seq_lens_stream > 0).all())
        self.assertTrue(
            torch.equal(seq_lens_stream, (fd_stream[Feats.Dt_BINS] >= 0).sum(dim=1))
        )

    def test_noaction_recovers_original_trace(self):
        times = torch.tensor([[0.0, 0.02, 0.04, 0.06, 0.08]])
        dirs = torch.tensor([[UPLOAD, DOWNLOAD, UPLOAD, DOWNLOAD, UPLOAD]])
        padding = torch.zeros_like(dirs)

        X = {Feats.TIMES: times, Feats.DIRS: dirs, Feats.PADDING: padding}
        features = [Feats.TIME_BINS, Feats.Dt_BINS, Feats.UP_COUNT, Feats.DOWN_COUNT]

        streamer = WindowFeatureStreamer(X, self.DT, self.MAX_SILENCE_S, features)

        pkt_hist = {f: [] for f in (Feats.TIMES, Feats.DIRS, Feats.PADDING)}
        actions = [NoAction(time=0)]

        for _ in range(100):
            fd_t, fd_packet_level, active = streamer.step(actions)
            if active.sum() == 0:
                break

            for f in pkt_hist:
                pkt_hist[f].append(fd_packet_level[f][0])

            actions = [NoAction(time=int(fd_t[Feats.TIME_BINS][0, 0].item()))]

        recovered = {
            f: torch.cat(pkt_hist[f], dim=0).unsqueeze(0)
            for f in pkt_hist
            if pkt_hist[f]
        }

        self.assertTrue(torch.equal(recovered[Feats.TIMES], times))
        self.assertTrue(torch.equal(recovered[Feats.DIRS], dirs))
        self.assertTrue(torch.equal(recovered[Feats.PADDING], padding))

    def test_send_down_adds_packet(self):
        times = torch.tensor([[0.0, 0.02, 0.04]])
        dirs = torch.tensor([[UPLOAD, DOWNLOAD, UPLOAD]])
        padding = torch.zeros_like(dirs)

        X = {Feats.TIMES: times, Feats.DIRS: dirs, Feats.PADDING: padding}
        features = [Feats.TIME_BINS, Feats.Dt_BINS, Feats.UP_COUNT, Feats.DOWN_COUNT]

        streamer = WindowFeatureStreamer(X, self.DT, self.MAX_SILENCE_S, features)

        send_action = [
            StepAction(
                time=0,
                _actions={Actions.SEND_DOWN: ActSendDown(count=1, after_steps=2)},
            )
        ]

        fd_t0, fd_packet_level0, active0 = streamer.step(send_action)
        self.assertTrue(active0.any())
        self.assertEqual(int(fd_t0[Feats.UP_COUNT][0, 0].item()), 1)
        self.assertEqual(int(fd_t0[Feats.DOWN_COUNT][0, 0].item()), 0)
        self.assertTrue(
            torch.equal(fd_packet_level0[Feats.TIMES][0], torch.tensor([0.0]))
        )
        self.assertTrue(
            torch.equal(fd_packet_level0[Feats.DIRS][0], torch.tensor([UPLOAD]))
        )
        self.assertTrue(
            torch.equal(fd_packet_level0[Feats.PADDING][0], torch.tensor([0.0]))
        )

        fd_t1, _, active1 = streamer.step(
            [NoAction(time=int(fd_t0[Feats.TIME_BINS][0, 0].item()))]
        )
        self.assertTrue(active1.any())
        self.assertEqual(int(fd_t1[Feats.UP_COUNT][0, 0].item()), 0)
        self.assertEqual(int(fd_t1[Feats.DOWN_COUNT][0, 0].item()), 1)

        fd_t2, fd_packet_level2, active2 = streamer.step(
            [NoAction(time=int(fd_t1[Feats.TIME_BINS][0, 0].item()))]
        )
        self.assertTrue(active2.any())
        self.assertEqual(int(fd_t2[Feats.UP_COUNT][0, 0].item()), 1)
        self.assertEqual(int(fd_t2[Feats.DOWN_COUNT][0, 0].item()), 1)
        self.assertTrue(
            torch.equal(fd_packet_level2[Feats.TIMES][0], torch.tensor([0.04, 0.05]))
        )
        self.assertTrue(
            torch.equal(
                fd_packet_level2[Feats.DIRS][0], torch.tensor([UPLOAD, DOWNLOAD])
            )
        )
        self.assertTrue(
            torch.equal(fd_packet_level2[Feats.PADDING][0], torch.tensor([0.0, 1.0]))
        )

    def test_send_up_adds_packet(self):
        times = torch.tensor([[0.0, 0.02, 0.04]])
        dirs = torch.tensor([[DOWNLOAD, UPLOAD, DOWNLOAD]])
        padding = torch.zeros_like(dirs)

        X = {Feats.TIMES: times, Feats.DIRS: dirs, Feats.PADDING: padding}
        features = [Feats.TIME_BINS, Feats.Dt_BINS, Feats.UP_COUNT, Feats.DOWN_COUNT]

        streamer = WindowFeatureStreamer(X, self.DT, self.MAX_SILENCE_S, features)

        send_action = [
            StepAction(
                time=0,
                _actions={Actions.SEND_UP: ActSendUp(count=1, after_steps=2)},
            )
        ]

        fd_t0, fd_packet_level0, active0 = streamer.step(send_action)
        self.assertTrue(active0.any())
        self.assertEqual(int(fd_t0[Feats.UP_COUNT][0, 0].item()), 0)
        self.assertEqual(int(fd_t0[Feats.DOWN_COUNT][0, 0].item()), 1)
        self.assertTrue(
            torch.equal(fd_packet_level0[Feats.TIMES][0], torch.tensor([0.0]))
        )
        self.assertTrue(
            torch.equal(fd_packet_level0[Feats.DIRS][0], torch.tensor([DOWNLOAD]))
        )
        self.assertTrue(
            torch.equal(fd_packet_level0[Feats.PADDING][0], torch.tensor([0.0]))
        )

        fd_t1, _, active1 = streamer.step(
            [NoAction(time=int(fd_t0[Feats.TIME_BINS][0, 0].item()))]
        )
        self.assertTrue(active1.any())
        self.assertEqual(int(fd_t1[Feats.UP_COUNT][0, 0].item()), 1)
        self.assertEqual(int(fd_t1[Feats.DOWN_COUNT][0, 0].item()), 0)

        fd_t2, fd_packet_level2, active2 = streamer.step(
            [NoAction(time=int(fd_t1[Feats.TIME_BINS][0, 0].item()))]
        )
        self.assertTrue(active2.any())
        self.assertEqual(int(fd_t2[Feats.UP_COUNT][0, 0].item()), 1)
        self.assertEqual(int(fd_t2[Feats.DOWN_COUNT][0, 0].item()), 1)
        self.assertTrue(
            torch.equal(fd_packet_level2[Feats.TIMES][0], torch.tensor([0.04, 0.05]))
        )
        self.assertTrue(
            torch.equal(
                fd_packet_level2[Feats.DIRS][0], torch.tensor([DOWNLOAD, UPLOAD])
            )
        )
        self.assertTrue(
            torch.equal(fd_packet_level2[Feats.PADDING][0], torch.tensor([0.0, 1.0]))
        )

    def test_send_up_and_down_adds_both_packets(self):
        times = torch.tensor([[0.0, 0.02, 0.04]])
        dirs = torch.tensor([[UPLOAD, DOWNLOAD, UPLOAD]])
        padding = torch.zeros_like(dirs)

        X = {Feats.TIMES: times, Feats.DIRS: dirs, Feats.PADDING: padding}
        features = [Feats.TIME_BINS, Feats.Dt_BINS, Feats.UP_COUNT, Feats.DOWN_COUNT]

        streamer = WindowFeatureStreamer(X, self.DT, self.MAX_SILENCE_S, features)

        send_action = [
            StepAction(
                time=0,
                _actions={
                    Actions.SEND_UP: ActSendUp(count=1, after_steps=2),
                    Actions.SEND_DOWN: ActSendDown(count=1, after_steps=2),
                },
            )
        ]

        fd_t0, fd_packet_level0, active0 = streamer.step(send_action)
        self.assertTrue(active0.any())
        self.assertEqual(int(fd_t0[Feats.UP_COUNT][0, 0].item()), 1)
        self.assertEqual(int(fd_t0[Feats.DOWN_COUNT][0, 0].item()), 0)
        self.assertTrue(
            torch.equal(fd_packet_level0[Feats.TIMES][0], torch.tensor([0.0]))
        )

        fd_t1, _, active1 = streamer.step(
            [NoAction(time=int(fd_t0[Feats.TIME_BINS][0, 0].item()))]
        )
        self.assertTrue(active1.any())
        self.assertEqual(int(fd_t1[Feats.UP_COUNT][0, 0].item()), 0)
        self.assertEqual(int(fd_t1[Feats.DOWN_COUNT][0, 0].item()), 1)

        fd_t2, fd_packet_level2, active2 = streamer.step(
            [NoAction(time=int(fd_t1[Feats.TIME_BINS][0, 0].item()))]
        )
        self.assertTrue(active2.any())
        self.assertEqual(int(fd_t2[Feats.UP_COUNT][0, 0].item()), 2)
        self.assertEqual(int(fd_t2[Feats.DOWN_COUNT][0, 0].item()), 1)
        self.assertTrue(
            torch.equal(
                fd_packet_level2[Feats.TIMES][0], torch.tensor([0.04, 0.05, 0.05])
            )
        )
        self.assertTrue(
            torch.equal(
                fd_packet_level2[Feats.DIRS][0],
                torch.tensor([UPLOAD, DOWNLOAD, UPLOAD]),
            )
        )
        self.assertTrue(
            torch.equal(
                fd_packet_level2[Feats.PADDING][0], torch.tensor([0.0, 1.0, 1.0])
            )
        )

    def test_streamer_sleep_until_emit(self):
        times = torch.tensor([[0.06]])
        dirs = torch.tensor([[UPLOAD]])
        padding = torch.tensor([[0]])

        X = {Feats.TIMES: times, Feats.DIRS: dirs, Feats.PADDING: padding}
        features = [Feats.TIME_BINS, Feats.Dt_BINS, Feats.UP_COUNT, Feats.DOWN_COUNT]
        streamer = WindowFeatureStreamer(X, self.DT, self.MAX_SILENCE_S, features)

        fd_t, fd_packet_level, active = streamer.step([NoAction(time=0)])

        self.assertTrue(active.any())
        self.assertEqual(int(fd_t[Feats.UP_COUNT][0, 0].item()), 1)
        self.assertTrue(
            torch.equal(fd_packet_level[Feats.TIMES][0], torch.tensor([0.06]))
        )
        self.assertTrue(
            torch.equal(fd_packet_level[Feats.DIRS][0], torch.tensor([UPLOAD]))
        )
        self.assertTrue(
            torch.equal(fd_packet_level[Feats.PADDING][0], torch.tensor([0]))
        )


class TestVaryingSeqLens(unittest.TestCase):
    DT = 0.02
    MAX_SILENCE_S = 0.1

    def test_streaming_varying_seq_lens(self):
        times = torch.tensor(
            [
                [0.0, 0.02, 0.04, 0.06, 0.08],
                [0.0, 0.02, 0.04, 0.0, 0.0],
                [0.0, 0.02, 0.0, 0.0, 0.0],
            ]
        )
        dirs = torch.tensor(
            [
                [UPLOAD, DOWNLOAD, UPLOAD, DOWNLOAD, UPLOAD],
                [UPLOAD, DOWNLOAD, UPLOAD, 0, 0],
                [UPLOAD, DOWNLOAD, 0, 0, 0],
            ],
            dtype=torch.float32,
        )
        padding = torch.zeros_like(dirs)

        X = {Feats.TIMES: times, Feats.DIRS: dirs, Feats.PADDING: padding}
        features = [Feats.TIME_BINS, Feats.Dt_BINS, Feats.UP_COUNT, Feats.DOWN_COUNT]

        streamer = WindowFeatureStreamer(X, self.DT, self.MAX_SILENCE_S, features)

        steps = {f: [] for f in features}
        actions = [NoAction(time=0) for _ in range(X[Feats.TIMES].shape[0])]
        for _ in range(100):
            fd_t, _, active = streamer.step(actions)
            for f in features:
                steps[f].append(fd_t[f])
            if active.sum() == 0:
                break

        fd_stream = {f: torch.cat(steps[f], dim=1) for f in features}

        seq_lens = (fd_stream[Feats.TIME_BINS] >= 0).sum(dim=1)
        self.assertEqual(seq_lens.shape[0], 3)
        self.assertTrue((seq_lens > 0).all())

        for i in range(3):
            valid_count = (fd_stream[Feats.TIME_BINS][i] >= 0).sum().item()
            self.assertEqual(valid_count, seq_lens[i].item())

            if seq_lens[i].item() < fd_stream[Feats.TIME_BINS].shape[1]:
                invalid_start = seq_lens[i].item()
                self.assertTrue(
                    (fd_stream[Feats.TIME_BINS][i, invalid_start:] == -1).all()
                )
                self.assertTrue(
                    (fd_stream[Feats.Dt_BINS][i, invalid_start:] == -1).all()
                )

    def test_streaming_varying_seq_lens(self):
        times = torch.tensor(
            [
                [0.0, 0.02, 0.04, 0.06, 0.08],
                [0.0, 0.02, 0.04, 0.0, 0.0],
                [0.0, 0.02, 0.0, 0.0, 0.0],
            ]
        )
        dirs = torch.tensor(
            [
                [UPLOAD, DOWNLOAD, UPLOAD, DOWNLOAD, UPLOAD],
                [UPLOAD, DOWNLOAD, UPLOAD, 0, 0],
                [UPLOAD, DOWNLOAD, 0, 0, 0],
            ],
            dtype=torch.float32,
        )
        padding = torch.zeros_like(dirs)

        X = {Feats.TIMES: times, Feats.DIRS: dirs, Feats.PADDING: padding}
        features = [Feats.TIME_BINS, Feats.Dt_BINS, Feats.UP_COUNT, Feats.DOWN_COUNT]

        streamer = WindowFeatureStreamer(X, self.DT, self.MAX_SILENCE_S, features)

        steps = {f: [] for f in features}
        max_steps = 100
        actions = [NoAction(time=0) for _ in range(X[Feats.TIMES].shape[0])]
        for _ in range(max_steps):
            fd_t, _, active = streamer.step(actions)
            for f in features:
                steps[f].append(fd_t[f])
            if active.sum() == 0:
                break

        self.assertTrue(streamer.done.all(), "All traces should be done")

        fd_stream = {f: torch.cat(steps[f], dim=1) for f in features}

        seq_lens_stream = (fd_stream[Feats.TIME_BINS] >= 0).sum(dim=1)

        self.assertTrue((seq_lens_stream > 0).all())

        for i in range(3):
            valid_count = (fd_stream[Feats.TIME_BINS][i] >= 0).sum().item()
            self.assertEqual(valid_count, seq_lens_stream[i].item())


if __name__ == "__main__":
    unittest.main()
