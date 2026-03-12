import unittest

import torch

from kipl_ml.data.utils import DOWNLOAD, UPLOAD
from kipl_ml.rl.action import TraceExecState, execute_actions_from_sequence
from kipl_ml.rl.enums import Actions
from kipl_ml.rl.observation import WindowFeatureStreamer, get_window_feature_dict
from kipl_ml.rl.utils import (
    _boundary_time_to_bin_idx,
    _duration_to_bin_offsets,
    _time_to_bin_idx,
    fill_after_seq_end,
)
from kipl_ml.trace.enums import Feats


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
        keep_mask = torch.tensor([[True, True, False, False], [True, False, False, False]])
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


class TestWindowFeatureDict(unittest.TestCase):
    DT = 0.02
    MAX_SILENCE_S = 0.1

    def test_basic_trace(self):
        times = torch.tensor([[0.0, 0.02, 0.04, 0.06, 0.08]])
        dirs = torch.tensor([[UPLOAD, DOWNLOAD, UPLOAD, DOWNLOAD, UPLOAD]])
        padding = torch.zeros_like(dirs)

        X = {Feats.TIMES: times, Feats.DIRS: dirs, Feats.PADDING: padding}
        features = [Feats.TIMES, Feats.Dt, Feats.UP_COUNT, Feats.DOWN_COUNT]

        fd = get_window_feature_dict(X, self.DT, self.MAX_SILENCE_S, features)

        self.assertEqual(fd[Feats.TIMES].dtype, torch.long)
        self.assertEqual(fd[Feats.Dt].dtype, torch.long)
        self.assertEqual(fd[Feats.UP_COUNT].dtype, torch.long)
        self.assertEqual(fd[Feats.DOWN_COUNT].dtype, torch.long)

        self.assertTrue((fd[Feats.TIMES] >= 0).any())
        self.assertTrue((fd[Feats.Dt] >= 0).any())

    def test_int_bins_sentinel(self):
        times = torch.tensor([[0.0, 0.02, 0.04]])
        dirs = torch.tensor([[UPLOAD, DOWNLOAD, UPLOAD]])
        padding = torch.zeros_like(dirs)

        X = {Feats.TIMES: times, Feats.DIRS: dirs, Feats.PADDING: padding}
        features = [Feats.TIMES, Feats.Dt]

        fd = get_window_feature_dict(X, self.DT, self.MAX_SILENCE_S, features)

        valid_mask = fd[Feats.TIMES] >= 0
        self.assertTrue(valid_mask.any())
        invalid_mask = fd[Feats.TIMES] < 0
        if invalid_mask.any():
            self.assertTrue((fd[Feats.TIMES][invalid_mask] == -1).all())


class TestWindowFeatureStreamer(unittest.TestCase):
    DT = 0.02
    MAX_SILENCE_S = 0.1

    def test_basic_streaming(self):
        times = torch.tensor([[0.0, 0.02, 0.04, 0.06, 0.08]])
        dirs = torch.tensor([[UPLOAD, DOWNLOAD, UPLOAD, DOWNLOAD, UPLOAD]])
        padding = torch.zeros_like(dirs)

        X = {Feats.TIMES: times, Feats.DIRS: dirs, Feats.PADDING: padding}
        features = [Feats.TIMES, Feats.Dt, Feats.UP_COUNT, Feats.DOWN_COUNT]

        streamer = WindowFeatureStreamer(X, self.DT, self.MAX_SILENCE_S, features)

        steps = []
        for _ in range(100):
            fd_t = streamer.step()
            steps.append(fd_t)
            if streamer.done.all():
                break

        self.assertTrue(streamer.done.all())
        self.assertTrue(len(steps) > 0)

        for fd_t in steps:
            self.assertEqual(fd_t[Feats.TIMES].dtype, torch.long)
            self.assertEqual(fd_t[Feats.Dt].dtype, torch.long)
            self.assertEqual(fd_t[Feats.UP_COUNT].dtype, torch.long)
            self.assertEqual(fd_t[Feats.DOWN_COUNT].dtype, torch.long)

    def test_matches_single_pass(self):
        times = torch.tensor([[0.0, 0.02, 0.04, 0.06, 0.08]])
        dirs = torch.tensor([[UPLOAD, DOWNLOAD, UPLOAD, DOWNLOAD, UPLOAD]])
        padding = torch.zeros_like(dirs)

        X = {Feats.TIMES: times, Feats.DIRS: dirs, Feats.PADDING: padding}
        features = [Feats.TIMES, Feats.Dt, Feats.UP_COUNT, Feats.DOWN_COUNT]

        fd_full = get_window_feature_dict(
            {k: v.clone() for k, v in X.items()}, self.DT, self.MAX_SILENCE_S, features
        )

        streamer = WindowFeatureStreamer(X, self.DT, self.MAX_SILENCE_S, features)

        steps = {f: [] for f in features}
        for _ in range(100):
            fd_t = streamer.step()
            for f in features:
                steps[f].append(fd_t[f])
            if streamer.done.all():
                break

        fd_stream = {f: torch.cat(steps[f], dim=1) for f in features}

        seq_lens_full = fd_full[Feats.SEQ_LENS]
        seq_lens_stream = (fd_stream[Feats.TIMES] >= 0).sum(dim=1)
        self.assertTrue(torch.equal(seq_lens_stream, seq_lens_full))


class TestTraceExecState(unittest.TestCase):
    DT = 0.02

    def test_basic_execution(self):
        times = torch.tensor([[0.0, 0.02, 0.04, 0.06, 0.08]])
        dirs = torch.tensor([[UPLOAD, DOWNLOAD, UPLOAD, DOWNLOAD, UPLOAD]])
        padding = torch.zeros_like(dirs)

        X = {Feats.TIMES: times, Feats.DIRS: dirs, Feats.PADDING: padding}

        exec_state = TraceExecState(
            {k: v.clone() for k, v in X.items()}, time_step_s=self.DT
        )

        X_obs = exec_state.finalize()

        self.assertEqual(X_obs[Feats.TIMES].shape, times.shape)
        self.assertEqual(X_obs[Feats.DIRS].shape, dirs.shape)
        self.assertTrue(torch.isfinite(X_obs[Feats.TIMES]).all())

    def test_delay_execution(self):
        times = torch.tensor([[0.0, 0.02, 0.04, 0.06, 0.08]])
        dirs = torch.tensor([[UPLOAD, DOWNLOAD, UPLOAD, DOWNLOAD, UPLOAD]])
        padding = torch.zeros_like(dirs)

        X = {Feats.TIMES: times, Feats.DIRS: dirs, Feats.PADDING: padding}

        exec_state = TraceExecState(
            {k: v.clone() for k, v in X.items()}, time_step_s=self.DT
        )

        act_bins = torch.tensor([[1]])
        actions = {
            Actions.DELAY: torch.tensor([[1]]),
            Actions.SEND_COUNT_UP: torch.tensor([[0]]),
            Actions.SEND_COUNT_DOWN: torch.tensor([[0]]),
            Actions.SEND_UP_AFTER_TIME: torch.tensor([[0]]),
            Actions.SEND_DOWN_AFTER_TIME: torch.tensor([[0]]),
        }

        exec_state.step(trace_idx=torch.tensor([0]), times=act_bins, actions=actions)

        X_obs = exec_state.finalize()

        self.assertTrue(torch.isfinite(X_obs[Feats.TIMES]).all())

    def test_send_padding(self):
        times = torch.tensor([[0.0, 0.02, 0.04]])
        dirs = torch.tensor([[UPLOAD, DOWNLOAD, UPLOAD]])
        padding = torch.zeros_like(dirs)

        X = {Feats.TIMES: times, Feats.DIRS: dirs, Feats.PADDING: padding}

        exec_state = TraceExecState(
            {k: v.clone() for k, v in X.items()}, time_step_s=self.DT
        )

        act_bins = torch.tensor([[1]])
        actions = {
            Actions.SEND_COUNT_UP: torch.tensor([[2]]),
            Actions.SEND_UP_AFTER_TIME: torch.tensor([[1]]),
            Actions.SEND_COUNT_DOWN: torch.tensor([[0]]),
            Actions.SEND_DOWN_AFTER_TIME: torch.tensor([[0]]),
        }

        exec_state.step(trace_idx=torch.tensor([0]), times=act_bins, actions=actions)

        X_obs = exec_state.finalize()

        n_padding = X_obs[Feats.PADDING].sum().item()
        self.assertEqual(n_padding, 2)


class TestExecuteActionsFromSequence(unittest.TestCase):
    DT = 0.02

    def test_no_actions(self):
        times = torch.tensor([[0.0, 0.02, 0.04, 0.06, 0.08]])
        dirs = torch.tensor([[UPLOAD, DOWNLOAD, UPLOAD, DOWNLOAD, UPLOAD]])
        padding = torch.zeros_like(dirs)

        X = {Feats.TIMES: times, Feats.DIRS: dirs, Feats.PADDING: padding}

        act_times = torch.tensor([[0, 1, 2, 3, 4]], dtype=torch.long)
        actions = {
            Actions.SEND_COUNT_UP: torch.zeros((1, 5), dtype=torch.long),
            Actions.SEND_COUNT_DOWN: torch.zeros((1, 5), dtype=torch.long),
            Actions.SEND_UP_AFTER_TIME: torch.zeros((1, 5), dtype=torch.long),
            Actions.SEND_DOWN_AFTER_TIME: torch.zeros((1, 5), dtype=torch.long),
        }

        X_obs = execute_actions_from_sequence(X, act_times, actions, self.DT)

        self.assertTrue(torch.allclose(X_obs[Feats.TIMES], times, atol=1e-6))
        self.assertTrue(torch.equal(X_obs[Feats.DIRS], dirs))

    def test_delay_actions(self):
        times = torch.tensor([[0.0, 0.02, 0.04, 0.06, 0.08]])
        dirs = torch.tensor([[UPLOAD, DOWNLOAD, UPLOAD, DOWNLOAD, UPLOAD]])
        padding = torch.zeros_like(dirs)

        X = {Feats.TIMES: times, Feats.DIRS: dirs, Feats.PADDING: padding}

        act_times = torch.tensor([[0, 1, 2, 3, 4]], dtype=torch.long)
        actions = {
            Actions.DELAY: torch.tensor([[1, 0, 0, 0, 0]], dtype=torch.long),
            Actions.SEND_COUNT_UP: torch.zeros((1, 5), dtype=torch.long),
            Actions.SEND_COUNT_DOWN: torch.zeros((1, 5), dtype=torch.long),
            Actions.SEND_UP_AFTER_TIME: torch.zeros((1, 5), dtype=torch.long),
            Actions.SEND_DOWN_AFTER_TIME: torch.zeros((1, 5), dtype=torch.long),
        }

        X_obs = execute_actions_from_sequence(X, act_times, actions, self.DT)

        self.assertTrue(torch.isfinite(X_obs[Feats.TIMES]).all())


class TestSinglePassRollout(unittest.TestCase):
    DT = 0.02

    def test_single_pass_no_delay(self):
        from kipl_ml.models.trgen import AGENT1
        from kipl_ml.rl.simulate import policy_rollout_single_pass

        times = torch.tensor([[0.0, 0.02, 0.04, 0.06, 0.08]])
        dirs = torch.tensor([[UPLOAD, DOWNLOAD, UPLOAD, DOWNLOAD, UPLOAD]])
        padding = torch.zeros_like(dirs)

        X = {Feats.TIMES: times, Feats.DIRS: dirs, Feats.PADDING: padding}

        obs = AGENT1(
            time_step=self.DT,
            max_silence_s=0.1,
            enable_delay=False,
            send_mode="fixed",
            prob_eps=0.0,
        )
        obs.eval()

        with torch.no_grad():
            fd, act_times, actions, _, _, _, _, X_obs = policy_rollout_single_pass(
                obs, X, sample=False, extend_end_s=0.0
            )

        self.assertTrue(torch.isfinite(X_obs[Feats.TIMES]).all())
        self.assertEqual(fd[Feats.TIMES].dtype, torch.long)
        self.assertEqual(fd[Feats.Dt].dtype, torch.long)
        self.assertTrue((fd[Feats.TIMES] >= 0).any())

    def test_single_pass_matches_execute_actions(self):
        from kipl_ml.models.trgen import AGENT1
        from kipl_ml.rl.simulate import policy_rollout_single_pass

        times = torch.tensor([[0.0, 0.02, 0.04, 0.06, 0.08]])
        dirs = torch.tensor([[UPLOAD, DOWNLOAD, UPLOAD, DOWNLOAD, UPLOAD]])
        padding = torch.zeros_like(dirs)

        X = {Feats.TIMES: times, Feats.DIRS: dirs, Feats.PADDING: padding}

        obs = AGENT1(
            time_step=self.DT,
            max_silence_s=0.1,
            enable_delay=False,
            send_mode="fixed",
            prob_eps=0.0,
        )
        obs.eval()

        with torch.no_grad():
            fd, act_times, actions, _, _, _, _, X_obs = policy_rollout_single_pass(
                obs, X, sample=False, extend_end_s=0.0
            )

        valid_mask = act_times >= 0
        if valid_mask.any():
            X_obs2 = execute_actions_from_sequence(
                X, act_times, actions, time_step_s=self.DT
            )
            self.assertTrue(torch.allclose(X_obs[Feats.TIMES], X_obs2[Feats.TIMES], atol=1e-6))


if __name__ == "__main__":
    unittest.main()
