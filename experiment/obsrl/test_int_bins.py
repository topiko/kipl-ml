import torch
from unittest import TestCase

from kipl_ml.rl.enums import Actions
from kipl_ml.trace.enums import Feats


class TestObsrlIntBinsConsistency(TestCase):
    """Verify obsrl code uses int bins consistently after refactor."""

    def test_fd_uses_time_bins_not_times(self):
        """fd feature dict should contain TIME_BINS, not TIMES."""
        from kipl_ml.rl.observation import get_window_feature_dict

        times = torch.tensor([[0.0, 0.5, 1.0, 0.0, 0.0]])
        dirs = torch.tensor([[1, -1, 1, 0, 0]])  # 1=UPLOAD, -1=DOWNLOAD
        padding = torch.tensor([[0, 0, 0, 0, 0]], dtype=torch.bool)
        X = {Feats.TIMES: times, Feats.DIRS: dirs, Feats.PADDING: padding}

        fd = get_window_feature_dict(X, dt=0.5, max_silence_s=1.0, features=[Feats.TIME_BINS])

        self.assertIn(Feats.TIME_BINS, fd)
        self.assertNotIn(Feats.TIMES, fd)

    def test_actions_use_delay_bins_not_delay(self):
        """Actions enum should have DELAY_BINS, not DELAY."""
        self.assertTrue(hasattr(Actions, "DELAY_BINS"))
        self.assertFalse(hasattr(Actions, "DELAY"))

    def test_rollout_returns_time_bins(self):
        """Streaming rollout should return fd with TIME_BINS."""
        from kipl_ml.models.trgen import AGENT1
        from kipl_ml.rl.simulate import policy_rollout_streaming

        times = torch.tensor([[0.0, 0.5, 1.0, 1.5, 2.0]])
        dirs = torch.tensor([[1, -1, 1, -1, 1]])
        X = {Feats.TIMES: times, Feats.DIRS: dirs}

        obs = AGENT1(
            time_step=0.5,
            max_silence_s=1.0,
            hsize=32,
            nlayers=1,
            prob_eps=0.0,
            enable_delay=True,
        )
        obs.eval()

        fd, *_ = policy_rollout_streaming(obs, X, sample=False)

        self.assertIn(Feats.TIME_BINS, fd)
        self.assertNotIn(Feats.TIMES, fd)


class TestInvariants(TestCase):
    """Verify invariants code uses correct feature names."""

    def test_get_fd_bins_uses_time_bins(self):
        """_get_fd_bins_and_counts should read from TIME_BINS."""
        from experiment.obsrl.invariants import _get_fd_bins_and_counts

        time_bins = torch.tensor([[0, 2, 5, -1, -1]], dtype=torch.long)
        up_counts = torch.tensor([[1, 0, 1, 0, 0]], dtype=torch.long)
        down_counts = torch.tensor([[0, 1, 0, 0, 0]], dtype=torch.long)
        fd = {
            Feats.TIME_BINS: time_bins,
            Feats.UP_COUNT: up_counts,
            Feats.DOWN_COUNT: down_counts,
        }

        w_t, w_bins, up, down = _get_fd_bins_and_counts(fd, idx=0, dt_s=0.5)

        self.assertEqual(w_bins.tolist(), [0, 2, 5])
        self.assertEqual(up.tolist(), [1, 0, 1])
        self.assertEqual(down.tolist(), [0, 1, 0])


if __name__ == "__main__":
    import unittest
    unittest.main()
