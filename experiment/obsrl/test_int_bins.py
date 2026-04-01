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

        fd = get_window_feature_dict(
            X, dt=0.5, max_silence_s=1.0, features=[Feats.TIME_BINS]
        )

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

    def test_delay_uses_dedicated_duration_head(self):
        """Delay duration should come from delay head, not Dt_BINS."""
        from kipl_ml.models.trgen import AGENT1

        obs = AGENT1(
            time_step=0.02,
            max_silence_s=0.02,
            hsize=16,
            nlayers=1,
            prob_eps=0.0,
            enable_delay=True,
            delay_duration_bins=[1, 2, 4, 8],
        )
        obs.eval()

        # Force selector -> DELAY (index 4).
        sel_lin = obs.actor["action_selection"][-1]
        with torch.no_grad():
            sel_lin.weight.zero_()
            sel_lin.bias.zero_()
            sel_lin.bias[4] = 10.0

        # Force delay duration head -> index 2 => 4 bins.
        delay_lin = obs.actor["delay_dur"][-1]
        with torch.no_grad():
            delay_lin.weight.zero_()
            delay_lin.bias.zero_()
            delay_lin.bias[2] = 10.0

        x = {
            Feats.UP_COUNT: torch.tensor([[0]]),
            Feats.DOWN_COUNT: torch.tensor([[0]]),
            Feats.Dt_BINS: torch.tensor([[7]]),
            Feats.TIME_BINS: torch.tensor([[10]]),
            Feats.SILENCE_FLAG: torch.tensor([[1.0]]),
        }

        _, actions, *_ = obs.act_step(x, sample=False)

        self.assertEqual(int(actions[Actions.DELAY_BINS][0, 0].item()), 4)
        self.assertNotEqual(int(actions[Actions.DELAY_BINS][0, 0].item()), 7)

    def test_get_agent_uses_configurable_send_bins(self):
        """Sisyphus agent builder should honor send bins from config."""
        from omegaconf import OmegaConf

        from experiment.obsrl.sisyphus import get_agent_and_critic

        cfg = OmegaConf.create(
            {
                "obs": {
                    "prob_eps": 0.0,
                    "time_step_s": 0.02,
                    "max_silence_s": 0.02,
                    "hsize": 16,
                    "nhidden": 1,
                    "init_for_wait": False,
                    "send_mode": "fixed",
                    "enable_delay": True,
                    "delay_duration_bins": [1, 2, 4],
                    "send_count_bins": [3, 7, 11],
                    "send_after_bins": [0, 2, 5],
                    "separate_critic": False,
                },
                "trace": {"trim_beginning": 0},
            }
        )

        obs, critic = get_agent_and_critic(cfg)

        self.assertIsNone(critic)
        self.assertEqual(obs.send_count_bins.tolist(), [3, 7, 11])
        self.assertEqual(obs.send_after_bins.tolist(), [0, 2, 5])

    def test_tam_counts_follow_integer_bin_mapping(self):
        """TAM bin counts should match _time_to_bin_idx on boundary-ish values."""
        from kipl_ml.trace.features import TAM_BINS, TAM_UP
        from kipl_ml.utils.time import _time_to_bin_idx

        dt = 0.02
        times = torch.tensor(
            [
                6.21999979019165,
                6.21999979019165,
                6.239999771118164,
                6.239999771118164,
                6.239999771118164,
                6.239999771118164,
                6.259999752044678,
            ],
            dtype=torch.float32,
        )
        dirs = torch.ones_like(times, dtype=torch.int8)
        trace = {
            Feats.TIMES: times,
            Feats.DIRS: dirs,
            Feats.PADDING: torch.zeros_like(times, dtype=torch.bool),
        }

        tam_up = TAM_UP(max_load_time_s=10.0, window_width_s=dt)
        got = tam_up(trace)[Feats.TAM_UP_COUNTS].round().to(torch.long)

        b = _time_to_bin_idx(times, dt)
        exp = torch.zeros_like(got, dtype=torch.long)
        exp.scatter_add_(0, b, torch.ones_like(b, dtype=torch.long))

        self.assertTrue(torch.equal(got, exp))

        tam_bins = TAM_BINS(
            max_load_time_s=10.0,
            window_width_s=dt,
            prune_empty_bins=True,
        )
        got_bins = tam_bins(trace)[Feats.TAM_BINS].to(torch.long)
        self.assertEqual(got_bins.tolist(), sorted(set(int(v) for v in b.tolist())))


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
