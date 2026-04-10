import unittest

import torch

from kipl_ml.models.trgen import AGENT1
from kipl_ml.rl.enums import Actions, StepAction, AHKs, EntropyKeys
from kipl_ml.trace.features import Feats


def make_dummy_input(batch_size: int, seq_len: int) -> dict[Feats, torch.Tensor]:
    """Create dummy input features for testing."""
    return {
        Feats.UP_COUNT: torch.zeros(batch_size, seq_len, dtype=torch.long),
        Feats.DOWN_COUNT: torch.zeros(batch_size, seq_len, dtype=torch.long),
        Feats.TIME_BINS: torch.zeros(batch_size, seq_len, dtype=torch.long),
        Feats.Dt_BINS: torch.zeros(batch_size, seq_len, dtype=torch.long),
        Feats.SILENCE_FLAG: torch.zeros(batch_size, seq_len, dtype=torch.bool),
    }


class TestAGENT1Forward(unittest.TestCase):
    def setUp(self):
        self.model = AGENT1(
            hsize=32,
            nlayers=1,
            time_step=0.02,
            max_silence_s=1.0,
            enable_delay=False,
        )
        self.model.eval()

    def test_forward_basic_shapes(self):
        B, L = 2, 3
        x = make_dummy_input(B, L)

        out, h = self.model(x)

        self.assertIsInstance(out, dict)
        self.assertIn(AHKs.ACTION_SELECTION, out)
        self.assertIn(AHKs.SEND_COUNT_U, out)
        self.assertIn(AHKs.SEND_COUNT_D, out)

        self.assertEqual(out[AHKs.ACTION_SELECTION].shape, (B, L, 4))
        self.assertEqual(out[AHKs.SEND_COUNT_U].shape, (B, L, 5))
        self.assertEqual(out[AHKs.SEND_COUNT_D].shape, (B, L, 5))

    def test_forward_with_delay(self):
        model = AGENT1(
            hsize=32,
            nlayers=1,
            time_step=0.02,
            max_silence_s=1.0,
            enable_delay=True,
        )
        model.eval()

        B, L = 2, 3
        x = make_dummy_input(B, L)

        out, h = model(x)

        self.assertEqual(out[AHKs.ACTION_SELECTION].shape, (B, L, 7))
        self.assertIn(AHKs.DELAY_BINS_U, out)


class TestAGENT1Act(unittest.TestCase):
    def setUp(self):
        self.model = AGENT1(
            hsize=32,
            nlayers=1,
            time_step=0.02,
            max_silence_s=1.0,
            enable_delay=False,
        )
        self.model.eval()

    def test_act_requires_L1(self):
        B, L = 2, 3
        x = make_dummy_input(B, L)

        with self.assertRaises(ValueError) as ctx:
            self.model.act(x)

        self.assertIn("L=1", str(ctx.exception))

    def test_act_return_shapes(self):
        B = 4
        x = make_dummy_input(B, 1)

        time_bins, step_actions, log_probs, sel_probs, values, entropies, h = (
            self.model.act(x)
        )

        self.assertEqual(time_bins.shape, (B, 1))
        self.assertIsInstance(step_actions, list)
        self.assertEqual(len(step_actions), B)
        self.assertIsInstance(step_actions[0], StepAction)

        self.assertEqual(log_probs.shape, (B, 1))
        self.assertEqual(sel_probs.shape, (B, 1, 4))
        self.assertEqual(values.shape, (B, 1))

        self.assertIsInstance(entropies, dict)
        self.assertIn(EntropyKeys.SELECTION_ENTROPY, entropies)
        self.assertIn(EntropyKeys.COND_ENTROPY, entropies)

    def test_act_selector_do_nothing(self):
        model = AGENT1(
            hsize=32,
            nlayers=1,
            time_step=0.02,
            max_silence_s=1.0,
            enable_delay=False,
            prob_eps=1.0,
        )
        model.eval()

        x = make_dummy_input(1, 1)

        _, step_actions, _, _, _, _, _ = model.act(x)

        sa = step_actions[0]
        self.assertIsInstance(sa, StepAction)


class TestAGENT1ActStep(unittest.TestCase):
    def setUp(self):
        self.model = AGENT1(
            hsize=32,
            nlayers=1,
            time_step=0.02,
            max_silence_s=1.0,
            enable_delay=False,
        )
        self.model.eval()

    def test_act_step_input_validation(self):
        x = {
            Feats.UP_COUNT: torch.zeros(2, 2, dtype=torch.long),
            Feats.DOWN_COUNT: torch.zeros(2, 2, dtype=torch.long),
            Feats.TIME_BINS: torch.zeros(2, 2, dtype=torch.long),
            Feats.Dt_BINS: torch.zeros(2, 2, dtype=torch.long),
            Feats.SILENCE_FLAG: torch.zeros(2, 2, dtype=torch.bool),
        }

        with self.assertRaises(ValueError) as ctx:
            self.model.act_step(x)

        self.assertIn("(B,1)", str(ctx.exception))

    def test_act_step_return_shapes(self):
        B = 3
        x = make_dummy_input(B, 1)

        time_bins, step_actions, log_probs, sel_probs, values, entropies, h = (
            self.model.act_step(x)
        )

        self.assertEqual(time_bins.shape, (B, 1))
        self.assertEqual(len(step_actions), B)


class TestAGENT1WithDelay(unittest.TestCase):
    def setUp(self):
        self.model = AGENT1(
            hsize=32,
            nlayers=1,
            time_step=0.02,
            max_silence_s=1.0,
            enable_delay=True,
        )
        self.model.eval()

    def test_act_delay_return_shapes(self):
        B = 2
        x = make_dummy_input(B, 1)

        _, step_actions, _, _, _, _, _ = self.model.act(x)

        for sa in step_actions:
            self.assertIsInstance(sa, StepAction)


if __name__ == "__main__":
    unittest.main()
