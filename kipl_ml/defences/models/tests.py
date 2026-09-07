import unittest

import torch

from kipl_ml.defences.models.brick_selection_agent import (
    DEFAULT_BRICK_FEATURES,
    BrickSelectionAgent,
)
from kipl_ml.defences.models.trgen import AGENT1
from kipl_ml.rl.enums import Actions, AHKs, EntropyKeys, StepAction
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


def _brick_features(
    times: list[int], dts: list[int] | None = None
) -> dict[Feats, torch.Tensor]:
    if dts is None:
        dts = [0] * len(times)
    return {
        Feats.TIME_BINS: torch.tensor(times).reshape(-1, 1),
        Feats.Dt_BINS: torch.tensor(dts).reshape(-1, 1),
    }


def _force_client_transition(
    model: BrickSelectionAgent, time_idx: int, current: int, selected: int
) -> None:
    with torch.no_grad():
        model.client_transition_logits[time_idx, current].zero_()
        model.client_transition_logits[time_idx, current, selected] = 10.0


def _force_server_transition(
    model: BrickSelectionAgent, time_idx: int, current: int, selected: int
) -> None:
    with torch.no_grad():
        model.server_transition_logits[time_idx, current].zero_()
        model.server_transition_logits[time_idx, current, selected] = 10.0


class TestBrickSelectionAgent(unittest.TestCase):
    def test_transition_probs_are_time_indexed_stochastic_matrices(self) -> None:
        model = BrickSelectionAgent(
            time_step=0.05,
            time_steps=3,
            n_client_bricks=4,
            n_server_bricks=5,
        )

        client_probs, server_probs = model.transition_probs()

        self.assertEqual(client_probs.shape, (3, 4, 4))
        self.assertEqual(server_probs.shape, (3, 5, 5))
        self.assertTrue(torch.allclose(client_probs.sum(dim=-1), torch.ones(3, 4)))
        self.assertTrue(torch.allclose(server_probs.sum(dim=-1), torch.ones(3, 5)))

    def test_act_step_selects_from_time_and_current_brick_rows(self) -> None:
        model = BrickSelectionAgent(
            time_step=0.05,
            time_steps=4,
            n_client_bricks=5,
            n_server_bricks=6,
        )
        _force_client_transition(model, time_idx=0, current=0, selected=1)
        _force_client_transition(model, time_idx=1, current=1, selected=3)
        _force_client_transition(model, time_idx=3, current=2, selected=4)
        _force_server_transition(model, time_idx=0, current=0, selected=2)
        _force_server_transition(model, time_idx=1, current=2, selected=4)
        _force_server_transition(model, time_idx=3, current=4, selected=5)

        time_bins, actions, log_ps, sel_probs, values, entropies, h = model.act_step(
            _brick_features([0, 1, 2], dts=[0, 0, 1]),
            current_client_bricks=torch.tensor([0, 1, 2]),
            current_server_bricks=torch.tensor([0, 2, 4]),
            sample=False,
        )

        self.assertIsNone(h)
        self.assertEqual(time_bins.squeeze(1).tolist(), [0, 1, 3])
        self.assertEqual(log_ps.shape, (3, 1))
        self.assertEqual(sel_probs.shape, (3, 1, 11))
        self.assertEqual(values.shape, (3, 1))
        self.assertEqual(entropies[EntropyKeys.SELECTION_ENTROPY].shape, (3, 1))
        self.assertEqual(entropies[EntropyKeys.COND_ENTROPY].shape, (3, 1))
        self.assertEqual(
            [a[Actions.CLIENT_BRICK_SELECT].selected for a in actions],
            [1, 3, 4],
        )
        self.assertEqual(
            [a[Actions.SERVER_BRICK_SELECT].selected for a in actions],
            [2, 4, 5],
        )

    def test_action_time_clamps_to_last_transition_matrix(self) -> None:
        model = BrickSelectionAgent(
            time_step=0.05,
            time_steps=2,
            n_client_bricks=3,
            n_server_bricks=4,
        )
        _force_client_transition(model, time_idx=1, current=0, selected=2)
        _force_server_transition(model, time_idx=1, current=0, selected=3)

        _time_bins, actions, _log_ps, _sel_probs, _values, _entropies, _h = (
            model.act_step(
                _brick_features([100]),
                current_client_bricks=torch.tensor([0]),
                current_server_bricks=torch.tensor([0]),
                sample=False,
            )
        )

        self.assertEqual(actions[0][Actions.CLIENT_BRICK_SELECT].selected, 2)
        self.assertEqual(actions[0][Actions.SERVER_BRICK_SELECT].selected, 3)

    def test_rejects_sequence_inputs(self) -> None:
        model = BrickSelectionAgent(time_step=0.05, time_steps=2, n_client_bricks=2)

        with self.assertRaisesRegex(ValueError, r"\(B,1\)"):
            model.act_step(
                {
                    Feats.TIME_BINS: torch.zeros(2, 2),
                    Feats.Dt_BINS: torch.zeros(2, 2),
                },
                current_client_bricks=torch.tensor([0, 1]),
                current_server_bricks=torch.tensor([0, 1]),
            )

    def test_rejects_invalid_current_bricks(self) -> None:
        model = BrickSelectionAgent(time_step=0.05, time_steps=2, n_client_bricks=2)

        with self.assertRaisesRegex(ValueError, "current_client_bricks"):
            model.act_step(
                _brick_features([0]),
                current_client_bricks=torch.tensor([2]),
                current_server_bricks=torch.tensor([0]),
            )

        with self.assertRaisesRegex(ValueError, "current_server_bricks"):
            model.act_step(
                _brick_features([0]),
                current_client_bricks=torch.tensor([0]),
                current_server_bricks=torch.tensor([2]),
            )

    def test_default_features_are_time_features(self) -> None:
        self.assertEqual(DEFAULT_BRICK_FEATURES, (Feats.TIME_BINS, Feats.Dt_BINS))


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
