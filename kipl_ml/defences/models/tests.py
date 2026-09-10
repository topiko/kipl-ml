import unittest

import torch

from kipl_ml.defences.models.brick_selection_agent import (
    DEFAULT_BRICK_FEATURES,
    BrickSelectionAgent,
)
from kipl_ml.defences.models.trgen import AGENT1, CRITIC01, RNNDefenceAgent
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


def _brick_context_features(up_counts: list[int]) -> dict[Feats, torch.Tensor]:
    batch_size = len(up_counts)
    return {
        Feats.TIME_BINS: torch.zeros(batch_size, 1),
        Feats.Dt_BINS: torch.ones(batch_size, 1),
        Feats.UP_COUNT: torch.tensor(up_counts).reshape(-1, 1),
        Feats.DOWN_COUNT: torch.zeros(batch_size, 1),
        Feats.UP_DECOY_COUNT: torch.zeros(batch_size, 1),
        Feats.DOWN_DECOY_COUNT: torch.zeros(batch_size, 1),
        Feats.SILENCE_FLAG: (torch.tensor(up_counts) == 0).reshape(-1, 1),
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
    def test_time_step_compatibility_aliases(self) -> None:
        model = BrickSelectionAgent(time_step=0.05, time_steps=3, n_client_bricks=4)

        self.assertEqual(model.time_step_s, 0.05)
        self.assertEqual(model.time_step, model.time_step_s)
        self.assertEqual(model.n_time_steps, 3)
        self.assertEqual(model.time_steps, model.n_time_steps)

    def test_old_brick_artifact_time_step_fallback(self) -> None:
        model = BrickSelectionAgent.__new__(BrickSelectionAgent)
        model.__dict__["time_step"] = 0.05
        model.__dict__["time_steps"] = 3

        self.assertEqual(model.time_step_s, 0.05)
        self.assertEqual(model.n_time_steps, 3)

    def test_transition_probs_are_time_indexed_stochastic_matrices(self) -> None:
        model = BrickSelectionAgent(
            time_step_s=0.05,
            n_time_steps=3,
            n_client_bricks=4,
            n_server_bricks=5,
        )

        self.assertEqual(model.time_step, model.time_step_s)
        self.assertEqual(model.time_steps, model.n_time_steps)
        client_probs, server_probs = model.transition_probs()

        self.assertEqual(client_probs.shape, (3, 4, 4))
        self.assertEqual(server_probs.shape, (3, 5, 5))
        self.assertTrue(torch.allclose(client_probs.sum(dim=-1), torch.ones(3, 4)))
        self.assertTrue(torch.allclose(server_probs.sum(dim=-1), torch.ones(3, 5)))

    def test_act_step_selects_from_time_and_current_brick_rows(self) -> None:
        model = BrickSelectionAgent(
            time_step_s=0.05,
            n_time_steps=4,
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
            time_step_s=0.05,
            n_time_steps=2,
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

    def test_single_time_table_entropy_depends_on_current_brick_not_time(self) -> None:
        model = BrickSelectionAgent(
            time_step_s=0.05,
            n_time_steps=1,
            n_client_bricks=3,
            n_server_bricks=3,
        )
        with torch.no_grad():
            model.client_transition_logits[0, 0] = torch.tensor([4.0, 0.0, 0.0])

        _, _, _, _, _, same_row_entropies, _ = model.act_step(
            _brick_features([0, 100]),
            current_client_bricks=torch.tensor([0, 0]),
            current_server_bricks=torch.tensor([0, 0]),
            sample=False,
        )
        _, _, _, _, _, different_row_entropies, _ = model.act_step(
            _brick_features([0, 0]),
            current_client_bricks=torch.tensor([0, 1]),
            current_server_bricks=torch.tensor([0, 0]),
            sample=False,
        )

        same_row = same_row_entropies[EntropyKeys.SELECTION_ENTROPY].squeeze(1)
        different_rows = different_row_entropies[EntropyKeys.SELECTION_ENTROPY].squeeze(
            1
        )
        torch.testing.assert_close(same_row[0], same_row[1])
        self.assertNotEqual(
            float(different_rows[0].detach()),
            float(different_rows[1].detach()),
        )

    def test_feature_residual_can_condition_policy_and_value(self) -> None:
        features = tuple(_brick_context_features([0]))
        model = BrickSelectionAgent(
            time_step_s=1.0,
            n_time_steps=1,
            n_client_bricks=2,
            n_server_bricks=2,
            features=features,
            feature_hidden_size=2,
        )
        up_idx = features.index(Feats.UP_COUNT)
        with torch.no_grad():
            encoder = model.feature_encoder[0]
            encoder.weight.zero_()
            encoder.bias.zero_()
            encoder.weight[0, up_idx] = 1.0

            client_hidden = model.client_feature_head[0]
            client_hidden.weight.zero_()
            client_hidden.bias.zero_()
            client_hidden.weight[0, 0] = 1.0
            client_output = model.client_feature_head[-1]
            client_output.weight.zero_()
            client_output.bias.zero_()
            client_output.weight[0, 0] = 1.0

            value_hidden = model.value_feature_head[0]
            value_hidden.weight.zero_()
            value_hidden.bias.zero_()
            value_hidden.weight[0, 0] = 1.0
            value_output = model.value_feature_head[-1]
            value_output.weight.zero_()
            value_output.bias.zero_()
            value_output.weight[0, 0] = 1.0

        _, _, _, probs, values, _, _ = model.act_step(
            _brick_context_features([0, 20]),
            current_client_bricks=torch.tensor([0, 0]),
            current_server_bricks=torch.tensor([0, 0]),
            sample=False,
        )

        self.assertNotEqual(
            float(probs[0, 0, 0].detach()),
            float(probs[1, 0, 0].detach()),
        )
        self.assertNotEqual(
            float(values[0, 0].detach()),
            float(values[1, 0].detach()),
        )

    def test_feature_heads_receive_gradients(self) -> None:
        torch.manual_seed(0)
        model = BrickSelectionAgent(
            time_step_s=1.0,
            n_time_steps=1,
            n_client_bricks=2,
            n_server_bricks=2,
            features=tuple(_brick_context_features([0])),
            feature_hidden_size=4,
        )
        current = torch.zeros(2, dtype=torch.long)
        _, _, log_probs, _, values, _, _ = model.act_step(
            _brick_context_features([0, 20]),
            current_client_bricks=current,
            current_server_bricks=current,
            sample=True,
        )

        loss = -(log_probs * torch.tensor([[1.0], [-1.0]])).mean()
        loss = loss + (values - torch.tensor([[0.0], [1.0]])).pow(2).mean()
        loss.backward()

        self.assertGreater(
            float(model.client_feature_head[-1].weight.grad.abs().sum()), 0.0
        )
        self.assertGreater(
            float(model.server_feature_head[-1].weight.grad.abs().sum()), 0.0
        )
        self.assertGreater(
            float(model.value_feature_head[-1].weight.grad.abs().sum()), 0.0
        )

    def test_empty_features_use_static_tables_only(self) -> None:
        model = BrickSelectionAgent(
            time_step_s=1.0,
            n_time_steps=1,
            n_client_bricks=2,
            n_server_bricks=2,
            features=(),
        )
        with torch.no_grad():
            model.client_transition_logits[0, 0] = torch.tensor([3.0, 0.0])

        _, _, _, probs, values, _, _ = model.act_step(
            _brick_features([0, 100]),
            current_client_bricks=torch.tensor([0, 0]),
            current_server_bricks=torch.tensor([0, 0]),
            sample=False,
        )

        self.assertIsNone(model.feature_encoder)
        self.assertIsNone(model.client_feature_head)
        self.assertIsNone(model.server_feature_head)
        self.assertIsNone(model.value_feature_head)
        torch.testing.assert_close(probs[0], probs[1])
        torch.testing.assert_close(values, torch.zeros_like(values))

    def test_external_critic_disables_actor_value_parameters(self) -> None:
        model = BrickSelectionAgent(
            time_step_s=1.0,
            n_time_steps=1,
            n_client_bricks=2,
            features=tuple(_brick_context_features([0])),
            learn_values=False,
        )

        _, _, _, _, values, _, _ = model.act_step(
            _brick_context_features([0]),
            current_client_bricks=torch.tensor([0]),
            current_server_bricks=torch.tensor([0]),
            sample=False,
        )

        self.assertNotIn("value_table", dict(model.named_parameters()))
        self.assertIsNone(model.value_feature_head)
        torch.testing.assert_close(values, torch.zeros_like(values))

    def test_rejects_sequence_inputs(self) -> None:
        model = BrickSelectionAgent(time_step_s=0.05, n_time_steps=2, n_client_bricks=2)

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
        model = BrickSelectionAgent(time_step_s=0.05, n_time_steps=2, n_client_bricks=2)

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


class TestRNNDefenceAgentCompatibility(unittest.TestCase):
    def test_agent1_is_compatibility_alias(self) -> None:
        self.assertIs(AGENT1, RNNDefenceAgent)

    def test_time_step_compatibility_alias(self) -> None:
        model = RNNDefenceAgent(time_step=0.02, max_silence_s=0.04)

        self.assertEqual(model.time_step_s, 0.02)
        self.assertEqual(model.time_step, model.time_step_s)

    def test_old_rnn_artifact_time_step_fallback(self) -> None:
        model = RNNDefenceAgent.__new__(RNNDefenceAgent)
        model.__dict__["time_step"] = 0.02

        self.assertEqual(model.time_step_s, 0.02)


class TestBrickRecurrentCritic(unittest.TestCase):
    def test_brick_one_hot_inputs_preserve_padding_through_detach(self):
        agent = BrickSelectionAgent(
            time_step_s=1.0, n_time_steps=2, n_client_bricks=2, n_server_bricks=3
        )
        critic = CRITIC01(
            agent, hsize=8, nlayers=1, features=[Feats.TIME_BINS], brick_counts=(2, 3)
        )
        x = {
            Feats.TIME_BINS: torch.tensor([[0, 1, 2], [0, -1, -1]]),
            Feats.CURRENT_CLIENT_BRICK: torch.tensor([[0, 1, 1], [1, -1, -1]]),
            Feats.CURRENT_SERVER_BRICK: torch.tensor([[0, 2, 1], [2, -1, -1]]),
        }
        captured = []
        handle = critic.rnn.register_forward_pre_hook(
            lambda _module, args: captured.append(args[0].detach().clone())
        )
        try:
            output, _ = critic(x, h_detach_period=2, seq_lens=torch.tensor([3, 1]))
        finally:
            handle.remove()
        self.assertEqual(critic.rnn.input_size, 6)
        torch.testing.assert_close(
            captured[0][0, :, 1:],
            torch.tensor(
                [
                    [1.0, 0.0, 1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0, 1.0],
                ]
            ),
        )
        torch.testing.assert_close(captured[0][1, 1, 1:], torch.zeros(5))
        torch.testing.assert_close(
            captured[1][0, 0, 1:], torch.tensor([0.0, 1.0, 0.0, 1.0, 0.0])
        )
        values = output[Feats.STATE_VALUE]
        self.assertEqual(values.shape, (2, 3))
        (values[0].sum() + values[1, 0]).backward()
        self.assertGreater(critic.rnn.weight_ih_l0.grad[:, 1:].abs().sum().item(), 0)

    def test_numeric_critic_and_old_artifact_do_not_require_brick_inputs(self):
        agent = RNNDefenceAgent(time_step_s=0.1, max_silence_s=0.2)
        critic = CRITIC01(agent, hsize=8, nlayers=1, features=[Feats.TIME_BINS])
        x = {Feats.TIME_BINS: torch.tensor([[0, 1]])}
        expected, _ = critic(x)
        del critic.brick_counts  # Simulate an artifact saved before this change.
        actual, _ = critic(x)
        torch.testing.assert_close(
            actual[Feats.STATE_VALUE], expected[Feats.STATE_VALUE]
        )
        self.assertEqual(critic.rnn.input_size, 1)

    def test_static_actor_can_use_feature_conditioned_recurrent_critic(self) -> None:
        agent = BrickSelectionAgent(
            time_step_s=1.0,
            n_time_steps=1,
            n_client_bricks=2,
            features=(),
        )
        features = (
            Feats.TIME_BINS,
            Feats.Dt_BINS,
            Feats.UP_COUNT,
            Feats.DOWN_COUNT,
            Feats.UP_DECOY_COUNT,
            Feats.DOWN_DECOY_COUNT,
            Feats.SILENCE_FLAG,
        )
        critic = CRITIC01(
            agent,
            hsize=8,
            nlayers=1,
            features=features,
        )
        x = {feature: torch.zeros(2, 3) for feature in features}
        x[Feats.Dt_BINS].fill_(1)
        x[Feats.UP_COUNT][1].fill_(10)

        output, _ = critic(x)
        values = output[Feats.STATE_VALUE]
        values.sum().backward()

        self.assertEqual(critic.features, list(features))
        self.assertEqual(values.shape, (2, 3))
        self.assertGreater(float(critic.rnn.weight_ih_l0.grad.abs().sum()), 0.0)


class TestRNNDefenceAgentForward(unittest.TestCase):
    def setUp(self):
        self.model = RNNDefenceAgent(
            hsize=32,
            nlayers=1,
            time_step_s=0.02,
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
        model = RNNDefenceAgent(
            hsize=32,
            nlayers=1,
            time_step_s=0.02,
            max_silence_s=1.0,
            enable_delay=True,
        )
        model.eval()

        B, L = 2, 3
        x = make_dummy_input(B, L)

        out, h = model(x)

        self.assertEqual(out[AHKs.ACTION_SELECTION].shape, (B, L, 7))
        self.assertIn(AHKs.DELAY_BINS_U, out)


class TestRNNDefenceAgentAct(unittest.TestCase):
    def setUp(self):
        self.model = RNNDefenceAgent(
            hsize=32,
            nlayers=1,
            time_step_s=0.02,
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
        model = RNNDefenceAgent(
            hsize=32,
            nlayers=1,
            time_step_s=0.02,
            max_silence_s=1.0,
            enable_delay=False,
            prob_eps=1.0,
        )
        model.eval()

        x = make_dummy_input(1, 1)

        _, step_actions, _, _, _, _, _ = model.act(x)

        sa = step_actions[0]
        self.assertIsInstance(sa, StepAction)


class TestRNNDefenceAgentActStep(unittest.TestCase):
    def setUp(self):
        self.model = RNNDefenceAgent(
            hsize=32,
            nlayers=1,
            time_step_s=0.02,
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


class TestRNNDefenceAgentWithDelay(unittest.TestCase):
    def setUp(self):
        self.model = RNNDefenceAgent(
            hsize=32,
            nlayers=1,
            time_step_s=0.02,
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
