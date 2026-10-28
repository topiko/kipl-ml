"""Controller semantics, history gradients and serialized-model compatibility."""

import copy
import io
import unittest

import torch

from kipl_ml.defences.models.brick_selection_agent import BrickSelectionAgent
from kipl_ml.trace.enums import Feats


def make_actor(controller, **kwargs):
    return BrickSelectionAgent(
        time_step_s=.1, n_time_steps=2, n_client_bricks=3, n_server_bricks=4,
        features=(Feats.TIME_BINS, Feats.UP_COUNT, Feats.DOWN_COUNT),
        feature_hidden_size=8, controller=controller, prob_eps=.05, **kwargs,
    )


def observation(time=0., count=0.):
    return {Feats.TIME_BINS: torch.tensor([[time]]),
            Feats.Dt_BINS: torch.ones(1, 1),
            Feats.UP_COUNT: torch.tensor([[count]]),
            Feats.DOWN_COUNT: torch.ones(1, 1)}


class TestBrickControllers(unittest.TestCase):
    def test_table_only_uses_current_brick_including_initial_action(self):
        actor = make_actor("table", learn_values=False)
        self.assertEqual(actor.features, ())
        self.assertEqual(set(dict(actor.named_parameters())),
                         {"client_transition_logits", "server_transition_logits"})
        self.assertEqual(actor.client_transition_logits.shape, (1, 3, 3))
        with torch.no_grad():
            actor.client_transition_logits.normal_()
            actor.server_transition_logits.normal_()
        current = torch.tensor([1])
        first = observation()
        first[Feats.Dt_BINS].zero_()
        initial = actor.act_step(first, current, current, sample=False)
        later = actor.act_step(observation(500., 1000.), current, current, sample=False)
        torch.testing.assert_close(initial[3], later[3])
        different = actor.act_step(first, torch.tensor([2]), current, sample=False)
        self.assertFalse(torch.equal(initial[3], different[3]))

    def test_rnn_history_changes_probabilities_and_receives_gradients(self):
        torch.manual_seed(42)
        actor = make_actor("rnn")
        current = torch.tensor([0])
        past = observation(count=10.)
        past[Feats.UP_COUNT].requires_grad_()
        history = actor.act_step(past, current, current, sample=False)[-1]
        self.assertIsNotNone(history)
        result = actor.act_step(observation(2.), current, current,
                                sample=False, state=history)
        fresh = actor.act_step(observation(2.), current, current, sample=False)
        self.assertFalse(torch.allclose(result[3], fresh[3]))
        result[2].sum().backward()
        self.assertGreater(past[Feats.UP_COUNT].grad.abs().sum().item(), 0.)
        self.assertGreater(actor.client_rnn.weight_hh.grad.abs().sum().item(), 0.)
        self.assertFalse(any("transition" in name
                             for name, _ in actor.named_parameters()))

    def test_rnn_value_head_detaches_both_histories_when_requested(self):
        for detached in (False, True):
            actor = make_actor("rnn", detach_before_value_head=detached)
            result = actor.act_step(observation(count=10.), torch.tensor([0]),
                                    torch.tensor([1]), sample=False)
            result[4].sum().backward()
            self.assertEqual(actor.client_rnn.weight_ih.grad is None, detached)
            self.assertEqual(actor.server_rnn.weight_ih.grad is None, detached)
            self.assertIsNotNone(actor.rnn_value_head.weight.grad)

    def test_rnn_reuses_feature_scaling(self):
        actor = make_actor("rnn")
        actor.features = (Feats.TIME_BINS, Feats.Dt_BINS, Feats.UP_COUNT,
                          Feats.DOWN_COUNT, Feats.SILENCE_FLAG)
        x = observation(99., 9.)
        x[Feats.SILENCE_FLAG] = torch.ones(1, 1)
        scaled = actor._feature_vector(x, actor._action_time_bins(x))
        expected = torch.tensor([[.5, .1, 9., 1., 1.]])
        expected[:, 1:4] = expected[:, 1:4].log1p()
        torch.testing.assert_close(scaled, expected)

    def test_checkpoints_and_legacy_hybrid_without_controller_attribute(self):
        current = torch.tensor([0])
        for kind in ("hybrid", "table", "rnn"):
            actor = make_actor(kind)
            expected = actor.act_step(observation(), current, current, sample=False)
            if kind == "hybrid":
                del actor.controller
            checkpoint = io.BytesIO()
            torch.save(actor, checkpoint)
            checkpoint.seek(0)
            restored = torch.load(checkpoint, weights_only=False)
            clone = copy.deepcopy(restored)
            clone.load_state_dict(actor.state_dict(), strict=True)
            actual = clone.act_step(observation(), current, current, sample=False)
            torch.testing.assert_close(expected[3], actual[3])
            torch.testing.assert_close(expected[4], actual[4])
