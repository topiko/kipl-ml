"""Causal attention, bounded context, value isolation and policy semantics."""

import unittest

import torch

from kipl_ml.defences.models.test_brick_controllers import make_actor, observation
from kipl_ml.rl.enums import EntropyKeys
from kipl_ml.trace.enums import Feats


class TestBrickTransformer(unittest.TestCase):
    def test_causal_encoder_cannot_see_future_tokens_or_other_sides_bricks(self):
        torch.manual_seed(42)
        actor = make_actor("transformer")
        trunk = actor.transformer
        history = torch.randn(2, 5, 3 + 3 + 4)
        expected = trunk.encode(history)
        changed = history.clone()
        changed[:, 3:] += 100.
        actual = trunk.encode(changed)
        torch.testing.assert_close(expected[:, :3], actual[:, :3])
        torch.testing.assert_close(expected[:, :3], trunk.encode(history[:, :3]))
        changed = history.clone()
        changed[..., 6:] += 100.  # Server brick inputs cannot affect client outputs.
        torch.testing.assert_close(expected[:2], trunk.encode(changed)[:2])
        changed = history.clone()
        changed[..., 3:6] += 100.
        torch.testing.assert_close(expected[2:], trunk.encode(changed)[2:])

    def test_rolling_history_matches_fresh_window_and_forgets_evicted_tokens(self):
        actor = make_actor("transformer", transformer_max_tokens=3)
        current = torch.tensor([0])
        history = None
        for step in range(9):
            result = actor.act_step(observation(step, step * 3.), current, current,
                                    sample=False, state=history)
            history = result[-1]
            self.assertEqual(history.shape, (1, min(step + 1, 3), 10))
        # Reconstruct only the retained inputs; no hidden older context may leak in.
        fresh = None
        for step in range(6, 9):
            repeated = actor.act_step(observation(step, step * 3.), current, current,
                                      sample=False, state=fresh)
            fresh = repeated[-1]
        torch.testing.assert_close(result[3], repeated[3])
        torch.testing.assert_close(result[4], repeated[4])
        no_history = actor.act_step(
            observation(8., 24.), current, current, sample=False,
        )
        self.assertFalse(torch.allclose(result[3], no_history[3]))

    def test_history_scaling_gradients_and_value_detachment(self):
        for detach in (False, True):
            actor = make_actor("transformer", detach_before_value_head=detach)
            past = observation(99., 9.)
            past[Feats.UP_COUNT].requires_grad_()
            current = torch.tensor([1])
            history = actor.act_step(past, current, current, sample=False)[-1]
            expected = torch.tensor([.5, 9., 1.])
            expected[1:] = expected[1:].log1p()
            torch.testing.assert_close(history[0, 0, :3], expected)
            result = actor.act_step(observation(100., 2.), current, current,
                                    sample=False, state=history)
            result[4].sum().backward(retain_graph=True)
            self.assertEqual(actor.transformer.client_projection.weight.grad is None,
                             detach)
            self.assertIsNotNone(actor.rnn_value_head.weight.grad)
            actor.zero_grad(set_to_none=True)
            past[Feats.UP_COUNT].grad = None
            result[2].sum().backward()
            self.assertGreater(past[Feats.UP_COUNT].grad.abs().sum().item(), 0.)
            for layer in actor.transformer.layers:
                self.assertGreater(layer.qkv.weight.grad.abs().sum().item(), 0.)

    def test_train_and_replay_match_and_exploration_leaves_entropy_unchanged(self):
        actor = make_actor("transformer")
        current = torch.tensor([1])
        state = actor.act_step(
            observation(count=8.), current, current, sample=False,
        )[-1]
        learned = None
        entropy = None
        for eps in (0., .05, 1.):
            actor.prob_eps = eps
            actor.train()
            result = actor.act_step(observation(1.), current, current,
                                    sample=False, state=state)
            actor.eval()
            with torch.inference_mode():
                replay = actor.act_step(observation(1.), current, current,
                                        sample=False, state=state)
            torch.testing.assert_close(result[3], replay[3])
            if learned is None:
                learned, entropy = result[3], result[5][EntropyKeys.SELECTION_ENTROPY]
            uniform = torch.tensor([1 / 3] * 3 + [1 / 4] * 4).reshape(1, 1, 7)
            torch.testing.assert_close(result[3], (1 - eps) * learned + eps * uniform)
            torch.testing.assert_close(
                result[5][EntropyKeys.SELECTION_ENTROPY], entropy,
            )

    def test_rejects_invalid_attention_dimensions(self):
        for kwargs in ({"transformer_nhead": 3}, {"transformer_nhead": 8},
                       {"transformer_nlayer": 0}, {"transformer_max_tokens": 0}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                make_actor("transformer", **kwargs)
