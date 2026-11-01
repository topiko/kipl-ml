"""Causal attention, bounded context, value isolation and policy semantics."""

import io
import unittest
from unittest.mock import patch

import torch

from kipl_ml.defences.models.tam_transformer import RoPEEncoderLayer
from kipl_ml.defences.models.test_brick_controllers import make_actor, observation
from kipl_ml.rl.enums import EntropyKeys
from kipl_ml.trace.enums import Feats


class TestBrickTransformer(unittest.TestCase):
    def test_normalized_attention_bounds_logits_and_resists_projection_scaling(self):
        torch.manual_seed(42)
        layer = RoPEEncoderLayer(64, 2, 128, 0., qk_norm=True)
        x = torch.randn(2, 20, 64, requires_grad=True)
        causal = torch.ones(20, 20, dtype=torch.bool).triu(1)
        padding = torch.zeros(2, 20, dtype=torch.bool)
        sdpa = torch.nn.functional.scaled_dot_product_attention
        logits = []

        def capture(query, key, value, **kwargs):
            logits.append(query @ key.transpose(-1, -2) / layer.head_dim ** .5)
            return sdpa(query, key, value, **kwargs)

        with patch("torch.nn.functional.scaled_dot_product_attention", capture):
            expected = layer(x, causal, padding)
            with torch.no_grad():
                layer.qkv.weight[:128].mul_(1000.)
                layer.qkv.bias[:128].mul_(1000.)
            actual = layer(x, causal, padding)
        torch.testing.assert_close(actual, expected)
        for scores in logits:
            self.assertLessEqual(scores.abs().max().item(), layer.head_dim ** .5 + 1e-5)
        actual.square().mean().backward()
        assert x.grad is not None and layer.qkv.weight.grad is not None
        for grad in (x.grad, layer.qkv.weight.grad[:64], layer.qkv.weight.grad[64:128]):
            self.assertTrue(torch.isfinite(grad).all())
            self.assertGreater(grad.abs().sum().item(), 0.)
        # Zero projections must also produce finite outputs and gradients.
        layer.zero_grad()
        with torch.no_grad():
            layer.qkv.weight[:128].zero_()
            layer.qkv.bias[:128].zero_()
        layer(x, causal, padding).sum().backward()
        assert layer.qkv.weight.grad is not None
        self.assertTrue(torch.isfinite(layer.qkv.weight.grad).all())

    def test_legacy_pickles_preserve_unnormalized_attention(self):
        actor = make_actor("transformer", transformer_qk_norm=False)
        history = torch.randn(2, 5, 10)
        expected = actor.transformer.encode(history)
        for layer in actor.transformer.layers:
            del layer.qk_norm  # Attribute absent in original full-model checkpoints.
        serialized = io.BytesIO()
        torch.save(actor, serialized)
        serialized.seek(0)
        restored = torch.load(serialized, weights_only=False)
        torch.testing.assert_close(restored.transformer.encode(history), expected)
        self.assertFalse(RoPEEncoderLayer(16, 2, 32, 0.).qk_norm)

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
