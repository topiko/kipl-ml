"""Recurrent rollout state alignment, reset, bootstrap and truncated BPTT."""

import unittest
from unittest.mock import patch

import torch

from kipl_ml.defences.models.test_brick_controllers import make_actor
from kipl_ml.rl.brick_simulate import brick_policy_rollout
from kipl_ml.rl.rollout_end import RolloutEnd
from kipl_ml.rl.tests import FakeBrickController
from kipl_ml.trace.enums import Feats


class UnevenController(FakeBrickController):
    def __init__(self, lengths, bricks):
        super().__init__(len(lengths))
        self.lengths = lengths
        self.client = list(bricks)
        self.server = list(bricks)

    def is_done(self):
        return [self.step_count >= length for length in self.lengths]


def run(actor, lengths, bricks, *, horizon=10., detach_period=None):
    endpoint = RolloutEnd(len(lengths), torch.device("cpu"))
    result = brick_policy_rollout(
        actor, ["unused"] * len(lengths), "cpu", client_bricks=None,
        server_bricks=None, network_context=None, max_packets=None,
        max_duration_s=horizon, trim_raw=0, seed=0, sample=False, relative=False,
        max_steps=100, controller=UnevenController(lengths, bricks),
        endpoint=endpoint, detach_period=detach_period,
    )
    return result, endpoint


class TestBrickRecurrence(unittest.TestCase):
    controller = "rnn"

    def make_actor(self):
        return make_actor(self.controller, transformer_max_tokens=4)

    def test_shrinking_batch_matches_independent_histories_and_resets(self):
        torch.manual_seed(42)
        actor = self.make_actor()
        lengths, bricks = [6, 1, 4], [0, 1, 2]
        batch, _ = run(actor, lengths, bricks)
        repeated, _ = run(actor, lengths, bricks)
        torch.testing.assert_close(batch[4], repeated[4])
        for row, (length, brick) in enumerate(zip(lengths, bricks, strict=True)):
            single, _ = run(actor, [length], [brick])
            for output in (3, 4, 5):
                torch.testing.assert_close(batch[output][row:row + 1, :length],
                                           single[output])
        batch[3][:, -1].sum().backward()
        trunk = (actor.transformer if self.controller == "transformer"
                 else actor.client_rnn)
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0
                            for p in trunk.parameters()))

    def test_bootstrap_uses_full_history_without_an_extra_executed_action(self):
        actor = self.make_actor()
        result, end = run(actor, [5, 1, 5], [0, 1, 2], horizon=.25)
        self.assertEqual(end.truncated.tolist(), [True, False, True])
        fd = result[0]
        for row in (0, 2):
            state = None
            n = int(fd[Feats.SEQ_LENS][row])
            self.assertEqual(len(result[2][row]), n)
            for step in range(n):
                x = {key: value[row:row + 1, step:step + 1]
                     for key, value in fd.items() if key != Feats.SEQ_LENS}
                state = actor.act_step(
                    x, x[Feats.CURRENT_CLIENT_BRICK].flatten(),
                    x[Feats.CURRENT_SERVER_BRICK].flatten(), sample=False, state=state,
                )[-1]
            x = {key: value[row:row + 1] for key, value in end.features.items()}
            expected = actor.act_step(
                x, x[Feats.CURRENT_CLIENT_BRICK].flatten(),
                x[Feats.CURRENT_SERVER_BRICK].flatten(), sample=False, state=state,
            )[4]
            torch.testing.assert_close(end.values[row], expected.reshape(()))
        self.assertFalse(end.values.requires_grad)

    def test_detach_period_cuts_history_gradients_but_preserves_outputs(self):
        actor = self.make_actor()
        results = []
        for period in (None, 2):
            states = []
            original = actor.act_step

            def capture(*args, **kwargs):
                # Transformer state holds raw tokens: probe history gradients via
                # differentiable observations rather than recurrent parameters.
                args[0][Feats.UP_COUNT].requires_grad_()
                output = original(*args, **kwargs)
                output[-1].retain_grad()
                states.append(output[-1])
                return output

            with patch.object(actor, "act_step", side_effect=capture):
                result, _ = run(actor, [4], [0], detach_period=period)
            result[3][:, -1].sum().backward()
            gradient = states[0].grad
            magnitude = 0. if gradient is None else gradient.abs().sum().item()
            self.assertEqual(magnitude == 0., period is not None)
            results.append(result[4])
            actor.zero_grad(set_to_none=True)
        torch.testing.assert_close(*results)


class TestBrickTransformerHistory(TestBrickRecurrence):
    controller = "transformer"
