from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import torch

from kipl_ml.defences.models.brick_selection_agent import BrickSelectionAgent
from kipl_ml.defences.nndefs import BrickSelectionDef
from kipl_ml.trace.enums import Feats


class TestBrickSelectionDef(unittest.TestCase):
    def test_replays_brick_policy_as_defence(self) -> None:
        policy = BrickSelectionAgent(
            time_step_s=0.1,
            n_time_steps=2,
            n_client_bricks=1,
            n_server_bricks=1,
        )
        defence = BrickSelectionDef(
            policy,
            client_bricks=[{}],
            server_bricks=[{}],
            n_packets=10,
            max_dur_s=1.0,
            seed=123,
            sample=False,
            relative=False,
            max_steps=7,
        )
        network_context = {"rtt": 0}
        defence.load_data = Mock(
            return_value={Feats.DIRS: torch.tensor([1.0, 0.0, -1.0])}
        )
        trace_batch = {
            Feats.TIMES: torch.tensor([[0.1, 0.2]]),
            Feats.DIRS: torch.tensor([[1.0, -1.0]]),
            Feats.DECOY: torch.tensor([[False, True]]),
        }

        with patch(
            "kipl_ml.defences.nndefs.brick_policy_rollout",
            return_value=(None, None, None, None, None, None, None, trace_batch),
        ) as rollout:
            trace = defence(
                Path("trace.log"),
                trim_raw=3,
                network_context=network_context,
            )

        kwargs = rollout.call_args.kwargs
        self.assertIs(kwargs["policy"], policy)
        self.assertEqual(kwargs["trace_paths"], ["trace.log"])
        self.assertEqual(kwargs["max_packets"], 10)
        self.assertEqual(kwargs["max_duration_s"], 1.0)
        self.assertEqual(kwargs["required_real_packets"], 2)
        self.assertEqual(kwargs["trim_raw"], 3)
        self.assertEqual(kwargs["seed"], 123)
        self.assertFalse(kwargs["sample"])
        self.assertFalse(kwargs["relative"])
        self.assertEqual(kwargs["max_steps"], 7)
        torch.testing.assert_close(
            trace[Feats.TIMES], trace_batch[Feats.TIMES].squeeze(0)
        )
        torch.testing.assert_close(
            trace[Feats.DIRS], trace_batch[Feats.DIRS].squeeze(0)
        )
        self.assertTrue(
            torch.equal(trace[Feats.DECOY], trace_batch[Feats.DECOY].squeeze(0))
        )
        torch.testing.assert_close(trace[Feats.SIZES], torch.ones(2))

    def test_rejects_non_brick_selection_agent(self) -> None:
        with self.assertRaisesRegex(TypeError, "BrickSelectionAgent"):
            BrickSelectionDef(
                torch.nn.Linear(1, 1),
                client_bricks=[{}],
                server_bricks=[{}],
            )


if __name__ == "__main__":
    unittest.main()
