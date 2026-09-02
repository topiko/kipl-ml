import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from kipl_ml.data.utils import DOWNLOAD, UPLOAD
from kipl_ml.rl.enums import (
    Actions,
    ActSelector,
    ActSendDown,
    ActSendUp,
    NoAction,
    StepAction,
)
from kipl_ml.rl.lego import (
    load_lego_brick_specs_from_defense_files,
    load_lego_bricks_from_defense_files,
    to_backend_brick_specs,
)
from kipl_ml.rl.simulate import LegoBatchController, NumpyTrace
from kipl_ml.rl.streaming import WindowFeatureStreamer
from kipl_ml.rl.utils import fill_after_seq_end
from kipl_ml.trace.enums import Feats
from kipl_ml.utils.time import (
    _boundary_time_to_bin_idx,
    _duration_to_bin_offsets,
    _time_to_bin_idx,
)


def _batch_packet_history(
    packet_hist: list[dict[Feats, list[torch.Tensor]]],
    refs: dict[Feats, torch.Tensor],
) -> dict[Feats, torch.Tensor]:
    out: dict[Feats, torch.Tensor] = {}
    bs = len(packet_hist)
    for feat in (Feats.TIMES, Feats.DIRS, Feats.DECOY):
        per_trace: list[torch.Tensor] = []
        max_len = 0
        for trace_hist in packet_hist:
            if trace_hist[feat]:
                t = torch.cat(trace_hist[feat], dim=0)
            else:
                t = torch.zeros((0,), dtype=refs[feat].dtype, device=refs[feat].device)
            per_trace.append(t)
            max_len = max(max_len, int(t.numel()))

        batched = torch.full(
            (bs, max_len),
            -1 if feat == Feats.TIMES else False if feat == Feats.DECOY else 0,
            dtype=refs[feat].dtype,
            device=refs[feat].device,
        )
        for i, t in enumerate(per_trace):
            if t.numel() > 0:
                batched[i, : t.numel()] = t
        out[feat] = batched

    return out


def run_streaming_trace(
    X: dict[Feats, torch.Tensor],
    dt: float,
    max_silence_s: float,
    features: list[Feats],
    actions: list[list[StepAction]] | None = None,
    max_steps: int = 1000,
) -> dict[Feats, torch.Tensor]:
    streamer = WindowFeatureStreamer(X, dt, max_silence_s, features)
    bs = int(X[Feats.TIMES].shape[0])
    packet_hist: list[dict[Feats, list[torch.Tensor]]] = [
        {Feats.TIMES: [], Feats.DIRS: [], Feats.DECOY: []} for _ in range(bs)
    ]

    next_actions: list[StepAction] = (
        actions[0]
        if actions and len(actions) > 0
        else [NoAction(time=0) for _ in range(bs)]
    )
    step_i = 0

    while not streamer.done.all():
        fd_t, fd_packet_level, active = streamer.step(next_actions)
        if active.sum() == 0:
            break

        active_idxs = torch.nonzero(active, as_tuple=False).flatten().tolist()
        for i, aidx in enumerate(active_idxs):
            for feat in (Feats.TIMES, Feats.DIRS, Feats.DECOY):
                packet_hist[aidx][feat].append(fd_packet_level[feat][i])

        step_i += 1
        if step_i >= max_steps:
            raise AssertionError("streaming trace helper exceeded max_steps")

        if actions is not None and step_i < len(actions):
            next_actions = actions[step_i]
        else:
            next_actions = [
                NoAction(
                    time=max(
                        0,
                        int(
                            fd_t[Feats.TIME_BINS][aidx, 0].item()
                            + fd_t[Feats.Dt_BINS][aidx, 0].item()
                        ),
                    )
                )
                for aidx in active_idxs
            ]

    return _batch_packet_history(packet_hist, refs=X)


def assert_trace_equal(
    actual: dict[Feats, torch.Tensor], expected: dict[Feats, torch.Tensor]
) -> None:
    for feat in (Feats.TIMES, Feats.DIRS, Feats.DECOY):
        assert torch.equal(actual[feat], expected[feat]), (
            f"Mismatch for {feat}: actual={actual[feat]} expected={expected[feat]}"
        )


class TestBinConversion(unittest.TestCase):
    DT = 0.02

    def test_time_to_bin_idx_basic(self):
        times = torch.tensor([0.0, 0.02, 0.04, 0.06, 0.08])
        bins = _time_to_bin_idx(times, self.DT)
        expected = torch.tensor([0, 1, 2, 3, 4])
        self.assertTrue(torch.equal(bins, expected))

    def test_time_to_bin_idx_fractional(self):
        times = torch.tensor([0.01, 0.019, 0.021, 0.039])
        bins = _time_to_bin_idx(times, self.DT)
        expected = torch.tensor([0, 0, 1, 1])
        self.assertTrue(torch.equal(bins, expected))

    def test_boundary_time_to_bin_idx(self):
        times = torch.tensor([0.0, 0.02, 0.04])
        bins = _boundary_time_to_bin_idx(times, self.DT)
        expected = torch.tensor([0, 1, 2])
        self.assertTrue(torch.equal(bins, expected))

    def test_duration_to_bin_offsets(self):
        durations = torch.tensor([0.0, 0.02, 0.04, 0.06])
        offsets = _duration_to_bin_offsets(durations, self.DT)
        expected = torch.tensor([0, 1, 2, 3])
        self.assertTrue(torch.equal(offsets, expected))

    def test_duration_to_bin_offsets_rounds(self):
        durations = torch.tensor([0.019, 0.021])
        offsets = _duration_to_bin_offsets(durations, self.DT)
        expected = torch.tensor([1, 1])
        self.assertTrue(torch.equal(offsets, expected))


class TestFillAfterSeqEnd(unittest.TestCase):
    def test_fill_nan(self):
        values = torch.tensor([[1.0, 2.0, 0.0, 0.0], [3.0, 0.0, 0.0, 0.0]])
        keep_mask = torch.tensor(
            [[True, True, False, False], [True, False, False, False]]
        )
        result = fill_after_seq_end(values.clone(), keep_mask, fill_val="nan")
        self.assertTrue(torch.isnan(result[0, 2:]).all())
        self.assertTrue(torch.isnan(result[1, 1:]).all())
        self.assertTrue(torch.equal(result[0, :2], torch.tensor([1.0, 2.0])))
        self.assertTrue(torch.equal(result[1, 0], torch.tensor(3.0)))

    def test_fill_last(self):
        values = torch.tensor([[1.0, 2.0, 0.0, 0.0]])
        keep_mask = torch.tensor([[True, True, False, False]])
        result = fill_after_seq_end(values.clone(), keep_mask, fill_val="last")
        expected = torch.tensor([[1.0, 2.0, 2.0, 2.0]])
        self.assertTrue(torch.equal(result, expected))

    def test_fill_max(self):
        values = torch.tensor([[1.0, 5.0, 0.0, 0.0]])
        keep_mask = torch.tensor([[True, True, False, False]])
        result = fill_after_seq_end(values.clone(), keep_mask, fill_val="max")
        expected = torch.tensor([[1.0, 5.0, 5.0, 5.0]])
        self.assertTrue(torch.equal(result, expected))


class FakeLegoBatch:
    def __init__(self) -> None:
        self.calls: list[tuple[list[int], list[int]]] = []

    def step(
        self, client_brick_selectors: list[int], server_brick_selectors: list[int]
    ) -> list[NumpyTrace]:
        self.calls.append((client_brick_selectors, server_brick_selectors))
        return []

    def is_done(self) -> list[bool]:
        return []


class TestLegoBatchController(unittest.TestCase):
    def test_dense_selectors_keep_current_on_missing_selector(self):
        batch = FakeLegoBatch()
        controller = LegoBatchController(batch, batch_size=3)

        out = controller.step(
            [
                StepAction(0, {Actions.SELECTOR: ActSelector(selected=2)}),
                NoAction(time=0),
            ],
            np.array([True, True, False]),
        )

        self.assertEqual(out, [])
        self.assertEqual(batch.calls[-1], ([2, 0, -1], [2, 0, -1]))

        controller.step(
            [
                NoAction(time=1),
                StepAction(1, {Actions.SELECTOR: ActSelector(selected=1)}),
            ],
            np.array([True, True, False]),
        )

        self.assertEqual(batch.calls[-1], ([2, 1, -1], [2, 1, -1]))

    def test_side_specific_selectors_are_independent(self):
        batch = FakeLegoBatch()
        controller = LegoBatchController(batch, batch_size=2)

        controller.step(
            [
                StepAction(
                    0,
                    {
                        Actions.CLIENT_BRICK_SELECT: ActSelector(selected=2),
                        Actions.SERVER_BRICK_SELECT: ActSelector(selected=5),
                    },
                ),
                NoAction(time=0),
            ],
            np.array([True, True]),
        )

        self.assertEqual(batch.calls[-1], ([2, 0], [5, 0]))

        controller.step(
            [
                StepAction(1, {Actions.SERVER_BRICK_SELECT: ActSelector(selected=3)}),
                StepAction(1, {Actions.CLIENT_BRICK_SELECT: ActSelector(selected=4)}),
            ],
            np.array([True, True]),
        )

        self.assertEqual(batch.calls[-1], ([2, 4], [3, 0]))

    def test_side_specific_selectors_override_shared_selector(self):
        batch = FakeLegoBatch()
        controller = LegoBatchController(batch, batch_size=1)

        controller.step(
            [
                StepAction(
                    0,
                    {
                        Actions.SELECTOR: ActSelector(selected=1),
                        Actions.CLIENT_BRICK_SELECT: ActSelector(selected=2),
                    },
                )
            ],
            np.array([True]),
        )

        self.assertEqual(batch.calls[-1], ([2], [1]))

    def test_rejects_active_mask_shape_mismatch(self):
        controller = LegoBatchController(FakeLegoBatch(), batch_size=2)

        with self.assertRaisesRegex(ValueError, "active_mask must have shape"):
            controller.step([], np.array([True]))

    def test_rejects_action_count_mismatch(self):
        controller = LegoBatchController(FakeLegoBatch(), batch_size=2)

        with self.assertRaisesRegex(ValueError, "step_actions length"):
            controller.step([], np.array([True, False]))


class TestLegoBricks(unittest.TestCase):
    def test_lego_brick_metadata_and_backend_conversion(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "7" / "defense.def"
            path.parent.mkdir()
            path.write_text(
                "example defense\n"
                + json.dumps({"client": ["client-a"], "server": ["server-a"]})
                + "\n"
            )

            client_bricks, server_bricks = load_lego_bricks_from_defense_files([path])
            client_specs, server_specs = to_backend_brick_specs(
                client_bricks, server_bricks
            )

        self.assertEqual(client_bricks[1].name, "7:c0")
        self.assertEqual(client_bricks[1].kind, "compiled_machines")
        self.assertEqual(client_bricks[1].side, "client")
        self.assertEqual(client_bricks[1].tags, ("atlas", "brick"))
        self.assertEqual(client_specs[1]["kind"], "machines")
        self.assertEqual(client_specs[1]["machines"], ["client-a"])
        self.assertEqual(server_specs[1]["machines"], ["server-a"])

    def test_loads_aligned_brick_specs_from_defense_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "7" / "defense.def"
            path.parent.mkdir()
            path.write_text(
                "example defense\n"
                + json.dumps(
                    {"client": ["client-a"], "server": ["server-a", "server-b"]}
                )
                + "\n"
            )

            load_specs = load_lego_brick_specs_from_defense_files
            client_specs, server_specs = load_specs([path])

        self.assertEqual(client_specs[0], {"kind": "noop"})
        self.assertEqual(server_specs[0], {"kind": "noop"})
        self.assertEqual(client_specs[1]["kind"], "machines")
        self.assertEqual(server_specs[1]["kind"], "machines")
        self.assertEqual(client_specs[1]["machines"], ["client-a"])
        self.assertEqual(server_specs[1]["machines"], ["server-a"])
        self.assertEqual(client_specs[1]["name"], "7:c0")
        self.assertEqual(server_specs[1]["name"], "7:s0")
        self.assertEqual(client_specs[2], {"kind": "noop"})
        self.assertEqual(server_specs[2]["machines"], ["server-b"])
        self.assertEqual(server_specs[2]["name"], "7:s1")

    def test_default_brick_specs_are_independent_copies(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "0" / "defense.def"
            path.parent.mkdir()
            path.write_text(
                "example defense\n"
                + json.dumps({"client": ["client-a"], "server": ["server-a"]})
                + "\n"
            )
            load_specs = load_lego_brick_specs_from_defense_files
            client_specs, server_specs = load_specs([path])

        self.assertIsNot(client_specs, server_specs)
        client_specs[0]["kind"] = "changed"
        self.assertEqual(server_specs[0]["kind"], "noop")

class TestWindowFeatureStreamer(unittest.TestCase):
    DT = 0.02
    MAX_SILENCE_S = 0.1

    def test_basic_streaming(self):
        times = torch.tensor([[0.0, 0.02, 0.04, 0.06, 0.08]])
        dirs = torch.tensor([[UPLOAD, DOWNLOAD, UPLOAD, DOWNLOAD, UPLOAD]])
        decoy = torch.zeros_like(dirs, dtype=torch.bool)

        X = {Feats.TIMES: times, Feats.DIRS: dirs, Feats.DECOY: decoy}
        out = run_streaming_trace(
            X,
            self.DT,
            self.MAX_SILENCE_S,
            [Feats.TIME_BINS, Feats.Dt_BINS, Feats.UP_COUNT, Feats.DOWN_COUNT],
        )

        assert_trace_equal(out, X)
        self.assertEqual(out[Feats.TIMES].dtype, torch.float32)
        self.assertEqual(out[Feats.DIRS].dtype, dirs.dtype)
        self.assertEqual(out[Feats.DECOY].dtype, decoy.dtype)

    def test_batched_streaming_round_trip(self):
        times = torch.tensor(
            [
                [0.0, 0.02, 0.04, -1.0, -1.0],
                [0.01, 0.03, 0.05, -1.0, -1.0],
            ]
        )
        dirs = torch.tensor(
            [
                [UPLOAD, DOWNLOAD, UPLOAD, 0, 0],
                [DOWNLOAD, UPLOAD, DOWNLOAD, 0, 0],
            ],
            dtype=torch.float32,
        )
        decoy = torch.zeros_like(times, dtype=torch.bool)

        X = {Feats.TIMES: times, Feats.DIRS: dirs, Feats.DECOY: decoy}
        out = run_streaming_trace(
            X,
            self.DT,
            self.MAX_SILENCE_S,
            [Feats.TIME_BINS, Feats.Dt_BINS, Feats.UP_COUNT, Feats.DOWN_COUNT],
        )

        expected = {
            Feats.TIMES: torch.tensor([[0.0, 0.02, 0.04], [0.01, 0.03, 0.05]]),
            Feats.DIRS: torch.tensor(
                [[UPLOAD, DOWNLOAD, UPLOAD], [DOWNLOAD, UPLOAD, DOWNLOAD]],
                dtype=torch.float32,
            ),
            Feats.DECOY: torch.zeros((2, 3), dtype=torch.bool),
        }
        assert_trace_equal(out, expected)

    def test_matches_streaming(self):
        times = torch.tensor([[0.0, 0.02, 0.04, 0.06, 0.08]])
        dirs = torch.tensor([[UPLOAD, DOWNLOAD, UPLOAD, DOWNLOAD, UPLOAD]])
        decoy = torch.zeros_like(dirs, dtype=torch.bool)

        X = {Feats.TIMES: times, Feats.DIRS: dirs, Feats.DECOY: decoy}
        out = run_streaming_trace(
            X,
            self.DT,
            self.MAX_SILENCE_S,
            [Feats.TIME_BINS, Feats.Dt_BINS, Feats.UP_COUNT, Feats.DOWN_COUNT],
        )

        seq_lens = (out[Feats.DIRS] != 0).sum(dim=1)
        self.assertEqual(seq_lens.shape, (1,))
        self.assertTrue((seq_lens > 0).all())
        self.assertEqual(seq_lens[0].item(), 5)

    def test_noaction_recovers_original_trace(self):
        times = torch.tensor([[0.0, 0.02, 0.04, 0.06, 0.08]])
        dirs = torch.tensor([[UPLOAD, DOWNLOAD, UPLOAD, DOWNLOAD, UPLOAD]])
        decoy = torch.zeros_like(dirs, dtype=torch.bool)

        X = {Feats.TIMES: times, Feats.DIRS: dirs, Feats.DECOY: decoy}

        out = run_streaming_trace(
            X,
            self.DT,
            self.MAX_SILENCE_S,
            [Feats.TIME_BINS, Feats.Dt_BINS, Feats.UP_COUNT, Feats.DOWN_COUNT],
        )

        assert_trace_equal(out, X)

    def test_send_down_adds_packet(self):
        times = torch.tensor([[0.0, 0.02, 0.04]])
        dirs = torch.tensor([[UPLOAD, DOWNLOAD, UPLOAD]])
        decoy = torch.zeros_like(dirs, dtype=torch.bool)

        X = {Feats.TIMES: times, Feats.DIRS: dirs, Feats.DECOY: decoy}

        out = run_streaming_trace(
            X,
            self.DT,
            self.MAX_SILENCE_S,
            [Feats.TIME_BINS, Feats.Dt_BINS, Feats.UP_COUNT, Feats.DOWN_COUNT],
            actions=[
                [
                    StepAction(
                        time_bin=0,
                        _actions={
                            Actions.SEND_DOWN: ActSendDown(count=1, after_steps=2)
                        },
                    )
                ]
            ],
        )

        expected = {
            Feats.TIMES: torch.tensor([[0.0, 0.02, 0.04, 0.05]]),
            Feats.DIRS: torch.tensor([[UPLOAD, DOWNLOAD, UPLOAD, DOWNLOAD]]),
            Feats.DECOY: torch.tensor([[False, False, False, True]]),
        }
        assert_trace_equal(out, expected)

    def test_send_up_adds_packet(self):
        times = torch.tensor([[0.0, 0.02, 0.04]])
        dirs = torch.tensor([[DOWNLOAD, UPLOAD, DOWNLOAD]])
        decoy = torch.zeros_like(dirs, dtype=torch.bool)

        X = {Feats.TIMES: times, Feats.DIRS: dirs, Feats.DECOY: decoy}

        out = run_streaming_trace(
            X,
            self.DT,
            self.MAX_SILENCE_S,
            [Feats.TIME_BINS, Feats.Dt_BINS, Feats.UP_COUNT, Feats.DOWN_COUNT],
            actions=[
                [
                    StepAction(
                        time_bin=0,
                        _actions={Actions.SEND_UP: ActSendUp(count=1, after_steps=2)},
                    )
                ]
            ],
        )

        expected = {
            Feats.TIMES: torch.tensor([[0.0, 0.02, 0.04, 0.05]]),
            Feats.DIRS: torch.tensor([[DOWNLOAD, UPLOAD, DOWNLOAD, UPLOAD]]),
            Feats.DECOY: torch.tensor([[False, False, False, True]]),
        }
        assert_trace_equal(out, expected)

    def test_send_up_and_down_adds_both_packets(self):
        times = torch.tensor([[0.0, 0.02, 0.04]])
        dirs = torch.tensor([[UPLOAD, DOWNLOAD, UPLOAD]])
        decoy = torch.zeros_like(dirs, dtype=torch.bool)

        X = {Feats.TIMES: times, Feats.DIRS: dirs, Feats.DECOY: decoy}

        out = run_streaming_trace(
            X,
            self.DT,
            self.MAX_SILENCE_S,
            [Feats.TIME_BINS, Feats.Dt_BINS, Feats.UP_COUNT, Feats.DOWN_COUNT],
            actions=[
                [
                    StepAction(
                        time_bin=0,
                        _actions={
                            Actions.SEND_UP: ActSendUp(count=1, after_steps=2),
                            Actions.SEND_DOWN: ActSendDown(count=1, after_steps=2),
                        },
                    )
                ]
            ],
        )

        expected = {
            Feats.TIMES: torch.tensor([[0.0, 0.02, 0.04, 0.05, 0.05]]),
            Feats.DIRS: torch.tensor([[UPLOAD, DOWNLOAD, UPLOAD, DOWNLOAD, UPLOAD]]),
            Feats.DECOY: torch.tensor([[False, False, False, True, True]]),
        }
        assert_trace_equal(out, expected)

    def test_streamer_sleep_until_emit(self):
        times = torch.tensor([[0.06]])
        dirs = torch.tensor([[UPLOAD]])
        decoy = torch.tensor([[False]])

        X = {Feats.TIMES: times, Feats.DIRS: dirs, Feats.DECOY: decoy}
        out = run_streaming_trace(
            X,
            self.DT,
            self.MAX_SILENCE_S,
            [Feats.TIME_BINS, Feats.Dt_BINS, Feats.UP_COUNT, Feats.DOWN_COUNT],
        )

        assert_trace_equal(out, X)


class TestVaryingSeqLens(unittest.TestCase):
    DT = 0.02
    MAX_SILENCE_S = 0.1

    def test_streaming_varying_seq_lens_round_trip(self):
        cases = [
            (
                [0.0, 0.02, 0.04, 0.06, 0.08],
                [UPLOAD, DOWNLOAD, UPLOAD, DOWNLOAD, UPLOAD],
            ),
            ([0.0, 0.02, 0.04], [UPLOAD, DOWNLOAD, UPLOAD]),
            ([0.0, 0.02], [UPLOAD, DOWNLOAD]),
        ]

        for times_l, dirs_l in cases:
            times = torch.tensor([times_l])
            dirs = torch.tensor([dirs_l], dtype=torch.float32)
            decoy = torch.zeros_like(dirs, dtype=torch.bool)
            times[0, : len(times_l)] = torch.tensor(times_l)
            dirs[0, : len(dirs_l)] = torch.tensor(dirs_l, dtype=torch.float32)
            X = {Feats.TIMES: times, Feats.DIRS: dirs, Feats.DECOY: decoy}

            out = run_streaming_trace(
                X,
                self.DT,
                self.MAX_SILENCE_S,
                [Feats.TIME_BINS, Feats.Dt_BINS, Feats.UP_COUNT, Feats.DOWN_COUNT],
            )

            expected = {
                Feats.TIMES: torch.tensor([times_l]),
                Feats.DIRS: torch.tensor([dirs_l], dtype=torch.float32),
                Feats.DECOY: torch.zeros((1, len(times_l)), dtype=torch.bool),
            }
            assert_trace_equal(out, expected)

    def test_streaming_varying_seq_lens_counts(self):
        cases = [
            (
                [0.0, 0.02, 0.04, 0.06, 0.08],
                [UPLOAD, DOWNLOAD, UPLOAD, DOWNLOAD, UPLOAD],
            ),
            ([0.0, 0.02, 0.04], [UPLOAD, DOWNLOAD, UPLOAD]),
            ([0.0, 0.02], [UPLOAD, DOWNLOAD]),
        ]

        for times_l, dirs_l in cases:
            times = torch.tensor([times_l])
            dirs = torch.tensor([dirs_l], dtype=torch.float32)
            decoy = torch.zeros_like(dirs, dtype=torch.bool)
            times[0, : len(times_l)] = torch.tensor(times_l)
            dirs[0, : len(dirs_l)] = torch.tensor(dirs_l, dtype=torch.float32)
            X = {Feats.TIMES: times, Feats.DIRS: dirs, Feats.DECOY: decoy}

            out = run_streaming_trace(
                X,
                self.DT,
                self.MAX_SILENCE_S,
                [Feats.TIME_BINS, Feats.Dt_BINS, Feats.UP_COUNT, Feats.DOWN_COUNT],
            )

            seq_lens = (out[Feats.DIRS] != 0).sum(dim=1)
            self.assertEqual(seq_lens.shape, (1,))
            self.assertEqual(seq_lens[0].item(), len(times_l))


if __name__ == "__main__":
    unittest.main()
