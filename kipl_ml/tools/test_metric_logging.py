import math
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import mlflow
import pandas as pd
from mlflow.tracking import fluent

from kipl_ml.tools.metric_logging import (
    ALIASES,
    get_metric,
    get_metric_history,
    has_metric,
    internal_metric_name,
    log_metric,
    log_metrics,
    normalize_metrics,
    runs_to_dataframe,
    tracking_metric_name,
    tracking_metrics,
)
from kipl_ml.tools.mlflow_utils import list_runs


class TestMetricNames(unittest.TestCase):
    def test_aliases_round_trip_and_keep_existing_diagnostic_names(self):
        pairs = {
            **ALIASES,
            "test_accuracy": "test/accuracy",
            "test_balanced_accuracy": "test/balanced_accuracy",
            "valid_loss": "validation/loss",
            "train_loss": "train/loss",
            "test_0001 - classrecall": "test/0001 - classrecall",
            "attack_eval/def.bandwidth_mean": "attack_eval/overhead/bandwidth/mean",
            "attack_eval/test_accuracy": "attack_eval/test/accuracy",
        }
        for old, new in pairs.items():
            with self.subTest(old=old):
                self.assertEqual(tracking_metric_name(old), new)
                self.assertEqual(tracking_metric_name(new), new)
                self.assertEqual(internal_metric_name(new), old)
                self.assertEqual(internal_metric_name(old), old)
        for name in ("disc/train/loss", "disc/train/lr", "obs/reward", "custom.value"):
            self.assertEqual(tracking_metric_name(name), name)
            self.assertEqual(internal_metric_name(name), name)

    def test_canonical_presence_wins_over_legacy_even_for_zero_or_nan(self):
        values = {
            "test/accuracy": 0.0, "test_accuracy": .8,
            "overhead/delay/mean": float("nan"), "def.delay_mean": 0.0,
        }
        for source in (values, dict(reversed(list(values.items())))):
            normalized = normalize_metrics(source)
            self.assertEqual(set(normalized), {"test_accuracy", "def.delay_mean"})
            self.assertEqual(normalized["test_accuracy"], 0.0)
            self.assertTrue(math.isnan(normalized["def.delay_mean"]))
            self.assertTrue(has_metric(source, "test_accuracy"))
            self.assertEqual(get_metric(source, "test/accuracy"), 0.0)
            self.assertEqual(get_metric(source, "absent", -1), -1)
            self.assertEqual(set(tracking_metrics(source)), {
                "test/accuracy", "overhead/delay/mean",
            })

    def test_dataframe_coalesces_by_key_presence_before_filling_absent_cells(self):
        def run(metrics):
            return SimpleNamespace(
                info=SimpleNamespace(
                    run_id="id", experiment_id="1", status="FINISHED",
                    artifact_uri="file:///artifacts", start_time=1000, end_time=2000,
                ),
                data=SimpleNamespace(metrics=metrics, params={}, tags={}),
            )
        frame = runs_to_dataframe([
            run({"def.delay_mean": .2}),
            run({"overhead/delay/mean": .3}),
            run({"overhead/delay/mean": float("nan"), "def.delay_mean": 0.0}),
        ])
        self.assertEqual(frame["metrics.def.delay_mean"].iloc[:2].tolist(), [.2, .3])
        self.assertTrue(pd.isna(frame["metrics.def.delay_mean"].iloc[2]))
        self.assertNotIn("metrics.overhead/delay/mean", frame)
        self.assertEqual(
            frame["start_time"].iloc[0], pd.Timestamp(1, unit="s", tz="UTC")
        )
        self.assertIn("status", runs_to_dataframe([]))

    def test_logger_forwards_options_and_tags_once_per_active_run(self):
        active = SimpleNamespace(
            info=SimpleNamespace(run_id="new"), data=SimpleNamespace(tags={})
        )
        with (
            patch("mlflow.active_run", return_value=active),
            patch("mlflow.set_tag") as tag,
            patch("mlflow.log_metrics") as batch,
            patch("mlflow.log_metric") as scalar,
        ):
            batch_result = log_metrics(
                {"def.bandwidth_mean": .5}, step=4, timestamp=1234
            )
            scalar_result = log_metric("valid_loss", .2, step=5, timestamp=2345)
        self.assertIs(batch_result, batch.return_value)
        self.assertIs(scalar_result, scalar.return_value)
        batch.assert_called_once_with(
            {"overhead/bandwidth/mean": .5}, step=4, timestamp=1234
        )
        scalar.assert_called_once_with("validation/loss", .2, step=5, timestamp=2345)
        tag.assert_called_once_with("metric_schema", "2")

    def test_history_uses_canonical_then_legacy_without_rewriting_points(self):
        client = Mock()
        point = SimpleNamespace(value=.7, step=17, timestamp=12345)
        client.get_metric_history.side_effect = [[], [point]]
        self.assertEqual(get_metric_history(client, "old", "test_accuracy"), [point])
        self.assertEqual(
            [call.args for call in client.get_metric_history.call_args_list],
            [("old", "test/accuracy"), ("old", "test_accuracy")],
        )
        self.assertEqual((point.step, point.timestamp), (17, 12345))
        client.reset_mock(side_effect=True)
        client.get_metric_history.return_value = [point]
        self.assertEqual(get_metric_history(client, "mixed", "test_accuracy"), [point])
        client.get_metric_history.assert_called_once_with("mixed", "test/accuracy")

    def test_explicit_run_id_tags_the_target_not_the_active_parent(self):
        parent = SimpleNamespace(
            info=SimpleNamespace(run_id="parent"), data=SimpleNamespace(tags={})
        )
        with (
            patch("mlflow.active_run", return_value=parent),
            patch("mlflow.MlflowClient") as client,
            patch("mlflow.set_tag") as tag,
            patch("mlflow.log_metric") as scalar,
        ):
            log_metric("test_accuracy", .7, run_id="child", synchronous=False)
        scalar.assert_called_once_with(
            "test/accuracy", .7, run_id="child", synchronous=False
        )
        client.return_value.set_tag.assert_called_once_with(
            "child", "metric_schema", "2"
        )
        tag.assert_not_called()
        self.assertEqual(parent.data.tags, {})

    def test_mixed_schema_store_and_old_run_are_preserved(self):
        uri = mlflow.get_tracking_uri()
        last_run = fluent._last_active_run_id.get()
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.dict(os.environ),
            patch("mlflow.tracking.fluent._active_experiment_id", None),
        ):
            mlflow.set_tracking_uri(f"sqlite:///{Path(tmp) / 'tracking.db'}")
            try:
                self._exercise_store(tmp)
            finally:
                mlflow.set_tracking_uri(uri)
                fluent._last_active_run_id.set(last_run)

    def _exercise_store(self, tmp):
        client = mlflow.MlflowClient()
        experiment = client.create_experiment(
            "metric-migration", str(Path(tmp) / "artifacts")
        )
        with mlflow.start_run(experiment_id=experiment) as old:
            mlflow.log_param("legacy_only", "old")
            mlflow.log_metrics(
                {"test_accuracy": .7, "def.bandwidth_mean": .25}, step=3, timestamp=1000
            )
            old_id = old.info.run_id
        with mlflow.start_run(experiment_id=experiment) as new:
            mlflow.log_param("new_only", "new")
            log_metrics(
                {"test_accuracy": .8, "def.bandwidth_mean": .5}, step=3, timestamp=1000
            )
            log_metric("valid_loss", .4, step=2, timestamp=500)
            new_id = new.info.run_id
        before = client.get_run(old_id).to_dictionary()
        frame = list_runs("metric-migration")
        raw_frame = mlflow.search_runs(experiment_ids=[experiment])
        assert isinstance(raw_frame, pd.DataFrame)
        metadata = [key for key in raw_frame if not key.startswith("metrics.")]
        pd.testing.assert_frame_equal(
            frame[metadata].sort_values("run_id").reset_index(drop=True),
            raw_frame[metadata].sort_values("run_id").reset_index(drop=True),
        )
        observed = frame.set_index("run_id")["metrics.test_accuracy"].to_dict()
        self.assertEqual(observed, {old_id: .7, new_id: .8})
        self.assertEqual(client.get_run(old_id).to_dictionary(), before)
        self.assertNotIn("metric_schema", client.get_run(old_id).data.tags)
        saved = client.get_run(new_id)
        self.assertEqual(saved.data.tags["metric_schema"], "2")
        self.assertEqual(set(saved.data.metrics), {
            "test/accuracy", "overhead/bandwidth/mean", "validation/loss",
        })
        for run_id, value in ((old_id, .7), (new_id, .8)):
            points = get_metric_history(client, run_id, "test_accuracy")
            self.assertEqual(
                [(p.value, p.step, p.timestamp) for p in points], [(value, 3, 1000)]
            )
        empty = client.create_experiment("empty-migration")
        self.assertTrue(list_runs("empty-migration", raise_on_empty=False).empty)
        with self.assertRaisesRegex(ValueError, "Empty df"):
            list_runs("empty-migration")
        with mlflow.start_run(experiment_id=empty):
            log_metric("test_accuracy", 0.0)
            self.assertTrue(list_runs("empty-migration", raise_on_empty=False).empty)
            self.assertEqual(len(list_runs("empty-migration", only_finished=False)), 1)


if __name__ == "__main__":
    unittest.main()
