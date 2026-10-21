import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import mlflow
import torch
from mlflow.exceptions import RestException
from mlflow.tracking import fluent
from torch import nn

from kipl_ml.tools.mlflow_utils import get_mlflow_expr, log_pytorch_model


class DictModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 3)

    def forward(self, features):
        return self.linear(features["x"])

    def predict(self, features):
        return self(features).argmax(dim=1)


class TestExperimentCreation(unittest.TestCase):
    def test_concurrent_creator_reuses_winner(self):
        winner = SimpleNamespace(experiment_id="84", lifecycle_stage="active")
        error = RestException({
            "error_code": "RESOURCE_ALREADY_EXISTS", "message": "race",
        })
        with (
            patch("mlflow.get_experiment_by_name", side_effect=[None, winner]),
            patch("mlflow.create_experiment", side_effect=error),
        ):
            self.assertEqual(get_mlflow_expr("shared"), "84")

    def test_creation_errors_are_not_hidden(self):
        for code, winner in (
            ("RESOURCE_ALREADY_EXISTS", None),
            ("RESOURCE_ALREADY_EXISTS", SimpleNamespace(lifecycle_stage="deleted")),
            ("PERMISSION_DENIED", None),
            ("INTERNAL_ERROR", None),
        ):
            with self.subTest(code=code, winner=winner):
                error = RestException({
                    "error_code": code, "message": "creation failed",
                })
                with (
                    patch("mlflow.get_experiment_by_name", side_effect=[None, winner]),
                    patch("mlflow.create_experiment", side_effect=error),
                    self.assertRaises(RestException) as raised,
                ):
                    get_mlflow_expr("shared")
                self.assertIs(raised.exception, error)


class TestModelLogging(unittest.TestCase):
    def test_old_mlflow_does_not_receive_serialization_keyword(self):
        def old_save_model(model, path, **kwargs):
            pass

        with (
            patch("mlflow.pytorch.save_model", old_save_model),
            patch("mlflow.pytorch.log_model") as log,
        ):
            model = DictModel()
            log_pytorch_model(model, name="model")
            log.assert_called_once_with(model, name="model")

    def test_model_id_roundtrip_preserves_dict_inputs_and_predict(self):
        old_uri = mlflow.get_tracking_uri()
        last_run_id = fluent._last_active_run_id.get()
        with tempfile.TemporaryDirectory() as tmp, patch(
            "mlflow.tracking.fluent._active_experiment_id", None
        ):
            root = Path(tmp)
            mlflow.set_tracking_uri(f"sqlite:///{root / 'tracking.db'}")
            try:
                experiment_id = mlflow.create_experiment(
                    "roundtrip", artifact_location=str(root / "artifacts")
                )
                model = DictModel().eval()
                with mlflow.start_run(experiment_id=experiment_id):
                    info = log_pytorch_model(
                        model, name="model", pip_requirements=["torch"]
                    )
                loaded = mlflow.pytorch.load_model(
                    f"models:/{info.model_id}", device="cpu"
                )
                self.assertIsInstance(loaded, DictModel)
                for batch_size in (1, 7):
                    features = {"x": torch.randn(batch_size, 2)}
                    torch.testing.assert_close(loaded(features), model(features))
                    torch.testing.assert_close(
                        loaded.predict(features), model.predict(features)
                    )
            finally:
                mlflow.set_tracking_uri(old_uri)
                fluent._last_active_run_id.set(last_run_id)


if __name__ == "__main__":
    unittest.main()
