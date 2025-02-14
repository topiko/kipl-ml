from typing import Any

import mlflow

from kipl_ml.logging.logger import get_logger

KEY_LEN = 25

logger = get_logger(__name__)


def log_multiline(log_str: str) -> None:
    for line in log_str.split("\n"):
        logger.info(line)


def key_val_fmt(key: str, val: Any, key_len: int = KEY_LEN, suffix: str = "\n") -> str:
    return f"{key:>{key_len}}: {val}{suffix}"


def get_mlflow_expr(experiment_name: str) -> str:
    """
    Retrieve the ID of an existing MLflow experiment or
    create a new one if it doesn't exist.

    If it does, the function returns its ID. If not,
    it creates a new experiment with the provided name and returns its ID.

    Args:
        experiment_name (str): Name of the MLflow experiment.

    Returns:
        str: ID of the existing or newly created MLflow experiment.

    """

    if experiment := mlflow.get_experiment_by_name(experiment_name):
        logger.info("Experiment '%s' already exists --> return.", experiment_name)
        return str(experiment.experiment_id)

    logger.info("Experiment '%s' does not exist --> create.", experiment_name)
    return mlflow.create_experiment(experiment_name)
