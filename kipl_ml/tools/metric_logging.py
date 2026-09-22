"""MLflow metric schema v2, with legacy names retained at Python read boundaries.

Only owned metric families are translated. In particular, existing ``disc/...``
and ``obs/...`` diagnostics remain untouched. ``attack_eval/`` is an explicit
namespace containing the same evaluation metrics as a standalone attack run.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import mlflow

METRIC_SCHEMA_TAG = "metric_schema"
METRIC_SCHEMA_VERSION = "2"

ALIASES = {
    "lr": "train/learning_rate",
    "sim.missing": "simulation/missing_fraction",
    "sim.overhead_pairs": "simulation/overhead_pairs",
    "sim.overhead_no_normal": "simulation/overhead_no_normal",
    "sim.delay_coverage": "simulation/delay_coverage",
    "evaluation_seconds": "evaluation/seconds",
    "paired_mixed_accuracy": "test/paired_mixed/accuracy",
    "paired_mixed_loss": "test/paired_mixed/loss",
}
for _metric in ("bandwidth", "delay"):
    for _full in (False, True):
        for _stat in (
            "min", "median", "mean", "std", "p90", "p95", "p99", "max", "count"
        ):
            _legacy = f"def.{_metric}{'_full' if _full else ''}_{_stat}"
            ALIASES[_legacy] = f"overhead/{_metric}/{'full/' if _full else ''}{_stat}"
REVERSE_ALIASES = {new: old for old, new in ALIASES.items()}
STAGES = {"train": "train", "valid": "validation", "test": "test"}


def tracking_metric_name(name: str) -> str:
    """Map an internal metric key to its UI name; already-new/unknown keys survive."""
    if name.startswith("attack_eval/"):
        return "attack_eval/" + tracking_metric_name(name[len("attack_eval/"):])
    if name in ALIASES:
        return ALIASES[name]
    for old, new in STAGES.items():
        if name.startswith(old + "_") and "/" not in name:
            return new + "/" + name[len(old) + 1:]
    return name


def internal_metric_name(name: str) -> str:
    """Resolve v2 keys to the existing report/trainer vocabulary."""
    if name.startswith("attack_eval/"):
        return "attack_eval/" + internal_metric_name(name[len("attack_eval/"):])
    if name in REVERSE_ALIASES:
        return REVERSE_ALIASES[name]
    for old, new in STAGES.items():
        prefix = new + "/"
        if name.startswith(prefix) and "/" not in name[len(prefix):]:
            return old + "_" + name[len(prefix):]
    return name


def metric_names(name: str) -> tuple[str, ...]:
    legacy = internal_metric_name(name)
    return tuple(dict.fromkeys((tracking_metric_name(legacy), legacy)))


def has_metric(metrics: Mapping[str, Any], name: str) -> bool:
    return any(key in metrics for key in metric_names(name))


def get_metric(metrics: Mapping[str, Any], name: str, default=None):
    for key in metric_names(name):
        if key in metrics:
            return metrics[key]  # Presence wins, including an explicitly logged NaN.
    return default


def normalize_metrics(metrics: Mapping[str, Any]) -> dict[str, Any]:
    """New spelling wins independent of insertion order; never count aliases twice."""
    return {
        internal_metric_name(key): get_metric(metrics, key)
        for key in metrics
    }


def tracking_metrics(metrics: Mapping[str, Any]) -> dict[str, float]:
    return {
        tracking_metric_name(key): float(value)
        for key, value in normalize_metrics(metrics).items()
    }


def mark_metric_schema(run_id: str | None = None) -> None:
    """Tag the run being written, caching the tag on the active Run entity."""
    active = mlflow.active_run()
    if run_id is not None and (active is None or active.info.run_id != run_id):
        mlflow.MlflowClient().set_tag(run_id, METRIC_SCHEMA_TAG, METRIC_SCHEMA_VERSION)
    elif (
        active is not None
        and active.data.tags.get(METRIC_SCHEMA_TAG) != METRIC_SCHEMA_VERSION
    ):
        mlflow.set_tag(METRIC_SCHEMA_TAG, METRIC_SCHEMA_VERSION)
        active.data.tags[METRIC_SCHEMA_TAG] = METRIC_SCHEMA_VERSION


def log_metrics(metrics: Mapping[str, Any], **kwargs):
    """Write only v2 metric names; forward step/timestamp and other MLflow options."""
    result = mlflow.log_metrics(tracking_metrics(metrics), **kwargs)
    mark_metric_schema(kwargs.get("run_id"))
    return result


def log_metric(name: str, value, **kwargs):
    result = mlflow.log_metric(tracking_metric_name(name), float(value), **kwargs)
    mark_metric_schema(kwargs.get("run_id"))
    return result


def log_metrics_under(
    metrics: Mapping[str, Any], key: str, *, step: int | None = None
) -> None:
    prefix = key.strip("/")
    if not prefix:
        raise ValueError("metric namespace must not be empty")
    log_metrics(
        {f"{prefix}/{name.lstrip('/')}": value for name, value in metrics.items()},
        step=step,
    )


def get_metric_history(client, run_id: str, name: str):
    """Prefer the canonical series, otherwise the old series; preserve all points."""
    for key in metric_names(name):
        history = client.get_metric_history(run_id, key)
        if history:
            return history
    return []


def runs_to_dataframe(runs):
    """Match MLflow's pandas layout, normalizing before absent keys become NaNs.

    A dataframe alone cannot distinguish a missing canonical cell from a metric
    explicitly logged as NaN. Run entities preserve that key-presence information.
    """
    import pandas as pd

    fields = (
        "run_id", "experiment_id", "status", "artifact_uri", "start_time", "end_time"
    )
    rows = []
    for run in runs:
        row = {key: getattr(run.info, key) for key in fields}
        for key in ("start_time", "end_time"):
            row[key] = pd.to_datetime(row[key], unit="ms", utc=True)
        row.update({
            f"metrics.{k}": v for k, v in normalize_metrics(run.data.metrics).items()
        })
        row.update({f"params.{k}": v for k, v in run.data.params.items()})
        row.update({f"tags.{k}": v for k, v in run.data.tags.items()})
        rows.append(row)
    frame = pd.DataFrame(rows) if rows else pd.DataFrame(columns=pd.Index(fields))
    for key in frame.columns:
        if key.startswith(("params.", "tags.")):
            frame[key] = frame[key].where(frame[key].notna(), None)
    return frame
