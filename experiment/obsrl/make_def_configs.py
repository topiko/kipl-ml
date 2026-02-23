import argparse
import re
from pathlib import Path

import pandas as pd
from mlflow.tracking import MlflowClient

from experiment.utils.list_models import list_logged_models_for_run
from kipl_ml.tools.mlflow_utils import (
    find_parent_run_id,
    list_child_runs,
    require_experiment_id,
    set_tracking_uri_from_env,
)


def _slugify(s: str) -> str:
    s = s.strip()
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", s)
    s = re.sub(r"_+", "_", s)
    return s.strip("_-") or "run"


def _write_rlobs_defence_yaml(
    path: Path,
    run_name: str,
    train_step: int,
    train_model_id: str,
    test_step: int,
    test_model_id: str,
) -> None:
    # Match existing defence config style in this repo (minimal keys).
    txt = (
        f'type: "rlobs"\nrun_name: {run_name}\ntrain_step: {train_step}\n'
        f"test_step: {test_step}\n"
        + f"model_id:\n  - {train_model_id}\n  - {test_model_id}\n  - {test_model_id}\n\n"
    )
    path.write_text(txt, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate tmp_def/<run_name>/<run_name>-<step>.yaml for each logged rlobs model in a run, "
            "or for all child runs under a parent run."
        )
    )
    parser.add_argument(
        "run_id",
        help=(
            "MLflow run id (default mode). If --from-parent is set, this is treated as the parent run name."
        ),
    )
    parser.add_argument(
        "--from-parent",
        action="store_true",
        default=False,
        help="Interpret the positional argument as a parent run name and read all nested child runs.",
    )
    parser.add_argument(
        "--experiment-name",
        default=None,
        help="MLflow experiment name (required with --from-parent).",
    )

    parser.add_argument(
        "--map-name",
        default=None,
        help="If you want to give your own name to the run.",
    )

    parser.add_argument(
        "--unpair",
        action="store_true",
        default=False,
        help="Unpair the train and test defences",
    )
    args = parser.parse_args()

    set_tracking_uri_from_env()

    client = MlflowClient()

    if args.from_parent:
        if not args.experiment_name:
            raise ValueError("--experiment-name is required with --from-parent")
        parent_run_name = str(args.run_id)

        experiment_id = require_experiment_id(args.experiment_name)
        parent_run_id = find_parent_run_id(experiment_id, parent_run_name)
        if parent_run_id is None:
            raise ValueError(
                f"No parent run found for name '{parent_run_name}' in experiment '{args.experiment_name}'"
            )

        child_runs = list_child_runs(experiment_id, parent_run_id)
        run_ids = [r.info.run_id for r in child_runs]
        run_name = parent_run_name
    else:
        run = client.get_run(args.run_id)
        run_name = getattr(run.info, "run_name", None) or getattr(run.info, "run_id")
        run_ids = [args.run_id]

    if args.unpair:
        run_name += "-unpaired"

    run_name = args.map_name or _slugify(str(run_name))

    dfs = []
    for run_id in run_ids:
        df = list_logged_models_for_run(run_id)
        if len(df) != 0:
            df = df.copy()
            df["_run_id"] = run_id
            dfs.append(df)

    if len(dfs) == 0:
        print("No logged models found.")
        return

    # Concatenate and keep deterministic ordering.

    df = pd.concat(dfs, ignore_index=True)

    if "name" not in df.columns:
        print("Logged model list is missing 'name' column; cannot filter rlobs models.")
        return

    df_rlobs = df[df["name"].astype(str).str.startswith("obs-")].copy()
    if len(df_rlobs) == 0:
        print("No obs-* models found for run.")
        return

    out_dir = Path("tmp_def") / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Prefer the step from run outputs; fall back to parsing from name if needed.
    def _infer_step(row) -> int | None:
        if (step := row.get("step")) is not None:
            return int(step)

        raise ValueError("No step found")

    created = 0

    # Deterministic ordering across child runs.
    sort_cols = [c for c in ["_run_id", "step", "model_id"] if c in df_rlobs.columns]
    if sort_cols:
        df_rlobs = df_rlobs.sort_values(by=sort_cols, kind="stable")

    df_rlobs.reset_index(inplace=True)
    for idx, row in df_rlobs.iterrows():
        train_model_id = row.get("model_id")
        if not isinstance(train_model_id, str) or not train_model_id:
            continue

        train_step = _infer_step(row)

        if idx + 1 in df_rlobs.index and args.unpair:
            test_row = df_rlobs.loc[idx + 1]
            test_step = _infer_step(test_row)
            test_model_id = test_row.get("model_id")
        else:
            test_step = train_step
            test_model_id = train_model_id

        step_str = f"{train_step:03d}" if train_step is not None else "na"
        out_path = out_dir / f"{run_name}-{step_str}.yaml"
        _write_rlobs_defence_yaml(
            out_path,
            run_name=run_name,
            train_step=train_step,
            train_model_id=train_model_id,
            test_step=test_step,
            test_model_id=test_model_id,
        )
        created += 1

    print(f"Wrote {created} config(s) to {out_dir}/")


if __name__ == "__main__":
    main()
