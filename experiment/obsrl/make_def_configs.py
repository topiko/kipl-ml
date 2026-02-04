import argparse
import re
from pathlib import Path

from mlflow.tracking import MlflowClient

from experiment.utils.list_models import list_logged_models_for_run
from kipl_ml.tools.mlflow_utils import set_tracking_uri_from_env


def _slugify(s: str) -> str:
    s = s.strip()
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", s)
    s = re.sub(r"_+", "_", s)
    return s.strip("_-") or "run"


def _write_rlobs_defence_yaml(
    path: Path, run_name: str, step: int, model_id: str
) -> None:
    # Match existing defence config style in this repo (minimal keys).
    txt = (
        f'type: "rlobs"\nrun_name: {run_name}\nstep: {step}\n'
        + f"model_id:\n  - {model_id}\n  - {model_id}\n  - {model_id}\n\n"
    )
    path.write_text(txt, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate tmp_def/<run_name>/<run_name>-<step>.yaml for each logged rlobs model in a run."
        )
    )
    parser.add_argument("run_id", help="MLflow run id")
    args = parser.parse_args()

    set_tracking_uri_from_env()

    client = MlflowClient()
    run = client.get_run(args.run_id)
    run_name = getattr(run.info, "run_name", None) or getattr(run.info, "run_id")
    run_name = _slugify(str(run_name))

    df = list_logged_models_for_run(args.run_id)
    if len(df) == 0:
        print("No logged models found for run.")
        return

    if "name" not in df.columns:
        print("Logged model list is missing 'name' column; cannot filter rlobs models.")
        return

    df_rlobs = df[df["name"].astype(str).str.startswith("rlobs-")].copy()
    if len(df_rlobs) == 0:
        print("No rlobs-* models found for run.")
        return

    out_dir = Path("tmp_def") / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Prefer the step from run outputs; fall back to parsing from name if needed.
    def _infer_step(row) -> int | None:
        step = row.get("step")
        if step is not None:
            try:
                return int(step)
            except Exception:
                pass
        name = str(row.get("name") or "")
        m = re.match(r"^rlobs-(\d+)$", name)
        return int(m.group(1)) if m else None

    created = 0
    for _, row in df_rlobs.iterrows():
        model_id = row.get("model_id")
        if not isinstance(model_id, str) or not model_id:
            continue

        step = _infer_step(row)
        step_str = f"{step:03d}" if step is not None else "na"
        out_path = out_dir / f"{run_name}-{step_str}.yaml"
        _write_rlobs_defence_yaml(
            out_path, run_name=run_name, step=step, model_id=model_id
        )
        created += 1

    print(f"Wrote {created} config(s) to {out_dir}/")


if __name__ == "__main__":
    main()
