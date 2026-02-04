import argparse

import pandas as pd
from mlflow.tracking import MlflowClient

from kipl_ml.tools.mlflow_utils import set_tracking_uri_from_env


def list_logged_models_for_run(run_id: str) -> pd.DataFrame:
    """Return logged model outputs (model_id, step) for a given run."""

    client = MlflowClient()
    run = client.get_run(run_id)

    model_outputs = []
    outputs = getattr(run, "outputs", None)
    if outputs is not None:
        model_outputs = getattr(outputs, "model_outputs", []) or []

    rows: list[dict[str, object]] = []
    for mo in model_outputs:
        model_id = getattr(mo, "model_id", None)
        step = getattr(mo, "step", None)

        row: dict[str, object] = {"model_id": model_id, "step": step}

        # Best-effort enrichment (name, type, uri, status).
        if model_id:
            try:
                lm = client.get_logged_model(model_id)
                row.update(
                    {
                        "name": getattr(lm, "name", None),
                        "model_type": getattr(lm, "model_type", None),
                        "model_uri": getattr(lm, "model_uri", None),
                        "status": getattr(lm, "status", None),
                    }
                )
            except Exception:
                pass

        rows.append(row)

    df = pd.DataFrame(rows)
    if len(df) == 0:
        return df

    # Keep requested columns first.
    cols = [
        c
        for c in ["model_id", "step", "name", "model_type", "model_uri", "status"]
        if c in df.columns
    ]
    df = df.loc[:, cols]

    if "step" in df.columns:
        df = df.sort_values(by=["step", "model_id"], kind="stable")

    return df.reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="List MLflow logged models for a run (model_id + step)."
    )
    parser.add_argument("run_id", help="MLflow run id")
    args = parser.parse_args()

    set_tracking_uri_from_env()
    df = list_logged_models_for_run(args.run_id)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
