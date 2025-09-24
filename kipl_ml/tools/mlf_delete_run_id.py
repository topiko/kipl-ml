import argparse

import mlflow

from kipl_ml.tools.mlflow_utils import list_runs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", type=str, required=False, help="Run id to delete.")
    parser.add_argument(
        "--experiment-id", type=str, required=False, help="Experiment id to delete."
    )

    args = parser.parse_args()

    if (args.run_id is None) and (args.experiment_id is None):
        runs = list_runs(only_finished=False)
        print(
            runs.loc[
                :,
                [
                    "start_time",
                    "run_id",
                    "experiment_id",
                    "tags.mlflow.runName",
                    "status",
                ],
            ]
        )
        return

    def _delete_run(run_id: str):
        yn = input(f"Run with id '{run_id}' is about to be deleted. Continue? (y/n) ")
        if yn.lower() == "y":
            mlflow.delete_run(run_id)
            print(f"Deleted run {run_id}.")
        else:
            print(f"Skipped deletion for run {run_id}.")

        run = mlflow.get_run(run_id)
        print("Lifecycle:", run.info.lifecycle_stage)

    def _delete_experiment(experiment_id: str):
        yn = input(
            f"Experiment with id '{experiment_id}' is about to be deleted. Continue? (y/n) "
        )
        if yn.lower() == "y":
            mlflow.delete_experiment(experiment_id)
            print(f"Deleted experiment {experiment_id}.")
        else:
            print(f"Skipped deletion for experiment {experiment_id}.")

    if args.experiment_id is not None:
        _delete_experiment(args.experiment_id)

    if args.run_id is not None:
        _delete_run(args.run_id)


if __name__ == "__main__":
    main()
