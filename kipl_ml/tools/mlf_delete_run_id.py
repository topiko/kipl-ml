import argparse

import mlflow
from kipl_ml.tools.mlflow_utils import list_runs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", type=str, required=False, help="Run id to delete.")

    args = parser.parse_args()

    if args.run_id is None:
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


    yn = input(f"Run with id '{args.run_id}' is about to be deleted. Continue? (y/n)")
    if yn.lower() == "y":
        mlflow.delete_run(args.run_id)
        print(f"Run with id {args.run_id} was deleted.")

    run = mlflow.get_run(args.run_id)
    print("Lifecycle:", run.info.lifecycle_stage)


if __name__ == "__main__":
    main()
