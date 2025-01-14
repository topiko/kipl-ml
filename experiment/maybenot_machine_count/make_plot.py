import argparse

import matplotlib.pyplot as plt
import seaborn as sns
from kipl_ml.tools.mlflow_utils import list_runs
from kipl_ml.visualize import style


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="Model name", default="df-multi")
    args = parser.parse_args()

    experiment_name = f"laserbeak->{args.model} vs. maybenot"

    runs_df = list_runs(experiment_names=experiment_name, only_finished=True)

    metrics_df = runs_df.loc[:, ["tags.mlflow.runName", "metrics.test_accuracy"]]
    metrics_df.loc[:, "n_machines"] = metrics_df.loc[:, "tags.mlflow.runName"].apply(
        lambda x: int(x.split(" ")[0])
    )

    metrics_df = metrics_df.drop(columns="tags.mlflow.runName").rename(
        columns={"metrics.test_accuracy": "accuracy"}
    )

    print(metrics_df)

    sns.barplot(
        data=metrics_df,
        x="n_machines",
        y="accuracy",
        capsize=0.3,
        err_kws={"color": "0", "linewidth": 2.5},
        alpha=0.6,
    )
    plt.title(experiment_name)
    plt.ylabel("Test Accuracy")
    plt.xlabel("Number of State Machines")
    plt.ylim(0.5, 1)
    plt.tight_layout()
    plt.savefig("n_machines_vs_accuracy.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    main()
