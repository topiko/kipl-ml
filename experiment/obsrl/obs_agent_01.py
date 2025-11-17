import os

import dotenv
import hydra
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import torch
from omegaconf import DictConfig
from torch import nn
from tqdm import tqdm

from experiment.obsrl.sim import rollout
from experiment.trace_gan.data_utils import dl_
from experiment.utils import defence_builder
from kipl_ml.data.utils import Datasets, assets
from kipl_ml.data.wf_dataset import WFDataset, dict_to_device, get_train_valid_test
from kipl_ml.logging.logger import TQDM_W, get_logger
from kipl_ml.models.trgen import AGENT1
from kipl_ml.tools.mlflow_utils import get_mlflow_expr
from kipl_ml.tools.plottr import plot_trace
from kipl_ml.trace.features import Feats, FeatureTrs

logger = get_logger(__name__)
dotenv.load_dotenv()

WORKING_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_DIR_PATH = os.path.join(WORKING_DIR, "config")


N_SPLITS = 5
TEST_XV = 0
TARGET = assets.PAGE_LABEL
DATASET = Datasets.BIGENOUGH


def returns(rewards: torch.Tensor, gamma: float) -> torch.Tensor:
    G = torch.zeros_like(rewards)
    R = 0
    i = 0

    for r in rewards.flip(dims=(1,)).T:
        R = r + gamma * R
        G[:, -1 - i] = R
        i += 1
    return G


def _plot_set(
    ds: WFDataset,
    obs: nn.Module,
    clf: nn.Module,
    e: int,
    device: torch.DeviceObjType,
):
    rng = np.random.default_rng(seed=42)

    ntraces = 5

    idxs = rng.integers(0, len(ds), size=ntraces)

    fig, axarr = plt.subplots(ntraces, 2, figsize=(10, ntraces * 3), sharex=True)

    def _unsqueeze(X: dict[Feats, torch.Tensor]) -> dict[Feats, torch.Tensor]:
        return {k: v.unsqueeze(0) for k, v in X.items()}

    def X_to_probs(X: dict[Feats, torch.Tensor]) -> torch.Tensor:
        logits, _ = clf(_unsqueeze(X))
        probs = nn.functional.softmax(logits, dim=-1)
        return probs

    for i, axrow in zip(idxs, axarr):
        X, y = ds[i]

        X = dict_to_device(X, device)
        y = y.to(device).unsqueeze(-1)

        plot_trace(
            X,
            ax=axrow[0],
            cl_probs=X_to_probs(X),
            true_class=y.item(),
        )
        axrow[0].set_title(f"True class: {y.item()}")

        Xobs = rollout(obs, clf, _unsqueeze(X), y)[0]
        Xobs = {k: v.squeeze(0) for k, v in Xobs.items()}

        mask = Xobs[Feats.TIMES] == 0
        mask[0] = False
        if mask.any():
            Xobs[Feats.TIMES] = Xobs[Feats.TIMES][~mask]
            Xobs[Feats.DIRS] = Xobs[Feats.DIRS][~mask]
        plot_trace(
            Xobs,
            ax=axrow[1],
            cl_probs=X_to_probs(Xobs),
            true_class=y.item(),
        )

        axrow[1].set_title("Obsfuscated")

    fig.canvas.draw()

    mlflow.log_figure(fig, f"bursts_clf_epoch={e:03d}.png")

    plt.close()


@hydra.main(config_path=CONFIG_DIR_PATH, config_name="battle-config", version_base=None)
def main(cfg: DictConfig):
    experiment_name = "obsrl"
    experiment_id = get_mlflow_expr(experiment_name=experiment_name)
    mlflow.set_experiment(experiment_id=experiment_id)
    discriminator_model_id = "m-f87f760721794c8090e741cf3e2456b4"

    model_uri = mlflow.get_logged_model(discriminator_model_id).model_uri

    discriminator = mlflow.pytorch.load_model(model_uri, map_location="cpu")

    feature_names = [Feats.DIRS, Feats.TIMES]
    npackets = 1000

    ds_train, ds_valid, _ = get_train_valid_test(
        dataset=DATASET,
        label=assets.PAGE_LABEL,
        n_splits=N_SPLITS,
        test_xv=TEST_XV,
        random_state=42,
        feature_trs=FeatureTrs(feature_names=feature_names, n_packets=npackets),
        **defence_builder.get_defence(cfg),
    )

    dl_train = dl_(ds_train, bs=32, collate_fn=None, shuffle=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    obs = AGENT1().to(device)
    discriminator = discriminator.to(device)

    optim = torch.optim.Adam(obs.parameters(), lr=0.001)

    e = 0
    i = 0
    with mlflow.start_run():
        while True:
            with tqdm(
                dl_train,
                desc=f"epoch {e:02d}",
                ncols=2 * TQDM_W,
            ) as pbar:
                for X, y in pbar:
                    optim.zero_grad()
                    X = dict_to_device(X, device)
                    y = y.to(device)

                    _, log_ps, values, rewards = rollout(obs, discriminator, X, y)[:4]

                    G = returns(rewards, gamma=0.99)

                    advantages = G - values
                    # Compute losses

                    policy_loss = -(log_ps * advantages.detach()).mean()
                    value_loss = 0.5 * (values - G).pow(2).mean()

                    loss = policy_loss + value_loss

                    loss.backward()
                    optim.step()

                    losses = {
                        "loss": loss.item(),
                        "policy_loss": policy_loss.item(),
                        "value_loss": value_loss.item(),
                        "avg_return": G.mean().item(),
                    }

                    pbar.set_postfix(losses)

                    mlflow.log_metrics(losses, step=i)
                    i += 1

                    if (i - 1) % 10 == 0:
                        _plot_set(ds_valid, obs, discriminator, i, device)
            e += 1


if __name__ == "__main__":
    main()
