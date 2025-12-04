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
from kipl_ml.tools.plottr import (
    plot_rewards,
    plot_trace,
)
from kipl_ml.trace.features import Feats, FeatureTrs

logger = get_logger(__name__)
dotenv.load_dotenv()

WORKING_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_DIR_PATH = os.path.join(WORKING_DIR, "config")


N_SPLITS = 5
TEST_XV = 0
TARGET = assets.PAGE_LABEL
DATASET = Datasets.BIGENOUGH


def get_returns(rewards: torch.Tensor, gamma: float) -> torch.Tensor:
    _, T = rewards.shape
    G = torch.zeros_like(rewards)
    R = 0

    # for r in rewards.flip(dims=(1,)).T:
    for t in reversed(range(T)):
        R = rewards[t] + gamma * R
        G[:, t] = R
    return G


def get_gae(
    rewards: torch.Tensor, values: torch.Tensor, lambda_: float, gamma: float
) -> torch.Tensor:
    """
    rewards, values: [B, T]
    returns: GAE advantages of shape [B, T]
    """

    B, T = rewards.shape
    # delta = r_t + gamma * V_t+1 - V_t
    deltas = torch.zeros_like(rewards)
    deltas[:, :-1] = rewards[:, :-1] + gamma * values[:, 1:] - values[:, :-1]
    # V_t+1 for last times step = 0
    deltas[:, -1] = rewards[:, -1] - values[:, -1]

    gae = torch.zeros_like(rewards)
    running = torch.zeros(B, device=rewards.device)

    y = lambda_ * gamma
    gae_ = 0
    for t in reversed(range(T)):
        gae_ = deltas[:, t] + running * y
        gae[:, t] = gae_

    return gae


def _plot_set(
    ds: WFDataset,
    obs: nn.Module,
    clf: nn.Module,
    e: int,
    dt: float,
    T: float,
    device: torch.DeviceObjType,
    ntraces: int = 3,
    max_len: int = 10_000,
):
    rng = np.random.default_rng(seed=42)

    idxs = rng.choice(len(ds), size=ntraces, replace=False)

    for idx in idxs:
        _plot_single(
            ds=ds,
            obs=obs,
            clf=clf,
            e=e,
            dt=dt,
            T=T,
            device=device,
            idx=idx,
            max_len=max_len,
        )


@torch.no_grad()
def _plot_single(
    ds: WFDataset,
    obs: nn.Module,
    clf: nn.Module,
    e: int,
    dt: float,
    T: float,
    device: torch.DeviceObjType,
    idx: int,
    max_len: int = 10_000,
):
    fig, (ax, ax_o, ax_a, ax_b) = plt.subplots(4, 1, figsize=(20, 9.0), sharex=True)

    def _unsqueeze(X: dict[Feats, torch.Tensor]) -> dict[Feats, torch.Tensor]:
        return {k: v.unsqueeze(0) for k, v in X.items()}

    def X_to_probs(X: dict[Feats, torch.Tensor]) -> torch.Tensor:
        with torch.no_grad():
            logits, _ = clf(_unsqueeze(X))
            probs = nn.functional.softmax(logits, dim=-1)
        return probs

    X, y = ds[idx]

    X = dict_to_device(X, device)
    y = y.to(device).unsqueeze(-1)

    plot_trace(
        X,
        ax=ax,
        cl_probs=X_to_probs(X),
        true_class=y.item(),
    )
    ax.set_title(f"True class: {y.item()}")

    rewards, _, _, times, actions_l, Xobs = rollout(
        obs, clf, _unsqueeze(X), y, dt=dt, maxT=T
    )[2:]

    mask = Xobs[Feats.DIRS] != 0
    if mask.sum() > max_len:
        logger.warning("Long seqs. detected -> truncating to %d.", max_len)

    if mask.any():
        Xobs[Feats.TIMES] = Xobs[Feats.TIMES][mask][:max_len]
        Xobs[Feats.DIRS] = Xobs[Feats.DIRS][mask][:max_len]
        Xobs[Feats.PADDING] = Xobs[Feats.PADDING][mask][:max_len]

    # Plot obsfuscated
    plot_trace(
        Xobs,
        ax=ax_o,
        cl_probs=X_to_probs(Xobs),
        true_class=y.item(),
    )

    # Plot actions
    # plot_actions(times, actions_l, idx2ackt=obs.action_map, ax=ax_a)
    ax_a.set_title("Actions, buffer, etc.")

    # Plot buffer and rewards
    # plot_packet_buffer(buffer, ax=ax_b)
    ax_b.set_ylabel("Buffer size [pkts]")
    ax_o.set_title("Obsfuscated")

    # Plot rewards
    ax_r = ax_b.twinx()
    ax_r.axes.spines["right"].set_visible(True)
    plot_rewards(times, rewards, ax=ax_r)

    ax_r.legend(frameon=False, loc=1)
    ax_b.legend(frameon=False, loc=2)
    ax_b.set_xlabel("Time [s]")

    fig.canvas.draw()

    plt.show()
    mlflow.log_figure(fig, f"trace_{idx}_clf_epoch={e:03d}.png")

    plt.close()


def check_grads(model, step):
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        if not torch.isfinite(p.grad).all():
            print(f"[step {step}] Non-finite grad in {name}")
            raise ValueError
        grad_norm = p.grad.data.norm(2).item()
        if grad_norm > 1e3:  # pick a threshold
            print(f"[step {step}] Large grad in {name}: {grad_norm:.2e}")


def assert_finite(name, x):
    raise_ = False
    if isinstance(x, (float, int)):
        if not np.isfinite(x):
            raise_ = True

    elif not torch.isfinite(x).all():
        raise_ = True

    if raise_:
        print(f"Non-finite in {name}")
        raise ValueError


def one_batch_train_disc(
    disc: nn.Module,
    X: dict[Feats, torch.Tensor],
    y: torch.Tensor,
    disc_opm: torch.optim.Optimizer,
):
    disc.train()

    disc_opm.zero_grad()

    logits, _ = disc(X)

    loss = nn.functional.cross_entropy(
        logits.permute(0, 2, 1), y.unsqueeze(-1).repeat(1, logits.shape[1])
    )

    loss.backward()

    disc_opm.step()

    loss_val = loss.item()

    disc.eval()

    accuracy = (disc.predict(X)[1] == y).float().mean().item()

    return loss_val, accuracy


@hydra.main(config_path=CONFIG_DIR_PATH, config_name="config", version_base=None)
def main(cfg: DictConfig):
    experiment_name = "obsrl"
    experiment_id = get_mlflow_expr(experiment_name=experiment_name)
    mlflow.set_experiment(experiment_id=experiment_id)
    discriminator_model_id = "m-126b3c004c14420a8dc098017714e23b"

    model_uri = mlflow.get_logged_model(discriminator_model_id).model_uri

    discriminator = mlflow.pytorch.load_model(model_uri, map_location="cpu")
    # discriminator = RNNCLF1(n_classes=95, features=[Feats.DIRS, Feats.TIMES])

    feature_names = [Feats.DIRS, Feats.TIMES]
    npackets = 5000

    ds_train, ds_valid, _ = get_train_valid_test(
        dataset=DATASET,
        label=assets.PAGE_LABEL,
        n_splits=N_SPLITS,
        test_xv=TEST_XV,
        random_state=42,
        feature_trs=FeatureTrs(feature_names=feature_names, n_packets=npackets),
        **defence_builder.get_defence(cfg),
    )

    dl_train = dl_(ds_train, bs=cfg.batch_size, collate_fn=None, shuffle=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    obs = AGENT1(zero_init=True).to(device)
    discriminator = discriminator.to(device)

    optim = torch.optim.Adam(obs.parameters(), lr=0.001)
    disc_optim = torch.optim.Adam(discriminator.parameters(), lr=0.001)

    e = 0
    dt = 0.05
    T = 20
    i = 0
    detach_period = 5.0
    clf_scale = 5
    gamma = cfg.discounting
    with mlflow.start_run(log_system_metrics=True):
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

                    log_ps, values, rewards, entropies, hidden_penalty, _, _, Xobs = (
                        rollout(
                            obs=obs,
                            disc=discriminator,
                            X=X,
                            y=y,
                            dt=dt,
                            maxT=T,
                            clf_scale=clf_scale,
                            detach_every_delta_t=detach_period,
                        )
                    )

                    if cfg.advantages.type == "mc":
                        G = get_returns(rewards, gamma=gamma)
                        advantages = G - values
                    elif cfg.advantages.type == "gae":
                        advantages = get_gae(
                            rewards, values, lambda_=cfg.advantages.lambda_, gamma=gamma
                        )
                        G = advantages + values
                    else:
                        raise NotImplementedError(
                            "Invalid advantage type: {cfg.advantages.type}"
                        )

                    if cfg.advantages.standardize:
                        advantages = (advantages - advantages.mean()) / (
                            advantages.std() + 1e-8
                        )

                    # Compute losses
                    policy_loss = -(log_ps * advantages.detach()).mean()
                    value_loss = 0.5 * (values - G).pow(2).mean()
                    entropy_loss = -10 * entropies.mean()
                    hidden_penalty = hidden_penalty * 1

                    loss = (
                        policy_loss + value_loss + entropy_loss + hidden_penalty.sum()
                    )

                    loss.backward()

                    for name, p in obs.named_parameters():
                        if p.grad is not None:
                            try:
                                assert_finite(f"{name}: grad", p.grad)
                            except ValueError as er:
                                print(er)
                                breakpoint()

                    # Gradient clipping
                    nn.utils.clip_grad_norm_(
                        obs.parameters(), cfg.grad_norm_clip, error_if_nonfinite=False
                    )

                    optim.step()

                    disc_loss, acc = one_batch_train_disc(
                        disc=discriminator, X=Xobs, y=y, disc_opm=disc_optim
                    )

                    losses = {
                        "loss": loss.item(),
                        "policy_loss": policy_loss.item(),
                        "value_loss": value_loss.item(),
                        "avg_return": G.mean().item(),
                        "entropy_loss": entropy_loss.item(),
                        "h_penalty": hidden_penalty[0].item(),
                        "c_penalty": hidden_penalty[1].item(),
                        "disc_loss": disc_loss,
                        "disc_acc": acc,
                        "gamma": gamma,
                    }

                    for k, v in losses.items():
                        try:
                            assert_finite(k, v)
                        except ValueError as er:
                            print(er)
                            breakpoint()

                    if i % 5 == 0:
                        mlflow.log_metrics(losses, step=i)

                    if i % 25 == 0:
                        _plot_set(ds_valid, obs, discriminator, i, dt, T, device)
                        # mlflow.pytorch.log_model(obs, name=f"rlobs-{i}")

                    pbar.set_postfix(
                        {
                            "avg_return": losses["avg_return"],
                            "disc_loss": disc_loss,
                            "disc_acc": acc,
                        }
                    )
                    i += 1

            e += 1


if __name__ == "__main__":
    main()
