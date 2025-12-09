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
        R = rewards[:, t] + gamma * R
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
    clf_orig: nn.Module,
    clf_trained: nn.Module,
    e: int,
    dt: float,
    reward_scales: dict[str, float],
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
            clf_orig=clf_orig,
            clf_trained=clf_trained,
            e=e,
            dt=dt,
            reward_scales=reward_scales,
            device=device,
            idx=idx,
            max_len=max_len,
        )


@torch.no_grad()
def _plot_single(
    ds: WFDataset,
    obs: nn.Module,
    clf_orig: nn.Module,
    clf_trained: nn.Module,
    e: int,
    dt: float,
    reward_scales: dict[str, float],
    device: torch.DeviceObjType,
    idx: int,
    max_len: int = 10_000,
):
    fig, (ax, ax_o, ax_a, ax_b) = plt.subplots(4, 1, figsize=(20, 9.0), sharex=True)

    def _unsqueeze(X: dict[Feats, torch.Tensor]) -> dict[Feats, torch.Tensor]:
        return {k: v.unsqueeze(0) for k, v in X.items()}

    def X_to_probs(clf: nn.Module, X: dict[Feats, torch.Tensor]) -> torch.Tensor:
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
        cl_probs=X_to_probs(clf_orig, X),
        true_class=y.item(),
    )
    ax.set_title(f"True class: {y.item()}")

    rewards, _, times, actions, Xobs = rollout(
        obs, clf_trained, _unsqueeze(X), y, dt=dt, reward_scales=reward_scales
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
        cl_probs=X_to_probs(clf_trained, Xobs),
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
    train: bool = True,
) -> tuple[float, float]:
    disc.train()
    disc_opm.zero_grad()
    logits, _ = disc(X)
    loss = nn.functional.cross_entropy(
        logits.permute(0, 2, 1), y.unsqueeze(-1).repeat(1, logits.shape[1])
    )
    loss.backward()

    if train:
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

    discriminator_orig = mlflow.pytorch.load_model(model_uri, map_location="cpu")
    discriminator = mlflow.pytorch.load_model(model_uri, map_location="cpu")
    # discriminator = RNNCLF1(n_classes=95, features=[Feats.DIRS, Feats.TIMES])

    feature_names = [Feats.DIRS, Feats.TIMES]

    ds_train, ds_valid, _ = get_train_valid_test(
        dataset=DATASET,
        label=assets.PAGE_LABEL,
        n_splits=N_SPLITS,
        test_xv=TEST_XV,
        random_state=42,
        feature_trs=FeatureTrs(feature_names=feature_names, n_packets=cfg.trace_len),
        **defence_builder.get_defence(cfg),
    )

    dl_train = dl_(ds_train, bs=cfg.batch_size, collate_fn=None, shuffle=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    obs = AGENT1(zero_init=False).to(device)
    discriminator = discriminator.to(device)
    discriminator_orig = discriminator_orig.to(device)

    optim = torch.optim.Adam(obs.parameters(), lr=0.001)
    disc_optim = torch.optim.Adam(discriminator.parameters(), lr=0.001)

    e = 0
    dt = 0.05
    T = 30
    i = 0
    detach_period = 2.0
    gamma = cfg.discounting
    with mlflow.start_run(log_system_metrics=True):
        while True:
            losses: dict[str, list[float]] = {
                "loss": [],
                "policy_loss": [],
                "value_loss": [],
                "avg_return": [],
                "entropy_loss": [],
                "h_penalty": [],
                "c_penalty": [],
                "disc_loss": [],
                "disc_acc": [],
                "mean_padding_count": [],
                "mean_trace_len": [],
            }
            reward_scales = {"clf_scale": 1.0, "padding_scale": 0.01}

            train_obs = True
            train_disc = True
            # if i % 5 == 0:
            #    train_disc = True
            #    train_obs = False

            with tqdm(
                dl_train,
                desc=f"epoch {e:02d}",
                ncols=2 * TQDM_W,
            ) as pbar:
                for X, y in pbar:
                    X = dict_to_device(X, device)
                    y = y.to(device)

                    if train_obs:
                        optim.zero_grad()
                        (
                            log_ps,
                            values,
                            rewards,
                            entropies,
                            _,
                            _,
                            Xobs,
                        ) = rollout(
                            obs=obs,
                            disc=discriminator,
                            X=X,
                            y=y,
                            dt=dt,
                            detach_every_delta_t=detach_period,
                            reward_scales=reward_scales,
                        )

                        if cfg.advantages.type == "mc":
                            G = get_returns(rewards, gamma=gamma)
                            advantages = G - values
                        elif cfg.advantages.type == "gae":
                            advantages = get_gae(
                                rewards,
                                values,
                                lambda_=cfg.advantages.lambda_,
                                gamma=gamma,
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
                        entropy_loss = -entropies.mean()

                        loss = policy_loss + value_loss + 0.05 * entropy_loss

                        loss.backward()

                        skip = False
                        for name, p in obs.named_parameters():
                            if p.grad is not None:
                                try:
                                    assert_finite(f"{name}: grad", p.grad)
                                except ValueError as er:
                                    logger.warning(er)
                                    print(value_loss, policy_loss, entropy_loss)
                                    skip = True
                                    break

                        if skip:
                            continue

                        # Gradient clipping
                        nn.utils.clip_grad_norm_(
                            obs.parameters(),
                            cfg.grad_norm_clip,
                            error_if_nonfinite=False,
                        )

                        optim.step()

                        losses["loss"].append(loss.item())
                        losses["policy_loss"].append(policy_loss.item())
                        losses["value_loss"].append(value_loss.item())
                        losses["avg_return"].append(G.mean().item())
                        losses["entropy_loss"].append(entropy_loss.item())
                        for k, v in losses.items():
                            if len(v) == 0:
                                continue

                            try:
                                assert_finite(k, v[-1])
                            except ValueError as er:
                                print(er)
                                breakpoint()

                    # mlflow.pytorch.log_model(obs, name=f"rlobs-{i}")

                    elif train_disc:
                        with torch.no_grad():
                            Xobs = rollout(
                                obs=obs,
                                disc=discriminator,
                                X=X,
                                y=y,
                                dt=dt,
                                detach_every_delta_t=detach_period,
                                reward_scales=None,
                            )[-1]

                    disc_loss, acc = one_batch_train_disc(
                        disc=discriminator,
                        X=Xobs,
                        y=y,
                        disc_opm=disc_optim,
                        train=train_disc,
                    )
                    losses["disc_acc"].append(acc)
                    losses["disc_loss"].append(disc_loss)
                    losses["mean_padding_count"].append(
                        Xobs[Feats.PADDING].sum(dim=1).float().mean().item()
                    )
                    losses["mean_trace_len"].append(
                        (Xobs[Feats.DIRS] != 0).sum(dim=1).float().mean().item()
                    )

                    if train_obs:
                        pbar.set_postfix({"avg_return": np.mean(losses["avg_return"])})
                    elif train_disc:
                        pbar.set_postfix({"dacc": np.mean(losses["disc_acc"])})

                    if train_obs and (i % 10 == 0):
                        mlflow.log_metrics(
                            {k: np.mean(l_) for k, l_ in losses.items()}, step=i
                        )
                        losses = {k: [] for k in losses}

                    i += 1

            _plot_set(
                ds=ds_valid,
                obs=obs,
                clf_orig=discriminator_orig,
                clf_trained=discriminator,
                e=e,
                dt=dt,
                reward_scales=reward_scales,
                device=device,
                ntraces=20,
            )

            # if reward_scales["padding_scale"] > -1.0:
            #     reward_scales["padding_scale"] -= 0.02

            e += 1


if __name__ == "__main__":
    main()
