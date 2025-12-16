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
from kipl_ml.rl.advantages import get_gae, get_returns
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


def _plot_set(
    ds: WFDataset,
    obs: nn.Module,
    clf_orig: nn.Module,
    clf_trained: nn.Module,
    e: int,
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
        obs,
        clf_trained,
        _unsqueeze(X),
        y,
        reward_scales=reward_scales,
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


def get_advantages(
    rewards: torch.Tensor | dict[str, torch.Tensor],
    values: torch.Tensor,
    cfg: DictConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    if isinstance(rewards, dict):
        rewards = sum(rewards.values())

    if cfg.advantages.type == "mc":
        G = get_returns(rewards, gamma=cfg.discounting)
        advantages = G - values
    elif cfg.advantages.type == "gae":
        advantages = get_gae(
            rewards,
            values,
            lambda_=cfg.advantages.lambda_,
            gamma=cfg.discounting,
        )
        G = advantages + values
    else:
        raise NotImplementedError(f"Invalid advantage type: {cfg.advantages.type}")

    if cfg.advantages.standardize:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    return G, advantages


def assert_finite(name, x):
    raise_ = False
    if isinstance(x, (float, int)):
        if not np.isfinite(x):
            raise_ = True

    elif not torch.isfinite(x).all():
        raise_ = True
    elif torch.isnan(x).any():
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
    if train:
        disc.train()
        disc_opm.zero_grad()
        logits, _ = disc(X)
        loss = nn.functional.cross_entropy(
            logits.permute(0, 2, 1), y.unsqueeze(-1).repeat(1, logits.shape[1])
        )
        loss.backward()
        disc_opm.step()
    else:
        disc.eval()
        logits, _ = disc(X)
        loss = nn.functional.cross_entropy(
            logits.permute(0, 2, 1), y.unsqueeze(-1).repeat(1, logits.shape[1])
        )

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

    obs = AGENT1(
        time_step=cfg.obs_time_step_s,
        max_silence_s=cfg.obs_max_silence_s,
        zero_init=False,
    ).to(device)
    discriminator = discriminator.to(device)
    discriminator_orig = discriminator_orig.to(device)

    optim = torch.optim.Adam(obs.parameters(), lr=0.001)
    disc_optim = torch.optim.Adam(discriminator.parameters(), lr=0.001)

    e = 0
    detach_period = 50
    with mlflow.start_run(log_system_metrics=True):
        train_disc = True
        while True:
            reward_scales = {"clf_scale": 1.0, "padding_scale": -0.0001}
            losses_metrics_d: dict[str, list[float]] = {
                "loss": [],
                "policy_loss": [],
                "value_loss": [],
                "avg_return": [],
                "entropy_loss": [],
                "disc_loss": [],
                "disc_acc": [],
                "mean_padding_count": [],
                "mean_trace_len": [],
            }
            losses_metrics_d.update(
                {"mean_reward_" + k.replace("_scale", ""): [] for k in reward_scales}
            )

            with tqdm(
                dl_train,
                desc=f"epoch {e:02d}",
                ncols=2 * TQDM_W,
            ) as pbar:
                for X, y in pbar:
                    X = dict_to_device(X, device)
                    y = y.to(device)

                    optim.zero_grad()
                    log_ps, values, rewards, entropies, _, _, Xobs = rollout(
                        obs=obs,
                        disc=discriminator,
                        X=X,
                        y=y,
                        detach_period=detach_period,
                        reward_scales=reward_scales,
                    )

                    G, advantages = get_advantages(rewards, values, cfg)

                    # Compute losses
                    policy_loss = -(log_ps * advantages.detach()).mean()
                    value_loss = 0.5 * (values - G).pow(2).mean()
                    entropy_loss = -entropies.mean()

                    loss = policy_loss + value_loss + 0.05 * entropy_loss

                    loss.backward()

                    # Sanity checks:
                    # ==========================================
                    for k, v in rewards.items():
                        assert_finite(f"rewards-{k}", v)
                    assert_finite("values", values)
                    assert_finite("log_ps", log_ps)

                    for k, v in losses_metrics_d.items():
                        if len(v) == 0:
                            continue

                        try:
                            assert_finite(k, v[-1])
                        except ValueError as er:
                            print(er)
                            breakpoint()

                    for name, p in obs.named_parameters():
                        if p.grad is not None:
                            try:
                                assert_finite(f"{name}: grad", p.grad)
                            except ValueError:
                                print(value_loss, policy_loss, entropy_loss)
                                breakpoint()

                    if not torch.isclose(
                        rewards["padding"].sum(dim=1),
                        -Xobs[Feats.PADDING].sum(dim=1)
                        * reward_scales["padding_scale"],
                    ).all():
                        logger.warning("padding rewards issues")
                    # ==========================================

                    # Gradient clipping
                    nn.utils.clip_grad_norm_(
                        obs.parameters(), cfg.grad_norm_clip, error_if_nonfinite=True
                    )

                    optim.step()

                    disc_loss, acc = one_batch_train_disc(
                        disc=discriminator,
                        X=Xobs,
                        y=y,
                        disc_opm=disc_optim,
                        train=train_disc,
                    )

                    losses_metrics_d["loss"].append(loss.item())
                    losses_metrics_d["policy_loss"].append(policy_loss.item())
                    losses_metrics_d["value_loss"].append(value_loss.item())
                    losses_metrics_d["avg_return"].append(G.mean().item())
                    losses_metrics_d["entropy_loss"].append(entropy_loss.item())
                    losses_metrics_d["disc_acc"].append(acc)
                    losses_metrics_d["disc_loss"].append(disc_loss)
                    losses_metrics_d["mean_padding_count"].append(
                        Xobs[Feats.PADDING].sum(dim=1).float().mean().item()
                    )
                    losses_metrics_d["mean_trace_len"].append(
                        (Xobs[Feats.DIRS] != 0).sum(dim=1).float().mean().item()
                    )
                    if rewards is not None:
                        for k, v in rewards.items():
                            losses_metrics_d[f"mean_reward_{k}"].append(v.mean().item())

                    pbar.set_postfix(
                        {
                            "avg_return": np.mean(losses_metrics_d["avg_return"][-30:]),
                            "dacc": np.mean(losses_metrics_d["disc_acc"][-30:]),
                        }
                    )

            mlflow.log_metrics(
                {k: np.mean(l_) for k, l_ in losses_metrics_d.items()}, step=e
            )

            losses_metrics_d = {k: [] for k in losses_metrics_d}

            if e % 10 == 0:
                mlflow.pytorch.log_model(obs, name=f"rlobs-{e}")
                mlflow.pytorch.log_model(discriminator, name=f"rldisc-{e}")

            _plot_set(
                ds=ds_valid,
                obs=obs,
                clf_orig=discriminator_orig,
                clf_trained=discriminator,
                e=e,
                reward_scales=reward_scales,
                device=device,
                ntraces=20,
            )

            e += 1


if __name__ == "__main__":
    main()
