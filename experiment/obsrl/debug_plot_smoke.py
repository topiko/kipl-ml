"""Smoke test for obsrl plotting utilities.

This exercises experiment/obsrl/plot_utils._plot_single with synthetic tensors to
ensure recent invariant and axis-sharing changes don't crash.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import torch
from omegaconf import OmegaConf

from experiment.obsrl import plot_utils
from kipl_ml.data.utils import DOWNLOAD, UPLOAD
from kipl_ml.rl.enums import Actions
from kipl_ml.trace.enums import Feats


class _IdentityFeatures:
    def __call__(self, x: dict[Feats, torch.Tensor]) -> dict[Feats, torch.Tensor]:
        return x

    def transform_batch(
        self, x: dict[Feats, torch.Tensor]
    ) -> dict[Feats, torch.Tensor]:
        return x


class _DummyDisc(torch.nn.Module):
    def __init__(self, feat_mode: str = "dir"):
        super().__init__()
        self.feat_mode = feat_mode
        self.tam_dict = {"window_width_s": 0.02}

    def forward(self, x):
        raise RuntimeError("Not used")


def main() -> None:
    # Disable MLflow logging.
    plot_utils.mlflow.log_figure = lambda *args, **kwargs: None

    device = torch.device("cpu")
    B = 1
    L = 10
    T = 5
    C = 30

    # Trace dicts.
    times_trace = torch.linspace(0.0, 0.18, L).unsqueeze(0)
    dirs_trace = torch.tensor([[UPLOAD, DOWNLOAD] * (L // 2) + [UPLOAD] * (L % 2)], dtype=torch.float32)
    pad_trace = torch.zeros((B, L))

    X_orig = {Feats.TIMES: times_trace, Feats.DIRS: dirs_trace, Feats.PADDING: pad_trace}
    X_obs = {Feats.TIMES: times_trace.clone(), Feats.DIRS: dirs_trace.clone(), Feats.PADDING: pad_trace.clone()}

    # Obs features.
    fd = {
        Feats.TIMES: torch.linspace(0.0, 0.08, T).unsqueeze(0),
        Feats.Dt: torch.full((B, T), 0.02),
        Feats.UP_COUNT: torch.zeros((B, T)),
        Feats.DOWN_COUNT: torch.zeros((B, T)),
    }

    # Actions/times.
    act_times = fd[Feats.TIMES] + fd[Feats.Dt]  # (B,T)
    actions = {
        Actions.DELAY: torch.tensor([[0.0, 0.02, 0.02, 0.0, 0.0]]),
        Actions.DO_NOTHING: torch.zeros((B, T)),
        Actions.SELECTOR: torch.zeros((B, T)),
        Actions.SEND_COUNT_UP: torch.zeros((B, T)),
        Actions.SEND_COUNT_DOWN: torch.zeros((B, T)),
        Actions.SEND_UP_AFTER_TIME: torch.zeros((B, T)),
        Actions.SEND_DOWN_AFTER_TIME: torch.zeros((B, T)),
    }

    sel_probs = torch.full((B, T, 5), 1.0 / 5.0)
    values = torch.zeros((B, T))
    entropies = {
        "selection_entropy": torch.zeros((B, T)),
        "conditional_entropy": torch.zeros((B, T)),
    }

    weights = torch.ones((1,))
    G = torch.zeros((1, B, T))
    advantages = torch.zeros((1, B, T))
    league_rewards = {
        "clf": torch.zeros((1, B, T)),
        "padding": torch.zeros((1, B, T)),
    }

    probs_orig = torch.zeros((B, L, C))
    probs_obs = torch.zeros((B, L, C))
    probs_orig[:, :, 0] = 1.0
    probs_obs[:, :, 0] = 1.0

    plot_utils._plot_single(
        cfg=OmegaConf.create({}),
        disc_orig=_DummyDisc("dir"),
        disc_trained=_DummyDisc("dir"),
        disc_features=_IdentityFeatures(),
        e=1,
        device=device,
        ds_idx=0,
        batch_i=0,
        X_orig=X_orig,
        y_orig=torch.tensor(0),
        sel_probs=sel_probs,
        probs_orig=probs_orig,
        probs_obs=probs_obs,
        values=values,
        league_rewards=league_rewards,
        entropies=entropies,
        times=act_times,
        actions=actions,
        X_obs=X_obs,
        fd=fd,
        G=G,
        advantages=advantages,
        weights=weights,
    )

    print("OK: plot_utils smoke")


if __name__ == "__main__":
    main()
