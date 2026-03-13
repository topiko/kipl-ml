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
from kipl_ml.rl.observation import get_window_feature_dict
from kipl_ml.rl.enums import Actions
from kipl_ml.trace.enums import Feats


class _IdentityFeatures:
    def __init__(self, n_tam_bins: int = 10):
        self.n_tam_bins = n_tam_bins

    def __call__(self, x: dict[Feats, torch.Tensor]) -> dict[Feats, torch.Tensor]:
        out = dict(x)
        B = x[Feats.TIMES].shape[0]
        out[Feats.TAM_TIMES] = torch.linspace(0.0, 0.18, self.n_tam_bins).unsqueeze(0).expand(B, -1)
        out[Feats.TAM_UP_COUNTS] = torch.ones((B, self.n_tam_bins), dtype=torch.long)
        out[Feats.TAM_DOWN_COUNTS] = torch.ones((B, self.n_tam_bins), dtype=torch.long)
        return out

    def transform_batch(
        self, x: dict[Feats, torch.Tensor]
    ) -> dict[Feats, torch.Tensor]:
        return self(x)


class _DummyDisc(torch.nn.Module):
    def __init__(self, feat_mode: str = "tam"):
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
    C = 30
    dt = 0.02
    max_silence_s = 0.1

    # Trace dicts.
    times_trace = torch.linspace(0.0, 0.18, L).unsqueeze(0)
    dirs_trace = torch.tensor([[UPLOAD, DOWNLOAD] * (L // 2) + [UPLOAD] * (L % 2)], dtype=torch.float32)
    pad_trace = torch.zeros((B, L))

    X_orig = {Feats.TIMES: times_trace, Feats.DIRS: dirs_trace, Feats.PADDING: pad_trace}
    X_obs = {Feats.TIMES: times_trace.clone(), Feats.DIRS: dirs_trace.clone(), Feats.PADDING: pad_trace.clone()}

    # Obs features consistent with X_obs.
    fd = get_window_feature_dict(
        X_obs,
        dt=dt,
        max_silence_s=max_silence_s,
        features=[Feats.TIME_BINS, Feats.Dt_BINS, Feats.UP_COUNT, Feats.DOWN_COUNT],
    )
    T = int(fd[Feats.TIME_BINS].shape[1])

    # Actions/times.
    act_times = fd[Feats.TIME_BINS] + fd[Feats.Dt_BINS]  # (B,T)
    actions = {
        Actions.DELAY_BINS: torch.zeros((B, T), dtype=torch.long),
        Actions.DO_NOTHING: torch.zeros((B, T)),
        Actions.SELECTOR: torch.zeros((B, T)),
        Actions.SEND_COUNT_UP: torch.zeros((B, T)),
        Actions.SEND_COUNT_DOWN: torch.zeros((B, T)),
        Actions.SEND_UP_AFTER_BINS: torch.zeros((B, T), dtype=torch.long),
        Actions.SEND_DOWN_AFTER_BINS: torch.zeros((B, T), dtype=torch.long),
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
        disc_orig=_DummyDisc("tam"),
        disc_trained=_DummyDisc("tam"),
        disc_features=_IdentityFeatures(n_tam_bins=10),
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
        obs_dt_s=dt,
    )

    print("OK: plot_utils smoke")


if __name__ == "__main__":
    main()
