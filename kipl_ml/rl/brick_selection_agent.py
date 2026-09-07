from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn
from torch.distributions import Categorical

from kipl_ml.rl.enums import Actions, ActSelector, EntropyKeys, StepAction, StepActions
from kipl_ml.trace.enums import Feats

DEFAULT_BRICK_FEATURES = (Feats.TIME_BINS, Feats.Dt_BINS)


class BrickSelectionAgent(nn.Module):
    """Time-indexed transition policy over client/server brick indices.

    For each action time `t`, `client_transition_probs()[t, i, j]` is the
    probability of moving client brick `i` to `j`; server transitions are
    analogous. The module stores logits, not probabilities, so rows remain
    normalized while RL optimizes unconstrained parameters.
    """

    def __init__(
        self,
        time_step: float,
        time_steps: int,
        n_client_bricks: int,
        n_server_bricks: int | None = None,
        features: Sequence[Feats] = DEFAULT_BRICK_FEATURES,
        prob_eps: float = 0.0,
        stay_bias: float = 0.0,
        train_env: dict[str, object] | None = None,
    ) -> None:
        super().__init__()
        if time_steps <= 0:
            raise ValueError("time_steps must be > 0")
        if n_client_bricks <= 0:
            raise ValueError("n_client_bricks must be > 0")
        if n_server_bricks is None:
            n_server_bricks = n_client_bricks
        if n_server_bricks <= 0:
            raise ValueError("n_server_bricks must be > 0")
        if not (0.0 <= prob_eps <= 1.0):
            raise ValueError("prob_eps must be in [0, 1]")

        self.time_step = float(time_step)
        self.time_steps = int(time_steps)
        self.n_client_bricks = int(n_client_bricks)
        self.n_server_bricks = int(n_server_bricks)
        self.features = tuple(features)
        self.prob_eps = float(prob_eps)
        self.train_env = {} if train_env is None else dict(train_env)

        client_logits = torch.zeros(
            self.time_steps, self.n_client_bricks, self.n_client_bricks
        )
        server_logits = torch.zeros(
            self.time_steps, self.n_server_bricks, self.n_server_bricks
        )
        if stay_bias != 0.0:
            client_diag = torch.arange(self.n_client_bricks)
            server_diag = torch.arange(self.n_server_bricks)
            client_logits[:, client_diag, client_diag] = float(stay_bias)
            server_logits[:, server_diag, server_diag] = float(stay_bias)

        self.client_transition_logits = nn.Parameter(client_logits)
        self.server_transition_logits = nn.Parameter(server_logits)
        self.value_table = nn.Parameter(
            torch.zeros(self.time_steps, self.n_client_bricks, self.n_server_bricks)
        )

    def client_transition_probs(self) -> torch.Tensor:
        return self._transition_probs(
            self.client_transition_logits, self.n_client_bricks
        )

    def server_transition_probs(self) -> torch.Tensor:
        return self._transition_probs(
            self.server_transition_logits, self.n_server_bricks
        )

    def transition_probs(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.client_transition_probs(), self.server_transition_probs()

    def _transition_probs(self, logits: torch.Tensor, n_bricks: int) -> torch.Tensor:
        probs = torch.softmax(logits, dim=-1)
        if self.prob_eps == 0.0:
            return probs
        return (1.0 - self.prob_eps) * probs + self.prob_eps / n_bricks

    def act_step(
        self,
        x: dict[Feats, torch.Tensor],
        current_client_bricks: torch.Tensor,
        current_server_bricks: torch.Tensor,
        sample: bool = True,
    ) -> tuple[
        torch.Tensor,
        StepActions,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        dict[str, torch.Tensor],
        None,
    ]:
        time_bins = self._action_time_bins(x)
        current_client = self._current_bricks(
            current_client_bricks,
            time_bins.device,
            self.n_client_bricks,
            "current_client_bricks",
        )
        current_server = self._current_bricks(
            current_server_bricks,
            time_bins.device,
            self.n_server_bricks,
            "current_server_bricks",
        )
        if current_client.shape[0] != time_bins.shape[0]:
            raise ValueError("current_client_bricks length must match batch size")
        if current_server.shape[0] != time_bins.shape[0]:
            raise ValueError("current_server_bricks length must match batch size")

        time_idx = time_bins.squeeze(1).long().clamp(0, self.time_steps - 1)
        client_probs = self.client_transition_probs()[time_idx, current_client]
        server_probs = self.server_transition_probs()[time_idx, current_server]
        client_selected, client_log_ps, client_entropy = self._select(
            client_probs, sample
        )
        server_selected, server_log_ps, server_entropy = self._select(
            server_probs, sample
        )

        actions = [
            StepAction(
                int(time_bins[idx, 0].item()),
                {
                    Actions.CLIENT_BRICK_SELECT: ActSelector(
                        int(client_selected[idx].item())
                    ),
                    Actions.SERVER_BRICK_SELECT: ActSelector(
                        int(server_selected[idx].item())
                    ),
                },
            )
            for idx in range(time_bins.shape[0])
        ]
        entropy = client_entropy + server_entropy
        entropies = {
            EntropyKeys.SELECTION_ENTROPY: entropy,
            EntropyKeys.COND_ENTROPY: torch.zeros_like(entropy),
        }
        values = self.value_table[time_idx, current_client, current_server].unsqueeze(1)
        sel_probs = torch.cat([client_probs, server_probs], dim=-1).unsqueeze(1)
        return (
            time_bins,
            actions,
            client_log_ps + server_log_ps,
            sel_probs,
            values,
            entropies,
            None,
        )

    def _select(
        self, probs: torch.Tensor, sample: bool
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dist = Categorical(probs=probs)
        if sample:
            selected = dist.sample()
            log_ps = dist.log_prob(selected).unsqueeze(1)
        else:
            selected = probs.argmax(dim=-1)
            log_ps = torch.log(
                probs.gather(-1, selected.unsqueeze(1)).clamp(min=1e-12)
            )
        return selected, log_ps, dist.entropy().unsqueeze(1)

    def _action_time_bins(self, x: dict[Feats, torch.Tensor]) -> torch.Tensor:
        for feature in self.features:
            value = x[feature]
            if value.ndim != 2 or value.shape[1] != 1:
                raise ValueError(
                    "BrickSelectionAgent.act_step expects each feature to be (B,1)"
                )
        return x[Feats.TIME_BINS] + x[Feats.Dt_BINS]

    def _current_bricks(
        self,
        current_bricks: torch.Tensor,
        device: torch.device,
        n_bricks: int,
        name: str,
    ) -> torch.Tensor:
        if current_bricks.ndim != 1:
            raise ValueError(f"{name} must be a 1D tensor")
        current = current_bricks.to(device=device, dtype=torch.long)
        if ((current < 0) | (current >= n_bricks)).any():
            raise ValueError(f"{name} entries must be valid brick indices")
        return current
