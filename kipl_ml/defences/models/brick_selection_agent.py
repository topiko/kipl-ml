from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn
from torch.distributions import Categorical

from kipl_ml.defences.models._time import (
    get_time_step_s_attr,
    resolve_time_step_s,
    set_time_step_s_attr,
)
from kipl_ml.rl.enums import Actions, ActSelector, EntropyKeys, StepAction, StepActions
from kipl_ml.trace.enums import Feats

DEFAULT_BRICK_FEATURES = (Feats.TIME_BINS, Feats.Dt_BINS)
SUPPORTED_BRICK_FEATURES = {
    Feats.TIME_BINS,
    Feats.Dt_BINS,
    Feats.UP_COUNT,
    Feats.DOWN_COUNT,
    Feats.UP_DECOY_COUNT,
    Feats.DOWN_DECOY_COUNT,
    Feats.SILENCE_FLAG,
}


def _validate_features(features: Sequence[Feats]) -> tuple[Feats, ...]:
    features = tuple(features)
    unsupported = set(features) - SUPPORTED_BRICK_FEATURES
    if unsupported:
        names = sorted(map(str, unsupported))
        raise ValueError(f"unsupported brick features: {names}")
    return features


def _resolve_n_time_steps(
    n_time_steps: int | None,
    time_steps: int | None,
) -> int:
    if n_time_steps is None:
        if time_steps is None:
            raise TypeError("n_time_steps is required")
        return int(time_steps)
    if time_steps is not None and int(n_time_steps) != int(time_steps):
        raise ValueError("time_steps and n_time_steps must match")
    return int(n_time_steps)


def _validate_init(
    n_time_steps: int,
    n_client_bricks: int | None,
    n_server_bricks: int | None,
    prob_eps: float,
    feature_hidden_size: int,
) -> tuple[int, int]:
    if n_time_steps <= 0:
        raise ValueError("n_time_steps must be > 0")
    if n_client_bricks is None:
        raise TypeError("n_client_bricks is required")
    if n_client_bricks <= 0:
        raise ValueError("n_client_bricks must be > 0")
    n_server_bricks = n_client_bricks if n_server_bricks is None else n_server_bricks
    if n_server_bricks <= 0:
        raise ValueError("n_server_bricks must be > 0")
    if not 0.0 <= prob_eps <= 1.0:
        raise ValueError("prob_eps must be in [0, 1]")
    if feature_hidden_size <= 0:
        raise ValueError("feature_hidden_size must be > 0")
    return int(n_client_bricks), int(n_server_bricks)


class BrickSelectionAgent(nn.Module):
    """Feature-conditioned transition policy over client/server brick indices.

    For each action time `t`, `client_transition_probs()[t, i, j]` is the
    baseline probability of moving client brick `i` to `j`; server transitions
    are analogous. A reactive MLP adds residual logits and value based on the
    completed window and current brick state. Residual outputs start at zero, so
    initialization exactly matches the tabular policy. An empty feature list
    disables the residual MLP and leaves a static transition-table policy.
    """

    def __init__(
        self,
        time_step_s: float | None = None,
        n_time_steps: int | None = None,
        n_client_bricks: int | None = None,
        n_server_bricks: int | None = None,
        features: Sequence[Feats] = DEFAULT_BRICK_FEATURES,
        feature_hidden_size: int = 64,
        learn_values: bool = True,
        prob_eps: float = 0.0,
        stay_bias: float = 0.0,
        train_env: dict[str, object] | None = None,
        *,
        time_step: float | None = None,
        time_steps: int | None = None,
    ) -> None:
        super().__init__()
        time_step_s = resolve_time_step_s(time_step_s, time_step)
        n_time_steps = _resolve_n_time_steps(n_time_steps, time_steps)
        n_client_bricks, n_server_bricks = _validate_init(
            n_time_steps,
            n_client_bricks,
            n_server_bricks,
            prob_eps,
            feature_hidden_size,
        )

        self.time_step_s = time_step_s
        self.n_time_steps = n_time_steps
        self.n_client_bricks = n_client_bricks
        self.n_server_bricks = n_server_bricks
        self.features = _validate_features(features)
        self.feature_hidden_size = int(feature_hidden_size)
        self.learn_values = bool(learn_values)
        self.prob_eps = float(prob_eps)
        self.train_env = {} if train_env is None else dict(train_env)

        client_logits = torch.zeros(
            self.n_time_steps, self.n_client_bricks, self.n_client_bricks
        )
        server_logits = torch.zeros(
            self.n_time_steps, self.n_server_bricks, self.n_server_bricks
        )
        if stay_bias != 0.0:
            client_diag = torch.arange(self.n_client_bricks)
            server_diag = torch.arange(self.n_server_bricks)
            client_logits[:, client_diag, client_diag] = float(stay_bias)
            server_logits[:, server_diag, server_diag] = float(stay_bias)

        self.client_transition_logits = nn.Parameter(client_logits)
        self.server_transition_logits = nn.Parameter(server_logits)
        value_table = torch.zeros(
            self.n_time_steps, self.n_client_bricks, self.n_server_bricks
        )
        if self.learn_values:
            self.value_table = nn.Parameter(value_table)
        else:
            self.register_buffer("value_table", value_table)
        self.feature_encoder: nn.Sequential | None = None
        self.client_feature_head: nn.Sequential | None = None
        self.server_feature_head: nn.Sequential | None = None
        self.value_feature_head: nn.Sequential | None = None
        if self.features:
            self.feature_encoder = nn.Sequential(
                nn.Linear(len(self.features), self.feature_hidden_size),
                nn.Tanh(),
            )
            self.client_feature_head = self._feature_head(
                self.feature_hidden_size + self.n_client_bricks,
                self.n_client_bricks,
            )
            self.server_feature_head = self._feature_head(
                self.feature_hidden_size + self.n_server_bricks,
                self.n_server_bricks,
            )
            if self.learn_values:
                self.value_feature_head = self._feature_head(
                    self.feature_hidden_size
                    + self.n_client_bricks
                    + self.n_server_bricks,
                    1,
                )

    def _feature_head(self, input_size: int, output_size: int) -> nn.Sequential:
        output = nn.Linear(self.feature_hidden_size, output_size)
        nn.init.zeros_(output.weight)
        nn.init.zeros_(output.bias)
        head = nn.Sequential(
            nn.Linear(input_size, self.feature_hidden_size),
            nn.Tanh(),
            output,
        )
        return head

    @property
    def time_step_s(self) -> float:
        return get_time_step_s_attr(self)

    @time_step_s.setter
    def time_step_s(self, value: float) -> None:
        set_time_step_s_attr(self, value)

    @property
    def n_time_steps(self) -> int:
        value = self.__dict__.get("_n_time_steps")
        if value is None:
            value = self.__dict__["time_steps"]
        return int(value)

    @n_time_steps.setter
    def n_time_steps(self, value: int) -> None:
        value = int(value)
        self.__dict__["_n_time_steps"] = value
        self.__dict__["time_steps"] = value

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

        time_idx = time_bins.squeeze(1).long().clamp(0, self.n_time_steps - 1)
        client_logits = self.client_transition_logits[time_idx, current_client]
        server_logits = self.server_transition_logits[time_idx, current_server]
        values = self.value_table[time_idx, current_client, current_server].unsqueeze(1)
        if self.feature_encoder is not None:
            if (
                self.client_feature_head is None
                or self.server_feature_head is None
            ):
                raise RuntimeError("feature heads are not initialized")
            features = self.feature_encoder(self._feature_vector(x, time_bins))
            client_state = nn.functional.one_hot(
                current_client, num_classes=self.n_client_bricks
            ).to(features.dtype)
            server_state = nn.functional.one_hot(
                current_server, num_classes=self.n_server_bricks
            ).to(features.dtype)
            client_logits = client_logits + self.client_feature_head(
                torch.cat((features, client_state), dim=1)
            )
            server_logits = server_logits + self.server_feature_head(
                torch.cat((features, server_state), dim=1)
            )
            if self.value_feature_head is not None:
                values = values + self.value_feature_head(
                    torch.cat((features, client_state, server_state), dim=1)
                )
        client_probs = self._transition_probs(client_logits, self.n_client_bricks)
        server_probs = self._transition_probs(server_logits, self.n_server_bricks)
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
        required = tuple(
            dict.fromkeys((Feats.TIME_BINS, Feats.Dt_BINS, *self.features))
        )
        for feature in required:
            value = x[feature]
            if value.ndim != 2 or value.shape[1] != 1:
                raise ValueError(
                    "BrickSelectionAgent.act_step expects each feature to be (B,1)"
                )
        return x[Feats.TIME_BINS] + x[Feats.Dt_BINS]

    def _feature_vector(
        self,
        x: dict[Feats, torch.Tensor],
        time_bins: torch.Tensor,
    ) -> torch.Tensor:
        values = []
        for feature in self.features:
            value = x[feature].to(torch.float32)
            if feature == Feats.TIME_BINS:
                time_s = time_bins.to(torch.float32) * self.time_step_s
                value = time_s / (time_s + 10.0)
            elif feature == Feats.Dt_BINS:
                value = torch.log1p(value * self.time_step_s)
            elif feature != Feats.SILENCE_FLAG:
                value = torch.log1p(value)
            values.append(value)
        return torch.cat(values, dim=1)

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
