"""Post-transition observations at finite rollout boundaries."""

import torch

from kipl_ml.trace.enums import Feats


class RolloutEnd:
    """Optional collector enabling hard cutoffs and normal-packet termination.

    Each feature row is the input to the *next* policy decision. No action is
    selected or executed at a cutoff. Simulator iteration exhaustion is an error.
    """

    def __init__(self, batch_size: int, device: torch.device):
        self.terminated = torch.zeros(batch_size, dtype=torch.bool, device=device)
        self.truncated = torch.zeros_like(self.terminated)
        self.values = torch.zeros(batch_size, device=device)
        self.times = torch.zeros(batch_size, device=device, dtype=torch.float64)
        self.features: dict[Feats, torch.Tensor] = {}
        self.reasons: list[str | None] = [None] * batch_size

    def record(
        self, index: int, features: dict[Feats, torch.Tensor],
        native_reason: str | None, cutoff: str | None, time_step_s: float,
    ) -> bool:
        if native_reason == "max_sim_steps":
            raise RuntimeError(
                f"simulator iteration budget exhausted for instance {index}"
            )
        if native_reason == "max_trace_length":
            raise RuntimeError("bootstrap requires a complete observation window")
        terminal = native_reason in {"all_normal_processed", "exhausted"}
        if not terminal and cutoff is None:
            return False
        self.terminated[index] = terminal
        self.truncated[index] = not terminal
        self.reasons[index] = native_reason if terminal else cutoff
        for key, value in features.items():
            value = value.detach().to(self.values.device).reshape(1)
            if key not in self.features:
                self.features[key] = torch.zeros(
                    (len(self.reasons), 1), device=value.device, dtype=value.dtype,
                )
            self.features[key][index] = value
        self.times[index] = (
            self.features[Feats.TIME_BINS][index, 0].double()
            + self.features[Feats.Dt_BINS][index, 0].double()
        ) * time_step_s
        return True

    def attach(self, fd: dict[Feats, torch.Tensor]) -> None:
        if any(reason is None for reason in self.reasons):
            raise RuntimeError("missing rollout endpoint")
        fd[Feats.TERMINATED] = self.terminated
        fd[Feats.TRUNCATED] = self.truncated
        fd[Feats.BOOTSTRAP_VALUE] = self.values.detach()

    def critic_inputs(
        self, fd: dict[Feats, torch.Tensor],
    ) -> dict[Feats, torch.Tensor]:
        """Append endpoint at each truncated row's own length, before padding."""
        lengths = fd[Feats.SEQ_LENS]
        rows = self.truncated.nonzero().flatten()
        result = {}
        for key, endpoint in self.features.items():
            value = fd[key]
            extended = torch.cat((value, torch.zeros_like(value[:, :1])), dim=1)
            extended[rows, lengths[rows]] = endpoint[rows, 0].to(value.dtype)
            result[key] = extended
        result[Feats.SEQ_LENS] = lengths + self.truncated.long()
        return result
