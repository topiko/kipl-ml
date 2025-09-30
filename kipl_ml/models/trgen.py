import torch
from torch import nn

from kipl_ml.trace.features import Feats


class TRGEN1(nn.Module):
    name: str = "trgen1"

    def __init__(
        self,
        trace_len: int,
        features: list[Feats],
        seed_dim: int = 100,
        latent_dim: int = 500,
    ):
        if features != [Feats.DIRS]:
            raise ValueError("TRGEN1 only supports 'dirs' feature.")

        super().__init__()

        self.features = features
        self.trace_len = trace_len

        self.generator = nn.Sequential(
            nn.Linear(seed_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Linear(2048, trace_len),
            nn.Tanh(),
        )

    def forward(self, seed: torch.Tensor) -> dict[Feats, torch.tensor]:
        gen_trace = self.generator(seed)
        return {Feats.DIRS: gen_trace}
