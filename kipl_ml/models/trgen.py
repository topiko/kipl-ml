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
    ):
        if features != [Feats.DIRS]:
            raise ValueError("TRGEN1 only supports 'dirs' feature.")

        super().__init__()

        self.features = features
        self.trace_len = trace_len

        self.generator = nn.Sequential(
            nn.ConvTranspose1d(
                1,
                1,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.LeakyReLU(),
            nn.Linear(seed_dim, 2 * seed_dim),
            nn.LeakyReLU(),
            nn.ConvTranspose1d(1, 1, kernel_size=5, stride=1, padding=2),
            nn.LeakyReLU(),
            nn.Linear(2 * seed_dim, 10 * seed_dim),
            nn.LeakyReLU(),
            nn.ConvTranspose1d(1, 1, kernel_size=5, stride=1, padding=2),
            nn.LeakyReLU(),
            nn.Linear(10 * seed_dim, trace_len),
            nn.LeakyReLU(),
            nn.ConvTranspose1d(1, 1, kernel_size=5, stride=1, padding=2),
            nn.Hardtanh(),
        )

    def forward(self, seed: torch.Tensor) -> dict[Feats, torch.tensor]:
        seed = seed.unsqueeze(1)
        gen_trace = self.generator(seed).squeeze()


        return {Feats.DIRS: gen_trace}
