import torch
from torch import nn

from kipl_ml.trace.features import Feats


class _ConvBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, ks: int):
        super().__init__()

        self.block = nn.Sequential(
            nn.ConvTranspose1d(in_dim, out_dim, ks, padding=(ks - 1) // 2),
            nn.BatchNorm1d(out_dim),
            nn.Tanh(),
        )

    def forward(self, x: torch.tensor) -> torch.tensor:
        # print(x.shape)
        x = self.block(x)
        # print(x.shape)
        # print()
        return x


class TRGEN1(nn.Module):
    name: str = "trgen1"

    def __init__(
        self,
        trace_len: int,
        features: list[Feats],
        seed_dim: int = 100,
        expand_fac: int = 128,
    ):
        if features != [Feats.DIRS]:
            raise ValueError("TRGEN1 only supports 'dirs' feature.")

        super().__init__()

        self.features = features
        self.trace_len = trace_len

        self.generator = nn.Sequential(
            _ConvBlock(1, expand_fac, ks=11),
            # dim = 100
            _ConvBlock(expand_fac, 2 * expand_fac, ks=21),
            # dim = 200 - 20 = 180
            _ConvBlock(2 * expand_fac, 1, ks=21),
            # dim = 180 * 3 - 20 = 520
            #_ConvBlock(3 * expand_fac, 4 * expand_fac, ks=21),
            # dim = 1020
            #_ConvBlock(4 * expand_fac, 1, ks=101),
            nn.Tanh(),
        )

    def forward(self, seed: torch.Tensor) -> dict[Feats, torch.tensor]:
        seed = seed.unsqueeze(1)

        gen_trace = self.generator(seed).squeeze()

        return {Feats.DIRS: gen_trace}
