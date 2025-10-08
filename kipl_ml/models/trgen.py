import torch
from torch import nn

from kipl_ml.trace.features import Feats


class _ConvBlock(nn.Module):
    def __init__(
        self, in_dim: int, out_dim: int, ks: int, stride: int = 1, padding: int = 0
    ):
        super().__init__()

        self.block = nn.Sequential(
            nn.ConvTranspose1d(
                in_dim, out_dim, ks, stride=stride, padding=padding, bias=False
            ),
            nn.BatchNorm1d(out_dim),
            nn.Tanh(),
        )

    def forward(self, x: torch.tensor) -> torch.tensor:
        print(x.shape)
        x = self.block(x)
        print(x.shape)
        print()
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
            # nn.Linear(seed_dim, 5000),
            _ConvBlock(seed_dim, 8 * expand_fac, ks=6, stride=1, padding=0),
            # size [B, 1024, 6]
            _ConvBlock(8 * expand_fac, 7 * expand_fac, ks=6, stride=2, padding=0),
            # size [B, 896, 16]
            _ConvBlock(7 * expand_fac, 6 * expand_fac, ks=6, stride=2, padding=0),
            # size [B, 768, 36]
            _ConvBlock(6 * expand_fac, 5 * expand_fac, ks=6, stride=2, padding=0),
            # size [B, 640, 76]
            _ConvBlock(5 * expand_fac, 4 * expand_fac, ks=6, stride=2, padding=0),
            # size [B, 512, 156]
            _ConvBlock(4 * expand_fac, 3 * expand_fac, ks=6, stride=2, padding=0),
            # size [B, 384, 316]
            _ConvBlock(3 * expand_fac, 2 * expand_fac, ks=6, stride=2, padding=0),
            # size [B, 256, 636]
            _ConvBlock(2 * expand_fac, 1 * expand_fac, ks=6, stride=2, padding=0),
            # size [B, 128, 1276]
            _ConvBlock(expand_fac, 64, ks=6, stride=2, padding=0),
            # size [B, 64, 2556]
            _ConvBlock(64, 1, ks=6, stride=2, padding=0),
            # size [B, 1, 5116]
            nn.Flatten(1, -1),
            # dim = 100
            # _ConvBlock(expand_fac, 2 * expand_fac, ks=21),
            # dim = 200 - 20 = 180
            # _ConvBlock(2 * expand_fac, 1, ks=21),
            # dim = 180 * 3 - 20 = 520
            # _ConvBlock(3 * expand_fac, 4 * expand_fac, ks=21),
            # dim = 1020
            # _ConvBlock(4 * expand_fac, 1, ks=101),
            nn.Tanh(),
        )

    def forward(self, seed: torch.Tensor) -> dict[Feats, torch.tensor]:
        gen_trace = self.generator(seed).squeeze(1)

        return {Feats.DIRS: gen_trace}


class TRGEN2(nn.Module):
    name: str = "trgen2"

    def __init__(
        self,
        features: list[Feats],
        in_channels: int = 2,
        hsize: int = 128,
        nlayer: int = 2,
    ):
        super().__init__()

        self.generator = nn.Sequential(
            nn.Tanh(),
        )

        self.rnn = nn.LSTM(in_channels, hsize, nlayer, batch_first=True)

        self.dir_lin = nn.Linear(hsize, 3)
        self.len_lin = nn.Linear(hsize, 1)

    def forward(
        self, x: dict[Feats, torch.Tensor], h: torch.Tensor | None = None
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], torch.Tensor | None]:
        dirs = x[Feats.BURST_DIRS].unsqueeze(-1)
        lens = x[Feats.BURST_LENS].unsqueeze(-1)

        X = torch.cat([dirs, lens], dim=-1)
        # (N, L, H)
        output, h = self.rnn(X, h)

        dirs = self.dir_lin(output).squeeze(-1)

        lens = self.len_lin(output).squeeze(-1)
        lens = torch.relu(lens) + 1

        return (dirs, lens), h
