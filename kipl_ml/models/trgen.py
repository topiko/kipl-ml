from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import spectral_norm

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
        inlen: int = 10,
        hsize: int = 128,
        nlayer: int = 2,
    ):
        if features != [Feats.DIRS]:
            raise ValueError("TRGEN1 only supports 'dirs' feature.")

        super().__init__()

        self.generator = nn.Sequential(
            nn.Tanh(),
        )

        self.rnn = nn.LSTM(inlen, hsize, nlayer)

        self.lin = nn.Linear(hsize, 1)

        self.activation = nn.Tanh()

    def forward(
        self, dirs: torch.Tensor, h: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        dirs, h = self.rnn(dirs, h)

        dirs = self.lin(dirs).squeeze(-1)

        dirs = self.activation(dirs)

        return dirs, h


class TraceUpsampleBlock(nn.Module):
    """Progressively upsamples latent features while injecting style noise."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        latent_dim: int,
        *,
        scale_factor: float = 2.0,
        target_length: Optional[int] = None,
    ) -> None:
        super().__init__()

        if latent_dim <= 0:
            raise ValueError("latent_dim must be positive.")

        self.scale_factor = scale_factor
        self.target_length = target_length

        self.latent_proj = nn.Linear(latent_dim, out_channels)
        self.res_conv = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        fused_in_channels = in_channels + out_channels

        self.net = nn.Sequential(
            nn.Conv1d(fused_in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm1d(out_channels, affine=True),
            nn.GELU(),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm1d(out_channels, affine=True),
        )

        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError("Input to TraceUpsampleBlock must be 3D (B, C, L).")

        if self.target_length is not None:
            x = F.interpolate(x, size=self.target_length, mode="linear", align_corners=False)
        else:
            x = F.interpolate(
                x, scale_factor=self.scale_factor, mode="linear", align_corners=False
            )

        residual = self.res_conv(x)

        style = self.latent_proj(latent).unsqueeze(-1)
        style = style.expand(-1, -1, x.size(-1))
        x = torch.cat([x, style], dim=1)

        x = self.net(x)
        x = x + residual

        return self.activation(x)


class TemporalSelfAttention(nn.Module):
    """Lightweight temporal self-attention for long-range conditioning."""

    def __init__(self, channels: int, num_heads: int = 4) -> None:
        super().__init__()
        if channels % num_heads != 0:
            raise ValueError("channels must be divisible by num_heads for attention.")

        self.attn = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=num_heads,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError("TemporalSelfAttention expects input of shape (B, C, L).")

        x_t = x.permute(0, 2, 1)
        attn_out, _ = self.attn(x_t, x_t, x_t, need_weights=False)
        x_t = self.norm(x_t + attn_out)
        return x_t.permute(0, 2, 1)


class TRGENResNet(nn.Module):
    """Progressive residual generator tailored for DF discriminator traces."""

    name: str = "trgen_resnet"

    def __init__(
        self,
        trace_len: int,
        features: list[Feats],
        *,
        seed_dim: int = 128,
        base_channels: int = 128,
        min_initial_resolution: int = 64,
        min_channels: int = 32,
        attn_threshold: int = 512,
        attn_heads: int = 4,
        use_spectral_norm: bool = True,
    ) -> None:
        if features != [Feats.DIRS]:
            raise ValueError("TRGENResNet only supports the 'dirs' feature.")

        super().__init__()

        if trace_len <= 0:
            raise ValueError("trace_len must be positive.")

        if attn_heads <= 0:
            raise ValueError("attn_heads must be positive.")

        self.trace_len = trace_len
        self.features = features
        self.seed_dim = seed_dim

        initial_resolution = max(4, min(trace_len, min_initial_resolution))
        self.initial_resolution = initial_resolution

        self.proj = nn.Linear(seed_dim, base_channels * initial_resolution)

        blocks: list[TraceUpsampleBlock] = []
        channels_schedule = []

        current_channels = base_channels
        current_length = initial_resolution

        while current_length < trace_len:
            next_length = min(trace_len, current_length * 2)
            next_channels = max(min_channels, current_channels // 2)

            scale_factor = 2.0 if next_length != trace_len else 2.0
            target_length = None
            if next_length == trace_len and next_length != current_length * 2:
                target_length = trace_len

            block = TraceUpsampleBlock(
                current_channels,
                next_channels,
                latent_dim=seed_dim,
                scale_factor=scale_factor,
                target_length=target_length,
            )
            blocks.append(block)
            channels_schedule.append(next_channels)

            current_channels = next_channels
            current_length = next_length

        self.blocks = nn.ModuleList(blocks)

        if trace_len >= attn_threshold and current_channels >= attn_heads:
            num_heads = min(attn_heads, max(1, current_channels // 32))
            num_heads = max(1, num_heads)
            # Ensure divisibility
            while current_channels % num_heads != 0 and num_heads > 1:
                num_heads -= 1
            self.attention = TemporalSelfAttention(current_channels, num_heads=num_heads)
        else:
            self.attention = None

        conv_out = nn.Conv1d(current_channels, 1, kernel_size=3, padding=1)
        self.to_trace = nn.Sequential(
            spectral_norm(conv_out) if use_spectral_norm else conv_out,
            nn.Tanh(),
        )

    def forward(self, seed: torch.Tensor) -> dict[Feats, torch.Tensor]:
        if seed.dim() == 3 and seed.size(-1) == 1:
            seed = seed.squeeze(-1)
        if seed.dim() != 2:
            raise ValueError("Seed tensor must have shape (B, seed_dim) or (B, seed_dim, 1).")

        if seed.size(1) != self.seed_dim:
            raise ValueError(
                f"Expected seed_dim={self.seed_dim}, received seed_dim={seed.size(1)}."
            )

        batch_size = seed.size(0)
        x = self.proj(seed)
        x = x.view(batch_size, -1, self.initial_resolution)

        for block in self.blocks:
            x = block(x, seed)

        if x.size(-1) != self.trace_len:
            x = F.interpolate(x, size=self.trace_len, mode="linear", align_corners=False)

        if self.attention is not None:
            x = self.attention(x)

        trace = self.to_trace(x).squeeze(1)
        if trace.size(-1) != self.trace_len:
            trace = F.interpolate(
                trace.unsqueeze(1), size=self.trace_len, mode="linear", align_corners=False
            ).squeeze(1)

        return {Feats.DIRS: trace}
