from __future__ import annotations

import torch
from torch import nn

from kipl_ml.trace.features import Feats


class MarchBlock(nn.Module):
    def __init__(
        self,
        seq_len: int,
        in_channels: int,
        embed_dim: int = 12,
        tr_kwargs: dict | None = None,
    ):

        super().__init__()

        tr_kwargs = tr_kwargs or {}

        self.pos_embedding = nn.Embedding(seq_len, in_channels)
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=in_channels,
            nhead=in_channels,
            dim_feedforward=tr_kwargs.get("dim_feedforward", 2048),
            dropout=tr_kwargs.get("dropout", 0.1),
            batch_first=True,
        )

        self.lin_layer = nn.Linear(seq_len * in_channels, embed_dim)

    def forward(self, x: torch.tensor) -> torch.tensor:
        # x shape: (batch_size, seq_len, in_channels)
        x = x + self.pos_embedding(torch.arange(x.shape[1], device=x.device)).unsqueeze(
            0
        )
        x = self.encoder_layer(x)
        x = x.flatten(1)
        x = self.lin_layer(x)
        # out shape: (batch_size, embed_dim)
        return x


class March(nn.Module):
    name: str = "march"

    def __init__(
        self,
        n_classes: int,
        in_channels: int,
        input_len: int,
        step_len: int,
        step_stride: int,
        embed_dim: int = 12,
        tr_kwargs: dict | None = None,
        rnn_kwargs: dict | None = None,
        verify_inputs: bool = False,
    ):
        super().__init__()

        self.march_block = MarchBlock(
            seq_len=step_len,
            in_channels=in_channels,
            embed_dim=embed_dim,
            tr_kwargs=tr_kwargs,
        )
        self.stride = step_stride
        self.step_len = step_len
        self.input_len = input_len
        self.verify_inputs = verify_inputs

        if self.input_len % self.step_len != 0:
            raise ValueError("Expected input_len to be divisible by step_len")

        rnn_kwargs = rnn_kwargs or {}
        bidir = True
        h_size = rnn_kwargs.get("hidden_size", 64)
        self.rnn = nn.LSTM(
            input_size=embed_dim,
            batch_first=True,
            bidirectional=bidir,
            hidden_size=h_size,
            num_layers=rnn_kwargs.get("num_layers", 1),
        )

        self.rnn_out_dim = h_size * 2 if bidir else h_size

        self.linear = nn.Linear(self.rnn_out_dim, n_classes)

    def forward(self, x: dict[str, torch.tensor]) -> torch.tensor:

        dirs = x[Feats.DIRS]
        seq_lens = (dirs == 0).int().argmax(dim=1) // self.stride + 1
        seq_lens = torch.clip(seq_lens, 1, self.input_len // self.stride - 2)

        x_ = torch.cat([x[k].unsqueeze(2) for k in x], dim=2)
        # x shape: (batch_size, seq_len, in_channels)

        x_ = x_.unfold(1, size=self.step_len, step=self.stride)
        # x shape: (batch_size, input_len / stride, in_channels, step_len)

        x_ = x_.permute(0, 1, 3, 2)
        # x shape: (batch_size, input_len / stride, step_len, in_channels)

        embeds = torch.vmap(self.march_block, in_dims=1, out_dims=1)(x_)
        # embeds shape: (batch_size, input_len / stride, embed_dim)

        x_, _ = self.rnn(embeds)
        # x_ shape: (batch_size, input_len / stride, rnn_out_dim)

        x_ = self.linear(x_)
        # x_ shape: (batch_size, input_len / stride, n_classes)

        x_ = x_[:, seq_lens, :]
        # x_ shape: (batch_size, n_classes)

        return x_
