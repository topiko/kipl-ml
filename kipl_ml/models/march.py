from __future__ import annotations

import torch
from torch import nn

from kipl_ml.models.utils import unsqueeze_batch
from kipl_ml.trace.features import Feats


class TrMarchBlock(nn.Module):
    def __init__(
        self,
        seq_len: int,
        in_channels: int,
        embed_dim: int = 64,
        tr_kwargs: dict | None = None,
    ):

        super().__init__()

        tr_kwargs = tr_kwargs or {}

        self.pos_embedding = nn.Embedding(seq_len, in_channels)
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=in_channels,
            nhead=in_channels,
            dim_feedforward=tr_kwargs.get("dim_feedforward", 256),
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


class CNNMarchBlock(nn.Module):
    def __init__(
        self,
        seq_len: int,
        in_channels: int,
        embed_dim: int,
    ):

        super().__init__()


class March(nn.Module):
    name: str = "march"

    def __init__(
        self,
        n_classes: int,
        in_channels: int,
        input_len: int,
        step_len: int,
        step_stride: int,
        embed_dim: int = 64,
        tr_kwargs: dict | None = None,
        rnn_kwargs: dict | None = None,
        verify_inputs: bool = False,
    ):
        super().__init__()

        self.use_vmap = False
        self.march_block = TrMarchBlock(
            seq_len=step_len,
            in_channels=in_channels,
            embed_dim=embed_dim,
            tr_kwargs=tr_kwargs,
        )
        self.stride = step_stride
        self.step_len = step_len
        self.input_len = input_len
        self.verify_inputs = verify_inputs

        if self.input_len % self.stride != 0:
            raise ValueError("Expected input_len to be divisible by stride")

        rnn_kwargs = rnn_kwargs or {}
        bidir = False
        h_size = rnn_kwargs.get("hidden_size", 512)
        self.rnn = nn.LSTM(
            input_size=embed_dim,
            batch_first=True,
            bidirectional=bidir,
            hidden_size=h_size,
            num_layers=rnn_kwargs.get("num_layers", 2),
        )

        self.rnn_out_dim = h_size * 2 if bidir else h_size

        self.linear = nn.Linear(self.rnn_out_dim, n_classes)

    def forward(self, x: dict[str, torch.tensor]) -> torch.tensor:

        dirs = x[Feats.DIRS]
        # dir == 0 marks the point where the packets ended.
        seq_lens = (dirs == 0).int().argmax(dim=1) // self.stride
        seq_lens = torch.clip(seq_lens, 1, self.input_len // self.stride - 2)

        dirs = dirs.unfold(1, size=self.step_len, step=self.stride)
        times = x[Feats.TIMES].unfold(1, size=self.step_len, step=self.stride)
        # times/dirs shape (batch_size, input_len / stride, step_len)

        # Normalize times:
        t_means = times.mean(axis=2).unsqueeze(2)
        t_stds = times.std(axis=2).unsqueeze(2)
        # shape (batch_size, input_len / stride, 1)

        times = torch.where(
            t_stds != 0, (times - t_means) / t_stds, torch.zeros_like(times)
        )

        x_ = torch.cat((dirs.unsqueeze(3), times.unsqueeze(3)), dim=3)
        # x shape: (batch_size, input_len / stride, step_len, in_channels)

        if self.use_vmap:
            embeds = torch.vmap(
                self.march_block, in_dims=1, out_dims=1, randomness="same"
            )(x_)
            # embeds shape: (batch_size, input_len / stride, embed_dim)

        else:
            embeds = []
            for i in range(x_.shape[1]):
                embeds.append(self.march_block(x_[:, i, ...]).unsqueeze(1))

            embeds = torch.cat(embeds, dim=1)
            # embeds shape: (batch_size, input_len / stride, embed_dim)

        x_, _ = self.rnn(embeds)
        # x_ shape: (batch_size, input_len / stride, rnn_out_dim)

        x_ = self.linear(x_)
        # x_ shape: (batch_size, input_len / stride, n_classes)

        x_ = x_[torch.arange(len(seq_lens)), seq_lens, :]
        # x_ shape: (batch_size, n_classes)

        return x_

    def example_input(self, x: dict[str, torch.tesnor]) -> dict[str, torch.tensor]:
        return unsqueeze_batch(x)
