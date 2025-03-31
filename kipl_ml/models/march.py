from __future__ import annotations

from collections import OrderedDict

import torch
from torch import nn

from kipl_ml.logging.logger import get_logger
from kipl_ml.models.utils import unsqueeze_batch
from kipl_ml.trace.features import Feats

logger = get_logger(__name__)


class CNNMarchBlock(nn.Module):
    def __init__(
        self,
        seq_len: int,
        in_channels: int,
        embed_dim: int = 64,
        cnn_kwargs: dict | None = None,
    ):

        super().__init__()

        cnn_kwargs = cnn_kwargs or {}
        ks = 30
        st = 10

        if ks > seq_len:
            raise ValueError("Kernel size must be smaller than sequence length")

        n_filters = cnn_kwargs.get("n_conv_filters", [32])
        n_filters.insert(0, in_channels)
        d = OrderedDict()
        t_len = seq_len
        for i in range(1, len(n_filters)):
            if (t_len := (t_len - ks) // st + 1) <= 0:
                raise ValueError("Invalid feat extr pipe.")
            d[f"conv_{i}"] = nn.Sequential(
                nn.Conv1d(
                    n_filters[i - 1],
                    out_channels=n_filters[i],
                    kernel_size=ks,
                    stride=st,
                ),
                nn.LayerNorm([n_filters[i], t_len]),
                nn.ReLU(),
                nn.Dropout(0.5),
            )

        logger.info("Time steps for lin layer %d", t_len)

        if (linear_in := t_len * n_filters[-1]) < embed_dim:
            raise ValueError(
                f"Embed dim {embed_dim} is larger than linear in {linear_in} \
                             --> bottleneck in nn."
            )
        self.conv1d = nn.Sequential(d)

        self.clf = nn.Sequential(
            nn.Linear(linear_in, linear_in),
            nn.LayerNorm(linear_in),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(linear_in, linear_in),
            nn.LayerNorm(linear_in),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(linear_in, embed_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.tensor) -> torch.tensor:
        # x shape: (batch_size, seq_len, in_channels)
        x = self.conv1d(x.permute(0, 2, 1)).flatten(1)
        # x shape: (batch_size, linear_in)

        x = self.clf(x)
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
        embed_dim: int,
        rand_start: bool = True,
        cnn_kwargs: dict | None = None,
        rnn_kwargs: dict | None = None,
        verify_inputs: bool = False,
        vmap: bool = False,
    ):
        super().__init__()

        self.use_vmap = vmap
        self.normalize_dirs = False
        self.rand_start = rand_start
        self.march_block = CNNMarchBlock(
            seq_len=step_len,
            in_channels=in_channels,
            embed_dim=embed_dim,
            cnn_kwargs=cnn_kwargs,
        )
        self.stride = step_stride
        self.step_len = step_len
        self.input_len = input_len
        self.verify_inputs = verify_inputs
        self.max_w_seq_len = (self.input_len - self.step_len) // self.stride + 1

        if (self.input_len - self.step_len) % self.stride != 0:
            raise ValueError(
                "Expected (input_len ({input_len}) - step_len ({step_len})) \
                             % stride ({step_stride}) == 0"
            )

        rnn_kwargs = rnn_kwargs or {}
        bidir = False
        h_size = rnn_kwargs.get("hidden_size", 512)
        self.rnn = nn.LSTM(
            input_size=embed_dim,
            batch_first=True,
            bidirectional=bidir,
            hidden_size=h_size,
            num_layers=rnn_kwargs.get("num_layers", 1),
            dropout=rnn_kwargs.get("dropout", 0.5),
        )

        self.rnn_out_dim = h_size * 2 if bidir else h_size

        self.linear = nn.Linear(self.rnn_out_dim, n_classes)

    def train(self, mode: bool = True):
        if self.rand_start and mode:
            self._rand_start = True
        else:
            self._rand_start = False
        super().train(mode)

    def forward(self, x: dict[str, torch.tensor]) -> torch.tensor:

        max_w_seq_len = self.max_w_seq_len
        if self._rand_start:
            start = torch.randint(0, self.stride, size=(1,))[0]
            x = {k: v[:, start : -(self.stride - start)] for k, v in x.items()}
            max_w_seq_len -= 1

        dirs = x[Feats.DIRS]
        # dir == 0 marks the point where the packets ended.
        seq_lens = (dirs == 0).int().argmax(dim=1) // self.stride
        seq_lens = torch.clip(seq_lens, 0, max_w_seq_len - 1)

        # Dirs
        dirs = dirs.unfold(1, size=self.step_len, step=self.stride)
        # dirs shape (batch_size, input_len / stride, step_len)

        # "normalize" dirs
        if self.normalize_dirs:
            dir_means = dirs.mean(dim=2).unsqueeze(2)
            dir_stds = dirs.std(dim=2).unsqueeze(2)

            dirs = torch.where(
                dir_stds != 0, (dirs - dir_means) / dir_stds, torch.zeros_like(dirs)
            )
            # dirs shape (batch_size, input_len / stride, step_len)

        # Times
        times = x[Feats.TIMES].unfold(1, size=self.step_len, step=self.stride)
        # times shape (batch_size, input_len / stride, step_len)

        # Normalize times:
        t_means = times.mean(dim=2).unsqueeze(2)
        t_stds = times.std(dim=2).unsqueeze(2)
        # shape (batch_size, input_len / stride, 1)

        times = torch.where(
            t_stds != 0, (times - t_means) / t_stds, torch.zeros_like(times)
        )
        # times shape (batch_size, input_len / stride, step_len)

        x_ = torch.cat((dirs.unsqueeze(3), times.unsqueeze(3)), dim=3)
        # x shape: (batch_size, input_len / stride, step_len, in_channels)

        if self.use_vmap:
            embeds = torch.vmap(
                self.march_block, in_dims=1, out_dims=1, randomness="different"
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

        x_ = x_[torch.arange(x_.shape[0]), seq_lens, :]
        # x_ shape: (batch_size, n_classes)

        return x_

    def example_input(self, x: dict[str, torch.tesnor]) -> dict[str, torch.tensor]:
        return unsqueeze_batch(x)
