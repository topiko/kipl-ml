import torch
from torch import nn
from torch.nn.utils.rnn import PackedSequence, pack_padded_sequence

from kipl_ml.rl.enums import Actions
from kipl_ml.trace.features import Feats


def _hidden_w_mask(
    h: tuple[torch.Tensor, ...],
    mask: torch.Tensor,
    hmasked: tuple[torch.Tensor, ...] | None = None,
) -> tuple[torch.Tensor, ...]:
    if hmasked is None:
        return tuple(h_[:, mask] for h_ in h)

    for i, h_ in enumerate(h):
        h_[:, mask] = hmasked[i]

    return h


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
        nfeat: int = 64,
        hsize: int = 128,
        nlayer: int = 2,
    ):
        super().__init__()

        self.rnn = nn.LSTM(nfeat, hsize, nlayer, batch_first=True)

        self.feat_lin = nn.Sequential(
            nn.Linear(in_channels, nfeat),
            nn.LayerNorm(nfeat),
            nn.GELU(),
        )

        self.dir_lin = nn.Linear(hsize, 3)
        self.len_lin = nn.Linear(hsize, 1)

    def forward(
        self, x: dict[Feats, torch.Tensor], h: torch.Tensor | None = None
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], torch.Tensor | None]:
        dirs = x[Feats.BURST_DIRS]
        lens = x[Feats.BURST_LENS]

        # (N, L, 2)
        X = torch.stack([dirs, lens], dim=-1)

        # (N, L, nfeat)
        X = self.feat_lin(X)

        # (N, L, H)
        output, h = self.rnn(X, h)

        dirs = self.dir_lin(output).squeeze(-1)

        lens = self.len_lin(output).squeeze(-1)
        lens = torch.relu(lens) + 1

        return (dirs, lens), h


class TRGEN3(nn.Module):
    name: str = "trgen3"

    def __init__(
        self,
        features: list[Feats],
        in_channels: int = 2,
        nfeat: int = 64,
        hsize: int = 128,
        nlayer: int = 2,
    ):
        super().__init__()

        self.rnn = nn.LSTM(nfeat + 1, hsize, nlayer, batch_first=True)

        self.feat_lin = nn.Sequential(
            nn.Linear(in_channels, nfeat),
            nn.LayerNorm(nfeat),
            nn.GELU(),
        )

        self.dir_lin = nn.Linear(hsize, 3)
        self.len_lin = nn.Linear(hsize, 1)

    def forward(
        self,
        x: dict[Feats, torch.Tensor],
        y: torch.Tensor,
        h: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        dirs = x[Feats.DIRS]
        iats = x[Feats.IATS_MAX_NORMALIZED]

        # (N, L, 2)
        X = torch.stack([dirs, iats], dim=-1)

        # (N, L, nfeat)
        X = self.feat_lin(X)

        y_ = y.reshape(-1, 1).repeat(1, X.shape[1]).unsqueeze(2)

        # (N, L, nfeat + 1)
        X = torch.cat([X, y_], dim=-1)

        # (N, L, H)
        output, h = self.rnn(X, h)

        dirs = self.dir_lin(output).squeeze(-1)

        return dirs, h


class TRGEN4(nn.Module):
    name: str = "trgen3"

    def __init__(
        self,
        features: list[Feats],
        nfeat: int = 64,
        hsize: int = 128,
        nlayer: int = 2,
        dir_activation: str = "gumbel_softmax",
    ):
        super().__init__()

        if set(features) != {Feats.DIR_PROBS, Feats.IATS_MAX_NORMALIZED}:
            raise ValueError(
                "TRGEN4 only supports 'dir_probs' and 'iats_max_normalized' features."
            )

        self.rnn = nn.LSTM(nfeat + 1, hsize, nlayer, batch_first=True)

        self.feat_lin = nn.Sequential(
            nn.Linear(4, nfeat),
            nn.LayerNorm(nfeat),
            nn.GELU(),
        )

        self.dir_lin = nn.Linear(hsize, 3)

        self.dir_activation = dir_activation
        self.len_lin = nn.Linear(hsize, 1)

    def forward(
        self,
        x: dict[Feats, torch.Tensor],
        y: torch.Tensor,
        h: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # (N, L, 3)
        dir_probs = x[Feats.DIR_PROBS]

        # (N, L, 1)
        iats = x[Feats.IATS_MAX_NORMALIZED].unsqueeze(-1)

        # (N, L, 4)
        X = torch.cat([dir_probs, iats], dim=-1)

        # (N, L, nfeat)
        X = self.feat_lin(X)

        y_ = y.reshape(-1, 1).repeat(1, X.shape[1]).unsqueeze(2)

        # (N, L, nfeat + 1)
        X = torch.cat([X, y_], dim=-1)

        # (N, L, H)
        output, h = self.rnn(X, h)

        dirs = self.dir_lin(output).squeeze(-1)

        if self.dir_activation == "gumbel_softmax":
            dir_probs = nn.functional.gumbel_softmax(dirs, tau=1.0, hard=False, dim=-1)
        else:
            raise NotImplementedError(f"Unknown dir_activation: {self.dir_activation}")

        dir_log_probs = torch.log(dir_probs)

        return dir_log_probs, h


class TRGEN5(nn.Module):
    name: str = "trgen5"

    def __init__(
        self,
        features: list[Feats],
        nfeat: int = 64,
        hsize: int = 128,
        nlayer: int = 2,
        dir_activation: str = "gumbel_softmax",
    ):
        super().__init__()

        if set(features) != {Feats.DIR_PROBS}:
            raise ValueError("TRGEN5 only supports 'dir_probs'")

        self.rnn = nn.LSTM(nfeat + 1, hsize, nlayer, batch_first=True)

        self.feat_lin = nn.Sequential(
            nn.Linear(3, nfeat),
            nn.LayerNorm(nfeat),
            nn.GELU(),
        )

        self.dir_lin = nn.Linear(hsize, 3)

        self.dir_activation = dir_activation
        self.len_lin = nn.Linear(hsize, 1)

    def forward(
        self,
        x: dict[Feats, torch.Tensor],
        y: torch.Tensor,
        h: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # (N, L, 3)
        dir_probs = x[Feats.DIR_PROBS]

        # (N, L, nfeat)
        X = self.feat_lin(dir_probs)

        y_ = y.reshape(-1, 1).repeat(1, X.shape[1]).unsqueeze(2)

        # (N, L, nfeat + 1)
        X = torch.cat([X, y_], dim=-1)

        # (N, L, H)
        output, h = self.rnn(X, h)

        dirs = self.dir_lin(output).squeeze(-1)

        if self.dir_activation == "gumbel_softmax":
            dir_probs = nn.functional.gumbel_softmax(dirs, tau=1.0, hard=False, dim=-1)
        else:
            raise NotImplementedError(f"Unknown dir_activation: {self.dir_activation}")

        return dir_probs, h


class RNNCLF1(nn.Module):
    name: str = "rnnclf1"

    def __init__(
        self,
        n_classes: int,
        features: list[Feats],
        hsize: int = 256,
        nlayer: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__()

        if (
            not set(features).issubset(
                {Feats.BURST_LENS, Feats.BURST_DURS, Feats.BURST_RELDURS}
            )
        ) and (
            not set(features).issubset(
                {Feats.DIRS, Feats.DIR_PROBS, Feats.IATS, Feats.TIMES}
            )
        ):
            raise ValueError(f"Invalid set of feats. {'-'.join(features)}")

        self.features = features
        nfeat = len(features)
        self.rnn = nn.LSTM(nfeat, hsize, nlayer, batch_first=True, dropout=dropout)

        self.final_lin = nn.Sequential(nn.Dropout(dropout), nn.Linear(hsize, n_classes))

    def pack_and_forward(
        self,
        x: dict[Feats, torch.Tensor],
        h: tuple[torch.Tensor, ...],
        seq_lens: torch.Tensor,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor]]:
        fs = []
        for f in self.features:
            if x[f].ndim != 2:
                raise ValueError("Inputs should be (B, L) tensors.")

            fs.append(x[f].unsqueeze(-1))

        mask = seq_lens != 0

        if mask.sum() == 0:
            return None, h

        # M = mask.sum()

        # (M, L, nfeat)
        inputs = torch.cat(fs, dim=-1)[mask, ...]

        packed_inputs = pack_padded_sequence(
            inputs, seq_lens[mask], batch_first=True, enforce_sorted=False
        )

        # Only select the needed h states:
        h_ = _hidden_w_mask(h, mask)

        # (M, L, H), hidden
        _, h_ = self.rnn(packed_inputs, h_)

        # h_[0].shape = (nhidden, B, hidden_size)
        # In the last index we have the last rnn output.
        # (M, H)
        logits = self.final_lin(h_[0][-1])

        # Update the hidden state:
        h = _hidden_w_mask(h, mask, h_)

        return logits, h

    def forward(
        self,
        x: dict[Feats, torch.Tensor | PackedSequence],
        h: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # (B, L) x nfeat
        fs = []
        for f in self.features:
            if (not isinstance(x[f], PackedSequence)) and (x[f].ndim != 2):
                raise ValueError("Inputs should be (B, L) tensors.")
            fs.append(x[f].unsqueeze(-1))

        # (B, L, nfeat)
        inputs = torch.cat(fs, dim=-1)

        # (B, L, H)
        output, h = self.rnn(inputs, h)

        # (B, L, n_classes)
        logits = self.final_lin(output)

        return logits, h

    def predict(
        self, x: dict[Feats, torch.Tensor], h: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if x[self.features[0]].ndim != 2:
            raise ValueError("Batched inputs expected!")

        # (N, nt, n_classes)
        logits, _ = self(x)

        # (N, nt, n_classes)
        probs = nn.functional.softmax(logits, dim=2)

        # (N, nt)
        max_p = probs.max(dim=2)[0]
        k = 21

        # (N, nt)
        max_p = nn.functional.conv1d(
            max_p.unsqueeze(1),
            weight=torch.Tensor([1.0 / k] * k)
            .to(max_p.device)
            .unsqueeze(0)
            .unsqueeze(0),
            padding="same",
        ).squeeze(1)

        # (N, )
        col_idxs = max_p.argmax(dim=1)

        bs = len(logits)
        row_idxs = torch.arange(bs).long()
        # col_idxs = torch.clip(
        #     (x[Feats.BURST_LENS] != 0).sum(dim=1).long(), 0, seq_len - 1
        # )

        # (N, n_classes)
        logits = logits[row_idxs, col_idxs, :]

        # (N, )
        preds = probs[row_idxs, col_idxs, :].argmax(-1)

        return logits, preds


class ANTINCLF1(nn.Module):
    name: str = "anticlf1"

    def __init__(
        self,
        features: list[Feats],
        hsize: int = 256,
        nlayers: int = 3,
        dropout: float = 0.2,
        zero_init: bool = False,
    ):
        super().__init__()

        if not set(features).issubset(
            {Feats.BURST_LENS, Feats.BURST_DURS, Feats.BURST_RELDURS}
        ):
            raise ValueError("Invalid set of feats.")

        self.features = features
        self.num_layers = nlayers
        self.hidden_size = hsize
        self.zero_init = zero_init
        nfeat = len(features)
        self.rnn = nn.LSTM(nfeat, hsize, nlayers, batch_first=True, dropout=dropout)

        self.final_lin = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(hsize, len(features))
        )

    def _get_init_h(self, x: dict[Feats, torch.Tensor]) -> torch.Tensor | None:
        if not self.zero_init:
            return None

        bs = x[self.features[0]].shape[0]
        device = x[self.features[0]].device
        return (
            torch.zeros(self.num_layers, bs, self.hidden_size, device=device),
            torch.zeros(self.num_layers, bs, self.hidden_size, device=device),
        )

    def forward(
        self, x: dict[Feats, torch.Tensor], h: torch.Tensor | None = None
    ) -> torch.Tensor:
        if h is None:
            h = self._get_init_h(x)

        # (N, L) x nfeat
        fs = []
        for f in self.features:
            fs.append(x[f].unsqueeze(-1))

        # (N, L, nfeat)
        inputs = torch.cat(fs, dim=-1)

        # (N, L, H)
        output, h = self.rnn(inputs, h)

        # (N, L, n_classes)
        addons = self.final_lin(output)

        # vals \in ]0, inf[
        scales = torch.nn.functional.elu(addons) + 1

        # The model tells how to modify the _next_ burst, not the current one.
        xobs = {k: v.clone() for k, v in x.items()}
        for i, f in enumerate(self.features):
            xobs[f][:, 1:] = x[f][:, :-1] * scales[:, :-1, i] + x[f][:, 1:]

        return xobs


class AGENT1(nn.Module):
    name: str = "agent"
    ACTIONS = (
        Actions.WAIT,
        Actions.SEND_BUFFER,
        Actions.SEND_PADDING_UP,
        Actions.SEND_PADDING_DOWN,
    )

    def __init__(
        self,
        hsize: int = 256,
        nlayers: int = 3,
        dropout: float = 0.2,
        zero_init: bool = False,
        send_counts: list[int] | None = None,
    ):
        super().__init__()

        self.nactions = len(self.ACTIONS)
        self.features = [Feats.UP_BUFFER, Feats.DOWN_BUFFER, Feats.TIMES]
        self.num_layers = nlayers
        self.hidden_size = hsize
        self.zero_init = zero_init
        self.counts = send_counts or [1, 2, 4, 8, 16, 32, 64]
        self.ncounts = len(self.counts)
        nfeat = len(self.features)

        # Mapping from idx to (action, count)
        self._action_map = [(Actions.WAIT, 1)]
        self._action_map += [(Actions.SEND_BUFFER, count) for count in self.counts]
        self._action_map += [(Actions.SEND_PADDING_UP, count) for count in self.counts]
        self._action_map += [
            (Actions.SEND_PADDING_DOWN, count) for count in self.counts
        ]

        self.rnn = nn.LSTM(nfeat, hsize, nlayers, batch_first=True, dropout=dropout)

        self.actor = nn.ModuleDict(
            {
                "action_selection": nn.Sequential(
                    nn.Dropout(dropout), nn.Linear(hsize, self.nactions)
                ),
                "count_selection": nn.Sequential(
                    nn.Dropout(dropout), nn.Linear(hsize, self.ncounts)
                ),
            }
        )

        self.critic = nn.Sequential(nn.Dropout(dropout), nn.Linear(hsize, 1))

    def _get_init_h(
        self, x: dict[Feats, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        if not self.zero_init:
            return None

        bs = x[self.features[0]].shape[0]
        device = x[self.features[0]].device
        return (
            torch.zeros(self.num_layers, bs, self.hidden_size, device=device),
            torch.zeros(self.num_layers, bs, self.hidden_size, device=device),
        )

    def forward(
        self, x: dict[Feats, torch.Tensor], h: torch.Tensor | None = None
    ) -> tuple[dict[Feats, torch.Tensor], torch.Tensor]:
        if h is None:
            h = self._get_init_h(x)

        # (N, L) x nfeat
        fs = []
        for f in self.features:
            fs.append(x[f].unsqueeze(-1))

        # (N, L, nfeat)
        inputs = torch.cat(fs, dim=-1)

        # (N, L, H)
        output, h = self.rnn(inputs, h)

        # (N, L, nactions)
        action_type = self.actor["action_selection"](output)

        # (N, L, ncounts)
        action_count = self.actor["count_selection"](output)

        # Combine action type and count into final action logits
        action_logits = torch.zeros(
            (
                action_type.shape[0],
                action_type.shape[1],
                (self.nactions - 1) * self.ncounts + 1,
            ),
            device=action_type.device,
        )

        action_logits[..., 0] = action_type[..., 0]  # WAIT action
        n = 1
        for i in range(1, self.nactions):
            action_logits[..., n : n + self.ncounts] = (
                action_type[..., i].unsqueeze(2) + action_count
            )
            n += self.ncounts

        # (N, L, 1)
        state_values = self.critic(output).squeeze(-1)

        return {Feats.ACTION_LOGITS: action_logits, Feats.STATE_VALUE: state_values}, h

    @property
    def action_map(self) -> list[tuple[Actions, int]]:
        return self._action_map

    def act(
        self, x: dict[Feats, torch.Tensor], h: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        action_outputs, h = self(x, h)

        # (B, L, nactions)
        action_logits = action_outputs[Feats.ACTION_LOGITS]
        action_probs = nn.functional.softmax(action_logits, dim=-1)

        # (B, L)
        actions = torch.distributions.Categorical(action_probs).sample()

        # (B, L, nactions)
        log_probs = torch.log(action_probs)

        # (B, L)
        log_probs = log_probs.gather(-1, actions.unsqueeze(-1)).squeeze(-1)

        # (B, L)
        values = action_outputs[Feats.STATE_VALUE]

        if actions.shape[1] != 1:
            raise ValueError("Expected action shape (B, 1)")

        return (
            actions.squeeze(1),
            log_probs.squeeze(1),
            values.squeeze(1),
            h,
        )
