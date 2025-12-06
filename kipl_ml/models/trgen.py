import torch
from torch import nn
from torch.distributions import Categorical, Normal, Poisson
from torch.nn.utils.rnn import PackedSequence, pack_padded_sequence

from kipl_ml.data.utils import DOWNLOAD, UPLOAD
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
        h: tuple[torch.Tensor, ...] | None,
        seq_lens: torch.Tensor,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor]]:
        if h is None:
            if (seq_lens == 0).any():
                raise ValueError(
                    "Initial hidden state must be provided when seq_lens has zeros."
                )

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
        if h is not None:
            h_ = _hidden_w_mask(h, mask)
        else:
            h_ = None

        # (M, L, H), hidden
        _, h_ = self.rnn(packed_inputs, h_)

        # h_[0].shape = (nhidden, B, hidden_size)
        # In the last index we have the last rnn output.
        # (M, H)
        logits = self.final_lin(h_[0][-1])

        # Update the hidden state:
        if h is not None:
            h = _hidden_w_mask(h, mask, h_)
        else:
            h = h_

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
                raise ValueError(f"Inputs should be (B, L) tensors, got {x[f].shape}.")
            fs.append(x[f].unsqueeze(-1))

        # (B, L, nfeat)
        inputs = torch.cat(fs, dim=-1)

        # (B, L, H)
        output, h = self.rnn(inputs, h)

        # (B, L, n_classes)
        logits = self.final_lin(output)

        return logits, h

    @torch.no_grad()
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

    def __init__(
        self,
        hsize: int = 256,
        nlayers: int = 3,
        dropout: float = 0.2,
        zero_init: bool = False,
    ):
        super().__init__()

        self.features = [Feats.UP_COUNT, Feats.DOWN_COUNT]
        self.num_layers = nlayers
        self.hidden_size = hsize
        self.zero_init = zero_init
        nfeat = len(self.features)

        self.scaler = nn.Sequential(nn.Linear(nfeat, nfeat, bias=False), nn.Tanh())
        self.rnn = nn.LSTM(
            nfeat, hsize, nlayers, batch_first=True, dropout=dropout, bias=False
        )

        self.actor = nn.ModuleDict(
            {
                "action_selection": nn.Sequential(
                    nn.Dropout(dropout), nn.Linear(hsize, 4)
                ),
                "send_count_u": nn.Sequential(nn.Dropout(dropout), nn.Linear(hsize, 1)),
                "send_count_d": nn.Sequential(nn.Dropout(dropout), nn.Linear(hsize, 1)),
                "send_time_u": nn.Sequential(nn.Dropout(dropout), nn.Linear(hsize, 1)),
                "send_time_d": nn.Sequential(nn.Dropout(dropout), nn.Linear(hsize, 1)),
            }
        )

        self.critic = nn.Sequential(nn.Dropout(dropout), nn.Linear(hsize, 1))

    def _get_init_h(
        self, x: dict[Feats, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        if not self.zero_init:
            return None

        bs = x[Feats.DIRS].shape[0]
        device = x[Feats.DIRS].device
        return (
            torch.zeros(self.num_layers, bs, self.hidden_size, device=device),
            torch.zeros(self.num_layers, bs, self.hidden_size, device=device),
        )

    def forward(
        self, x: dict[Feats, torch.Tensor], h: torch.Tensor | None = None
    ) -> tuple[dict[Feats | Actions, torch.Tensor], torch.Tensor]:
        if h is None:
            h = self._get_init_h(x)

        # Feature generation:
        x[Feats.UP_COUNT] = (x[Feats.DIRS] == UPLOAD).sum(dim=1, keepdim=True).float()
        x[Feats.DOWN_COUNT] = (
            (x[Feats.DIRS] == DOWNLOAD).sum(dim=1, keepdim=True).float()
        )

        # Typically we have time in dim=1, here we always(?)
        # operate on only single time step... Regardless we only
        # squeeze the time dim out in the end.
        # (N, 1) x nfeat
        fs = []
        for f in self.features:
            x_ = torch.log10(1 + x[f])
            fs.append(x_.unsqueeze(-1))

        if h is not None:
            h_norm = h[0].norm(2, dim=-1).max().item()
            c_norm = h[1].norm(2, dim=-1).max().item()
            if h_norm > 100 or c_norm > 100:
                print("Huge hidden/cell:", h_norm, c_norm)

        # (N, 1, nfeat)
        inputs = torch.cat(fs, dim=-1)

        # (N, 1, nfeat * feat_scale)
        inputs = self.scaler(inputs)

        # (N, 1, H) (N, n_hidden, H)
        output, h = self.rnn(inputs, h)

        # (N, H)
        output = output.squeeze(1)

        # (N, 4) (0=WAIT, 1=SEND_UP, 2=SEND_DOWN, 3=SEND_BOTH)
        action_selector = self.actor["action_selection"](output)

        # (N, 1)
        send_count_u = self.actor["send_count_u"](output)
        send_count_d = self.actor["send_count_d"](output)
        send_time_u = self.actor["send_time_u"](output)
        send_time_d = self.actor["send_time_d"](output)

        # (N, 1)
        state_values = self.critic(output)

        return {
            Actions.SELECTOR: action_selector,
            Actions.SEND_COUNT_UP: send_count_u,
            Actions.SEND_TIME_UP: send_time_u,
            Actions.SEND_COUNT_DOWN: send_count_d,
            Actions.SEND_TIME_DOWN: send_time_d,
            Feats.STATE_VALUE: state_values,
        }, h

    def act(
        self, x: dict[Feats, torch.Tensor], h: torch.Tensor | None = None
    ) -> tuple[
        dict[Actions, torch.Tensor],
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        action_outputs, h = self(x, h)

        # Select action:
        sel_dist = Categorical(logits=action_outputs[Actions.SELECTOR])

        # (B, )
        selections = sel_dist.sample()
        sel_log_probs = sel_dist.log_prob(selections)
        sel_entropy = sel_dist.entropy()

        # (B, 1)
        selections = selections.unsqueeze(1)
        sel_log_probs = sel_log_probs.unsqueeze(1)
        sel_entropy = sel_entropy.unsqueeze(1)

        # Send u/d, note! These are conditional on the selection.
        # They will be ignored if the selection is not SEND_UP/DOWN/BOTH.
        suc = Poisson(1 + nn.functional.softplus(action_outputs[Actions.SEND_COUNT_UP]))
        sut = Normal(
            0.01 + nn.functional.softplus(action_outputs[Actions.SEND_TIME_UP]) / 10,
            1e-3,
        )
        sdc = Poisson(
            1 + nn.functional.softplus(action_outputs[Actions.SEND_COUNT_DOWN])
        )
        sdt = Normal(
            0.01 + nn.functional.softplus(action_outputs[Actions.SEND_TIME_DOWN]) / 10,
            1e-3,
        )

        # (B, 1)
        send_count_u = suc.sample()
        send_count_u_logp = suc.log_prob(send_count_u)

        send_count_d = sdc.sample()
        send_count_d_logp = sdc.log_prob(send_count_d)

        send_time_u = sut.sample()
        send_time_u_logp = sut.log_prob(send_time_u)

        send_time_d = sdt.sample()
        send_time_d_logp = sdt.log_prob(send_time_d)

        # Conditional entropy H[A|S]:
        # =============================
        # (B, 4)
        # sel_probs = sel_dist.probs

        # (B, 1)
        # up_p = sel_probs[..., 1] + sel_probs[..., 3]
        # down_p = sel_probs[..., 2] + sel_probs[..., 3]

        # (B, 1)
        # Poisson does not have entropy implemented - ignoring these for now.
        # The correct entropy is computed as:
        # up_p * (suc_entropy + stu_entropy) + down_p * (sdc_entropy + sdt_entropy)
        # suc_entropy = suc.entropy()
        # sdc_entropy = sdc.entropy()
        # stu_entropy = sut.entropy()
        # sdt_entropy = sdt.entropy()
        # This cond entropy becomes redundant: if tha var of normals is fixed -> H[N] = const,
        # so we can ignore it in the optimization.
        # For Poisson, the entropy must be ~ to the "mean" -> however, that would only encourage
        # Larger send values -> ignore
        # cond_entropy = up_p * stu_entropy + down_p * sdt_entropy
        cond_entropy = 0.0

        # Entropy (B, 1) H[A] = H[S] + H[A|S]
        # Policy, \Pi[A] = \Pi[S] * \Pi[A|S]
        entropy = sel_entropy + cond_entropy

        # Use action selector to choose what to do:
        # (B, 1)
        log_probs = torch.zeros_like(sel_log_probs)
        actions = {
            Actions.SELECTOR: selections.detach().clone(),
            Actions.WAIT: torch.zeros_like(selections),
            Actions.SEND_COUNT_DOWN: send_count_d.detach().clone(),
            Actions.SEND_COUNT_UP: send_count_u.detach().clone(),
            Actions.SEND_TIME_DOWN: send_time_d.detach().clone(),
            Actions.SEND_TIME_UP: send_time_u.detach().clone(),
        }

        # WAIT:
        mask = selections == 0

        # (B, 1)
        log_probs[mask] = sel_log_probs[mask]
        actions[Actions.WAIT][mask] = 1
        actions[Actions.SEND_COUNT_UP][mask] = 0
        actions[Actions.SEND_COUNT_DOWN][mask] = 0
        actions[Actions.SEND_TIME_UP][mask] = 0
        actions[Actions.SEND_TIME_DOWN][mask] = 0

        # SEND UP:
        mask = selections == 1

        # (B, 1)
        log_probs[mask] = (
            sel_log_probs[mask] + send_count_u_logp[mask] + send_time_u_logp[mask]
        )
        actions[Actions.SEND_COUNT_DOWN][mask] = 0
        actions[Actions.SEND_TIME_DOWN][mask] = 0

        # SEND DOWN:
        mask = selections == 2

        # (B, 1)
        log_probs[mask] = (
            sel_log_probs[mask] + send_count_d_logp[mask] + send_time_d_logp[mask]
        )
        actions[Actions.SEND_COUNT_UP][mask] = 0
        actions[Actions.SEND_TIME_UP][mask] = 0

        # SEND BOTH:
        mask = selections == 3

        # (B, 1)
        log_probs[mask] = (
            sel_log_probs[mask]
            + send_count_u_logp[mask]
            + send_time_u_logp[mask]
            + send_count_d_logp[mask]
            + send_time_d_logp[mask]
        )

        # (B, 1)
        values = action_outputs[Feats.STATE_VALUE]

        # Squeeze the ch dim.
        # (B, )
        for _, v in actions.items():
            v.squeeze_(1)

        return (
            actions,
            log_probs.squeeze(1),
            values.squeeze(1),
            entropy.squeeze(1),
            h,
        )
