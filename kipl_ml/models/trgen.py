import torch
from torch import nn
from torch.distributions import Categorical, Normal, Poisson
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
        h_detach_period: int | None = None,
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

        self.features = [Feats.UP_COUNT, Feats.DOWN_COUNT, Feats.Dt]
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

    def _forward_w_detach(
        self,
        x: dict[Feats, torch.Tensor],
        h: torch.Tensor | None,
        h_detach_period: int,
    ) -> tuple[dict[Feats | Actions, torch.Tensor], torch.Tensor]:
        T = x[self.features[0]].shape[1]

        actions = []
        for i in range(T // h_detach_period + 1):
            if i * h_detach_period == T:
                break

            # (N, h_detach_period)
            x_chunk = {
                k: v[:, i * h_detach_period : (i + 1) * h_detach_period]
                for k, v in x.items()
            }

            actions_, h = self.forward(x_chunk, h)

            if isinstance(h, tuple):
                h = tuple(v.detach() for v in h)
            else:
                h = h.detach()

            actions.append(actions_)

        # Concatenate actions:
        actions_concat = {
            k: torch.cat([a[k] for a in actions], dim=1) for k in actions[0].keys()
        }

        return actions_concat, h

    def forward(
        self,
        x: dict[Feats, torch.Tensor],
        h: torch.Tensor | None = None,
        h_detach_period: int | None = None,
    ) -> tuple[dict[Feats | Actions, torch.Tensor], torch.Tensor]:
        if h_detach_period is not None:
            return self._forward_w_detach(x, h, h_detach_period)

        # Typically we have time in dim=1, here we always(?)
        # (N, L) x nfeat
        fs = []
        for f in self.features:
            x_ = torch.log10(1 + x[f])
            fs.append(x_.unsqueeze(-1))

        if h is not None:
            h_norm = h[0].norm(2, dim=-1).max().item()
            c_norm = h[1].norm(2, dim=-1).max().item()
            if h_norm > 100 or c_norm > 100:
                print("Huge hidden/cell:", h_norm, c_norm)

        # (N, L, nfeat)
        inputs = torch.cat(fs, dim=-1)

        # (N, L, nfeat * feat_scale)
        inputs = self.scaler(inputs)

        # (N, L, H) (N, n_hidden, H)
        output, h = self.rnn(inputs, h)

        # (N, L, 4) (0=WAIT, 1=SEND_UP, 2=SEND_DOWN, 3=SEND_BOTH)
        action_selector = self.actor["action_selection"](output)

        # (N, L)
        send_count_u = self.actor["send_count_u"](output).squeeze(-1)
        send_count_d = self.actor["send_count_d"](output).squeeze(-1)
        send_time_u = self.actor["send_time_u"](output).squeeze(-1)
        send_time_d = self.actor["send_time_d"](output).squeeze(-1)

        # (N, L)
        state_values = self.critic(output).squeeze(-1)

        return {
            Actions.SELECTOR: action_selector,
            Actions.SEND_COUNT_UP: send_count_u,
            Actions.SEND_TIME_UP: send_time_u,
            Actions.SEND_COUNT_DOWN: send_count_d,
            Actions.SEND_TIME_DOWN: send_time_d,
            Feats.STATE_VALUE: state_values,
        }, h

    def act(
        self,
        x: dict[Feats, torch.Tensor],
        h: torch.Tensor | None = None,
        h_detach_period: int | None = None,
    ) -> tuple[
        torch.Tensor,
        dict[Actions, torch.Tensor],
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        action_outputs, h = self(x, h, h_detach_period)

        # Select action:
        sel_dist = Categorical(logits=action_outputs[Actions.SELECTOR])

        # (B, L)
        selections = sel_dist.sample()
        sel_log_probs = sel_dist.log_prob(selections)
        sel_entropy = sel_dist.entropy()

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

        # (B, L)
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
        # (B, L)
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
        # (B, L)
        mask = selections == 0

        # (B, L)
        log_probs[mask] = sel_log_probs[mask]
        actions[Actions.WAIT][mask] = 1
        actions[Actions.SEND_COUNT_UP][mask] = 0
        actions[Actions.SEND_COUNT_DOWN][mask] = 0
        actions[Actions.SEND_TIME_UP][mask] = 0
        actions[Actions.SEND_TIME_DOWN][mask] = 0

        # SEND UP:
        # (B, L)
        mask = selections == 1

        # (B, L)
        log_probs[mask] = (
            sel_log_probs[mask] + send_count_u_logp[mask] + send_time_u_logp[mask]
        )
        actions[Actions.SEND_COUNT_DOWN][mask] = 0
        actions[Actions.SEND_TIME_DOWN][mask] = 0

        # SEND DOWN:
        # (B, L)
        mask = selections == 2

        # (B, L)
        log_probs[mask] = (
            sel_log_probs[mask] + send_count_d_logp[mask] + send_time_d_logp[mask]
        )
        actions[Actions.SEND_COUNT_UP][mask] = 0
        actions[Actions.SEND_TIME_UP][mask] = 0

        # SEND BOTH:
        # (B, L)
        mask = selections == 3

        # (B, L)
        log_probs[mask] = (
            sel_log_probs[mask]
            + send_count_u_logp[mask]
            + send_time_u_logp[mask]
            + send_count_d_logp[mask]
            + send_time_d_logp[mask]
        )

        # (B, L)
        values = action_outputs[Feats.STATE_VALUE]

        times = x[Feats.TIMES]

        return times, actions, log_probs, values, entropy, h
