from typing import Any

import torch
from torch import nn
from torch.distributions import Categorical
from torch.nn.utils.rnn import PackedSequence, pack_padded_sequence, pad_packed_sequence

from kipl_ml.logging.logger import get_logger
from kipl_ml.rl.enums import Actions
from kipl_ml.trace.features import Feats

logger = get_logger(__name__)


def _hidden_w_mask(
    h: tuple[torch.Tensor, ...] | None,
    mask: torch.Tensor,
    hmasked: tuple[torch.Tensor, ...] | None = None,
) -> tuple[torch.Tensor, ...] | None:
    if h is None:
        if not mask.all():
            raise ValueError("Hidden is none, but only a subet is selected")
        if hmasked is not None:
            return hmasked
        return None

    if hmasked is None:
        return tuple(h_[:, mask] for h_ in h)

    for i, h_ in enumerate(h):
        h_[:, mask] = hmasked[i]

    return h


def dir_seq_len_fun(x: dict[Feats, torch.Tensor]) -> torch.Tensor:
    return (x[Feats.DIRS] != 0).sum(dim=1)


def tam_seq_len_fun(x: dict[Feats, torch.Tensor]) -> torch.Tensor:
    bs, nt = x[Feats.TAM_UP_COUNTS].shape
    mask = x[Feats.TAM_UP_COUNTS] + x[Feats.TAM_DOWN_COUNTS] != 0

    seq_lens = (
        torch.where(
            mask, torch.arange(nt, device=mask.device)[None, :].repeat(bs, 1), 0
        )
        .max(dim=1)
        .values
        + 1
    )

    return seq_lens


def _feature_map(
    x: dict[Feats, torch.Tensor], f: Feats | list[Feats]
) -> list[torch.Tensor] | torch.Tensor:
    if isinstance(f, list):
        return [_feature_map(x, fi) for fi in f]

    if f == Feats.SILENCE_FLAG:
        return x[f].unsqueeze(-1)  # keep 0/1
    if f in (Feats.TIMES, Feats.TAM_TIMES):
        return (x[f] / (x[f] + 10)).unsqueeze(-1)

    return torch.log1p(x[f]).unsqueeze(-1)


class RNNCLF1(nn.Module):
    name: str = "rnnclf1"

    def __init__(
        self,
        n_classes: int,
        features: list[Feats],
        trim_beginning: int,
        hsize: int = 256,
        nlayer: int = 3,
        dropout: float = 0.2,
        predict_ks: int = 21,
        tam_dict: dict[str, int] | None = None,
    ):
        super().__init__()

        self.trim_beginning = trim_beginning

        if set(features).issubset(
            {Feats.BURST_LENS, Feats.BURST_DURS, Feats.BURST_RELDURS}
        ):
            raise NotImplementedError("Deprecated")

        elif set(features).issubset(
            {
                Feats.DIRS,
                Feats.DIR_PROBS,
                Feats.IATS,
                Feats.TIMES,
                Feats.LOG1P_IATS,
            }
        ):
            self.seq_len_fun = dir_seq_len_fun
            self.feat_mode = "dir"
        elif set(features).issubset(
            {
                Feats.TAM_UP_COUNTS,
                Feats.TAM_DOWN_COUNTS,
                Feats.TAM_TIMES,
            }
        ):
            self.seq_len_fun = tam_seq_len_fun
            self.tam_dict = tam_dict or {}
            self.feat_mode = "tam"
        else:
            raise ValueError(f"Invalid set of feats. {'-'.join(features)}")

        self.features = features
        nfeat = len(features)
        self.rnn = nn.LSTM(nfeat, hsize, nlayer, batch_first=True, dropout=dropout)

        self.final_lin = nn.Sequential(nn.Dropout(dropout), nn.Linear(hsize, n_classes))
        self.predict_ks = predict_ks

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

            fs.append(_feature_map(x, f))

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
        output_packed, h_ = self.rnn(packed_inputs, h_)

        output, lens = pad_packed_sequence(output_packed, batch_first=True)

        # h_[0].shape = (nhidden, B, hidden_size)

        # (M, L, n_classes)
        logits = self.final_lin(output)

        # Update the hidden state:
        h = _hidden_w_mask(h, mask, h_)

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
            fs.append(_feature_map(x, f))

        # (B, L, nfeat)
        inputs = torch.cat(fs, dim=-1)

        # (B, L, H)
        output, h = self.rnn(inputs, h)

        # (B, L, n_classes)
        logits = self.final_lin(output)

        return logits, h

    @torch.no_grad()
    def predict(
        self,
        x: dict[Feats, torch.Tensor],
        h: torch.Tensor | None = None,
        seq_lens: torch.Tensor | None = None,
        ks: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if x[self.features[0]].ndim != 2:
            raise ValueError("Batched inputs expected!")

        ks = ks or self.predict_ks

        # (N, nt, n_classes)
        logits, _ = self(x)
        bs, nt, n_classes = logits.shape

        # (N, nt, n_classes)
        probs = nn.functional.softmax(logits, dim=2)

        # (N, nt)
        max_p = probs.max(dim=2)[0]

        # (N, nt)
        max_p = nn.functional.conv1d(
            max_p.unsqueeze(1),
            weight=torch.Tensor([1.0 / ks] * ks)
            .to(max_p.device)
            .unsqueeze(0)
            .unsqueeze(0),
            padding="same",
        ).squeeze(1)

        if seq_lens is None:
            seq_lens = self.seq_len_fun(x)

        # (N, nt)
        seq_lens_mask = (
            torch.arange(nt, device=max_p.device).unsqueeze(0).repeat(bs, 1)
            > seq_lens.unsqueeze(1) - 1
        )
        # We are only interested in the domain of valid packets...
        max_p = max_p.masked_fill(seq_lens_mask, 0.0)

        # (N, )
        col_idxs = max_p.argmax(dim=1)
        row_idxs = torch.arange(bs).long()

        # (N, n_classes)
        logits = logits[row_idxs, col_idxs, :]

        # (N, )
        preds = probs[row_idxs, col_idxs, :].argmax(-1)

        return logits, preds


def _forward_w_detach(
    x: dict[Feats, torch.Tensor],
    h: torch.Tensor | None,
    h_detach_period: int,
    seq_lens: torch.Tensor,
    forward: callable,
) -> tuple[dict[Feats | Actions, torch.Tensor], torch.Tensor]:
    features = list(x.keys())
    bs, T = x[features[0]].shape

    outputs = []
    i = 0
    while True:
        if i * h_detach_period >= T:
            break

        active_seqs = seq_lens > i * h_detach_period

        if active_seqs.sum() == 0:
            break

        # (N, h_detach_period)
        x_chunk = {
            k: torch.where(
                v[
                    active_seqs, i * h_detach_period : (i + 1) * h_detach_period
                ].isfinite(),
                v[active_seqs, i * h_detach_period : (i + 1) * h_detach_period],
                0,
            )
            for k, v in x.items()
        }

        if h is not None:
            h_active = _hidden_w_mask(h, active_seqs)
        else:
            h_active = None
            if active_seqs.sum() != bs:
                raise ValueError(
                    "Initial hidden state must be provided when some sequences are inactive."
                )

        outputs_, h_active = forward(x_chunk, h_active)

        # Update h, i.e., place the updated h_active back to h:
        if h is None:
            h = h_active
        else:
            h = _hidden_w_mask(h, active_seqs, h_active)

        if isinstance(h, tuple):
            h = tuple(v.detach() for v in h)
        else:
            h = h.detach()

        # Place actions back to full size:
        for k, v in outputs_.items():
            if v.ndim == 2:
                t_ = torch.zeros((bs, v.shape[1]), device=v.device)
            elif v.ndim == 3:
                t_ = torch.zeros((bs, v.shape[1], v.shape[2]), device=v.device)
            else:
                raise ValueError("Invalid action tensor shape.")
            t_[active_seqs, ...] = v
            outputs_[k] = t_

        outputs.append(outputs_)
        i += 1

    # Concatenate actions:
    outputs_concat = {
        k: torch.cat([a[k] for a in outputs], dim=1) for k in outputs[0].keys()
    }

    for k, v in outputs_concat.items():
        if bs != v.shape[0]:
            raise ValueError("Batch size mismatch after concat.")
        if v.shape[1] != T:
            raise ValueError(f"Times len mismatch, {v.shape[1]} vs. {T}")

    return outputs_concat, h


class AGENT1(nn.Module):
    name: str = "agent"

    def __init__(
        self,
        time_step: float = 0.05,
        max_silence_s: float = 0.5,
        hsize: int = 256,
        nlayers: int = 3,
        send_count_bins: list[int] | None = None,
        send_time_bins: list[float] | None = None,
        dropout: float = 0.0,
        prob_eps: dict[Actions, float] | float | None = None,
        send_mode: str = "spread",
        prefer_wait_bias: float = 0.0,
        train_env: dict[str, Any] | None = None,
    ):
        super().__init__()

        self.train_env = train_env or {}
        self.ACTIONS: list[Actions] = [
            Actions.SELECTOR,
            Actions.SEND_COUNT_UP,
            Actions.SEND_COUNT_DOWN,
        ]
        send_count_bins = send_count_bins or [5, 20, 50, 100, 200]

        self.send_mode = send_mode
        if send_mode == "spread":
            self.ACTIONS += [Actions.SEND_TIME_UP, Actions.SEND_TIME_DOWN]
            send_time_bins = send_time_bins or [
                0.02,
                0.04,
                0.08,
                0.16,
            ]

        elif send_mode == "fixed":
            self.ACTIONS += [Actions.SEND_UP_AFTER_TIME, Actions.SEND_DOWN_AFTER_TIME]
            send_time_bins = send_time_bins or [
                0.00,
                0.02,
                0.06,
                0.10,
                0.14,
                0.18,
            ]

        else:
            raise ValueError(
                f"Invalid send_mode {send_mode}, expected 'spread' or 'fix'."
            )

        if abs(max_silence_s % time_step) > 1e-12:
            raise ValueError(
                "max_silence_s should be a multiple of time_step."
                + f"Got max_silence_s={max_silence_s}, time_step={time_step}."
            )

        n_send_counts = len(send_count_bins)
        n_decay_times = len(send_time_bins)

        self.register_buffer(
            "send_count_bins", torch.tensor(send_count_bins, dtype=torch.long)
        )
        self.register_buffer(
            "send_time_bins", torch.tensor(send_time_bins, dtype=torch.float)
        )

        # Time step between feature extractions.
        self.time_step = time_step
        # Maximum silence the model tolerates before acting.
        self.max_silence_s = max_silence_s

        # Exploration prob eps for each action:
        control_actions = [
            Actions.SELECTOR,
            Actions.SEND_COUNT_UP,
            Actions.SEND_COUNT_DOWN,
            Actions.SEND_TIME_UP,
            Actions.SEND_TIME_DOWN,
        ]
        if prob_eps is not None:
            if isinstance(prob_eps, float):
                self.prob_eps = {a: prob_eps for a in control_actions}
            else:
                if not set(prob_eps.keys()).issuperset(set(control_actions)):
                    raise ValueError("prob_eps keys must cover all actions.")
                for a in control_actions:
                    eps = float(prob_eps[a])
                    if eps < 0 or eps > 1:
                        raise ValueError(f"prob_eps[{a}] must be in [0, 1], got {eps}")
                self.prob_eps = {a: float(prob_eps[a]) for a in control_actions}
        else:
            self.prob_eps = {a: 0.0 for a in control_actions}

        self.features = [
            Feats.UP_COUNT,
            Feats.DOWN_COUNT,
            Feats.Dt,
            Feats.TIMES,
            Feats.SILENCE_FLAG,
        ]
        self.num_layers = nlayers
        self.hidden_size = hsize
        nfeat = len(self.features)

        self.scaler = nn.Sequential(nn.Linear(nfeat, nfeat, bias=False), nn.Tanh())
        self.rnn = nn.LSTM(
            nfeat, hsize, nlayers, batch_first=True, dropout=dropout, bias=True
        )

        self.out_norm = nn.LayerNorm(hsize)

        self.actor = nn.ModuleDict(
            {
                "action_selection": nn.Sequential(
                    nn.Dropout(dropout), nn.Linear(hsize, 4)
                ),
                "send_count_u": nn.Sequential(
                    nn.Dropout(dropout), nn.Linear(hsize, n_send_counts)
                ),
                "send_count_d": nn.Sequential(
                    nn.Dropout(dropout), nn.Linear(hsize, n_send_counts)
                ),
                "send_time_u": nn.Sequential(
                    nn.Dropout(dropout), nn.Linear(hsize, n_decay_times)
                ),
                "send_time_d": nn.Sequential(
                    nn.Dropout(dropout), nn.Linear(hsize, n_decay_times)
                ),
            }
        )

        self.critic = nn.Sequential(
            nn.Linear(hsize, hsize), nn.ReLU(), nn.Linear(hsize, 1)
        )
        self.cond_beta = 1.0

        if prefer_wait_bias != 0.0:
            self._init_action_selection_prefer_wait(prefer_wait_bias=prefer_wait_bias)

    def _init_action_selection_prefer_wait(self, prefer_wait_bias: float) -> None:
        """Initialize selector logits to heavily prefer WAIT (selector index 0)."""
        if prefer_wait_bias < 0:
            raise ValueError(f"prefer_wait_bias must be >= 0, got {prefer_wait_bias}")

        head = self.actor["action_selection"]
        lin = head[-1]
        if not isinstance(lin, nn.Linear) or lin.out_features != 4:
            raise TypeError("action_selection head must end with Linear(..., 4)")

        with torch.no_grad():
            lin.bias.zero_()
            lin.weight *= 0.1
            lin.bias[0] = float(prefer_wait_bias)

    @property
    def cond_beta(self) -> float:
        return self._cond_beta

    @cond_beta.setter
    def cond_beta(self, cond_beta: float):
        # This param is to steer the importance of the cond part
        # of the prop distr (for actions). High, beta, learn cond part.

        self._cond_beta = cond_beta

    def forward(
        self,
        x: dict[Feats, torch.Tensor],
        h: torch.Tensor | None = None,
        h_detach_period: int | None = None,
        seq_lens: torch.Tensor | None = None,
    ) -> tuple[dict[Feats | Actions, torch.Tensor], torch.Tensor]:
        if h_detach_period is not None:
            return _forward_w_detach(x, h, h_detach_period, seq_lens, self.forward)

        # Typically we have time in dim=1, here we always(?)
        # (N, L) x nfeat
        fs = _feature_map(x, self.features)

        # (N, L, nfeat)
        inputs = torch.cat(fs, dim=-1)

        # (N, L, nfeat * feat_scale)
        inputs = self.scaler(inputs)

        # (N, L, H) (N, n_hidden, H)
        output, h = self.rnn(inputs, h)

        # (N, L, H)
        output = self.out_norm(output)

        # (N, L, 4) (0=WAIT, 1=SEND_UP, 2=SEND_DOWN, 3=SEND_BOTH)
        action_selector = self.actor["action_selection"](output)

        # (N, L, SEND_COUNT_BINS)
        send_count_u = self.actor["send_count_u"](output)
        send_count_d = self.actor["send_count_d"](output)

        # (N, L, DECAY_TIME_BINS)
        send_time_u = self.actor["send_time_u"](output)
        send_time_d = self.actor["send_time_d"](output)

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

    def _get_probs(self, logits: torch.Tensor, eps: float) -> torch.Tensor:
        probs = nn.functional.softmax(logits, dim=-1)
        probs = (1 - eps) * probs + eps / probs.shape[-1]
        return probs

    def act(
        self,
        x: dict[Feats, torch.Tensor],
        h: torch.Tensor | None = None,
        h_detach_period: int | None = None,
        seq_lens: torch.Tensor | None = None,
        sample: bool = True,
    ) -> tuple[
        torch.Tensor,
        dict[Actions, torch.Tensor],
        torch.Tensor,
        torch.Tensor,
        dict[str, torch.Tensor],
        torch.Tensor,
    ]:
        action_outputs, h = self(x, h, h_detach_period, seq_lens)

        def _select_from_logits(
            logits: torch.Tensor, eps: float
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            probs = self._get_probs(logits, eps)
            if sample:
                dist = Categorical(probs=probs)
                idx = dist.sample()
                logp = dist.log_prob(idx)
                entropy = dist.entropy()
            else:
                # Use logits for the argmax to reduce sensitivity to tiny
                # softmax-level numerical differences between batched vs stepwise.
                idx = logits.argmax(dim=-1)
                logp = torch.log(
                    probs.gather(-1, idx.unsqueeze(-1)).squeeze(-1).clamp(min=1e-12)
                )
                entropy = -(probs * torch.log(probs.clamp(min=1e-12))).sum(dim=-1)
            return idx, logp, entropy, probs

        # Select action (selector head):
        selections, sel_log_probs, sel_entropy, sel_probs = _select_from_logits(
            action_outputs[Actions.SELECTOR], self.prob_eps[Actions.SELECTOR]
        )

        # Send u/d, note! These are conditional on the selection.
        # They will be ignored if the selection is not SEND_UP/DOWN/BOTH.
        send_count_u_idx, send_count_u_logp, suc_entropy, _ = _select_from_logits(
            action_outputs[Actions.SEND_COUNT_UP], self.prob_eps[Actions.SEND_COUNT_UP]
        )
        send_time_u_idx, send_time_u_logp, sudt_entropy, _ = _select_from_logits(
            action_outputs[Actions.SEND_TIME_UP], self.prob_eps[Actions.SEND_TIME_UP]
        )

        send_count_d_idx, send_count_d_logp, sdc_entropy, _ = _select_from_logits(
            action_outputs[Actions.SEND_COUNT_DOWN],
            self.prob_eps[Actions.SEND_COUNT_DOWN],
        )
        send_time_d_idx, send_time_d_logp, sddt_entropy, _ = _select_from_logits(
            action_outputs[Actions.SEND_TIME_DOWN],
            self.prob_eps[Actions.SEND_TIME_DOWN],
        )

        send_count_u = self.send_count_bins[send_count_u_idx]
        send_count_d = self.send_count_bins[send_count_d_idx]
        send_time_u = self.send_time_bins[send_time_u_idx]
        send_time_d = self.send_time_bins[send_time_d_idx]

        # Conditional entropy H[A|S]:
        # =============================
        # (B, L)
        up_p = sel_probs[..., 1] + sel_probs[..., 3]
        down_p = sel_probs[..., 2] + sel_probs[..., 3]

        # (B, L)
        # up_p * (suc_entropy + sudt_entropy) + down_p * (sdc_entropy + sddt_entropy)
        cond_entropy = self.cond_beta * (
            up_p * (sudt_entropy + suc_entropy) + down_p * (sddt_entropy + sdc_entropy)
        )

        # Entropy (B, 1) H[A] = H[S] + H[A|S]
        # Policy, \Pi[A] = \Pi[S] * \Pi[A|S]
        entropies = {
            "selection_entropy": sel_entropy,
            "conditional_entropy": cond_entropy,
        }

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
        log_probs[mask] = sel_log_probs[mask] + self.cond_beta * (
            send_count_u_logp[mask] + send_time_u_logp[mask]
        )
        actions[Actions.SEND_COUNT_DOWN][mask] = 0
        actions[Actions.SEND_TIME_DOWN][mask] = 0

        # SEND DOWN:
        # (B, L)
        mask = selections == 2

        # (B, L)
        log_probs[mask] = sel_log_probs[mask] + self.cond_beta * (
            send_count_d_logp[mask] + send_time_d_logp[mask]
        )
        actions[Actions.SEND_COUNT_UP][mask] = 0
        actions[Actions.SEND_TIME_UP][mask] = 0

        # SEND BOTH:
        # (B, L)
        mask = selections == 3

        # (B, L)
        log_probs[mask] = sel_log_probs[mask] + self.cond_beta * (
            send_count_u_logp[mask]
            + send_time_u_logp[mask]
            + send_count_d_logp[mask]
            + send_time_d_logp[mask]
        )

        # (B, L)
        values = action_outputs[Feats.STATE_VALUE]

        # The actions take place only after the current window is processed -> + Dt
        times = x[Feats.TIMES] + x[Feats.Dt]

        # Map the action names.
        if self.send_mode == "spread":
            actions[Actions.SPREAD_TIME_UP] = actions.pop(Actions.SEND_TIME_UP)
            actions[Actions.SPREAD_TIME_DOWN] = actions.pop(Actions.SEND_TIME_DOWN)
        elif self.send_mode == "fixed":
            actions[Actions.SEND_UP_AFTER_TIME] = actions.pop(Actions.SEND_TIME_UP)
            actions[Actions.SEND_DOWN_AFTER_TIME] = actions.pop(Actions.SEND_TIME_DOWN)
        else:
            raise ValueError(
                f"Invalid send_mode {self.send_mode}, expected 'spread' or 'fix'."
            )

        return times, actions, log_probs, sel_probs, values, entropies, h


class CRITIC01(nn.Module):
    name: str = "critic"

    def __init__(
        self,
        agent: AGENT1,
        hsize: int = 256,
        nlayers: int = 3,
        dropout: float = 0.0,
        n_classes: int = 100,
        label_embedding_dim: int = 1,
        use_label: bool = False,
    ):
        super().__init__()

        if dropout != 0:
            logger.warning("Dropout on critic is bad idea?")

        # Time step between feature extractions.
        self.time_step = agent.time_step
        # Maximum silence the model tolerates before acting.
        self.max_silence_s = agent.max_silence_s

        self.features = agent.features.copy()

        self.use_label = use_label
        if use_label:
            self.features.append(Feats.LABEL)
        else:
            label_embedding_dim = 0

        self.num_layers = nlayers
        self.hidden_size = hsize
        # scaler does not apply to embeddings
        nfeat = len(self.features) - use_label

        self.scaler = nn.Sequential(nn.Linear(nfeat, nfeat, bias=False), nn.Tanh())
        self.rnn = nn.LSTM(
            nfeat + label_embedding_dim,
            hsize,
            nlayers,
            batch_first=True,
            dropout=dropout,
            bias=True,
        )

        self.out_norm = nn.LayerNorm(hsize)

        self.critic = nn.Sequential(
            nn.Linear(hsize, hsize), nn.LeakyReLU(), nn.Linear(hsize, 1)
        )

        self.label_embedding = nn.Embedding(n_classes, label_embedding_dim)

    def forward(
        self,
        x: dict[Feats, torch.Tensor],
        h: torch.Tensor | None = None,
        h_detach_period: int | None = None,
        seq_lens: torch.Tensor | None = None,
    ) -> tuple[dict[Feats | Actions, torch.Tensor], torch.Tensor]:
        if h_detach_period is not None:
            return _forward_w_detach(x, h, h_detach_period, seq_lens, self.forward)

        # Typically we have time in dim=1, here we always(?)
        # (N, L) x nfeat
        fs = _feature_map(x, self.features)

        # (N, L, nfeat)
        inputs = torch.cat(fs, dim=-1)

        # (N, L, nfeat * feat_scale)
        inputs = self.scaler(inputs)

        # (N, L, nfeat * feat_scale + embed_dim + label_embed_dim)
        if self.use_label:
            inputs = torch.cat(
                (inputs, self.label_embedding(x[Feats.LABEL].long())), dim=-1
            )

        # (N, L, H) (N, n_hidden, H)
        output, h = self.rnn(inputs, h)

        # (N, L, H)
        output = self.out_norm(output)

        # (N, L)
        state_values = self.critic(output).squeeze(-1)

        return {Feats.STATE_VALUE: state_values}, h
