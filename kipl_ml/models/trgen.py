from typing import Any

import torch
from torch import nn
from torch.distributions import Categorical
from torch.nn.utils.rnn import PackedSequence, pack_padded_sequence, pad_packed_sequence

from kipl_ml.logging.logger import get_logger
from kipl_ml.rl.enums import AHKs, Actions
from kipl_ml.trace.features import Feats

logger = get_logger(__name__)


def _get_probs(logits: torch.Tensor, eps: float) -> torch.Tensor:
    probs = torch.nn.functional.softmax(logits, dim=-1)
    probs = (1 - eps) * probs + eps / probs.shape[-1]
    return probs


def _select_cat_from_logits(
    logits: torch.Tensor, eps: float, sample: bool = True
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    probs = _get_probs(logits, eps)
    if sample:
        dist = Categorical(probs=probs)
        idx = dist.sample()
        logp = dist.log_prob(idx)
        entropy = dist.entropy()
    else:
        idx = logits.argmax(dim=-1)
        logp = torch.log(
            probs.gather(-1, idx.unsqueeze(-1)).squeeze(-1).clamp(min=1e-12)
        )
        entropy = -(probs * torch.log(probs.clamp(min=1e-12))).sum(dim=-1)
    return idx, logp, entropy, probs


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
    x: dict[Feats, torch.Tensor],
    f: Feats | list[Feats],
    dt: float | None = None,
) -> list[torch.Tensor] | torch.Tensor:
    def _map_one(fi: Feats) -> torch.Tensor:
        v = x[fi]
        # Convert int bins to seconds; skip if already float.
        if (
            fi in (Feats.TIME_BINS, Feats.Dt_BINS)
            and dt is not None
            and not v.is_floating_point()
        ):
            v = v.float() * dt
        else:
            v = v.float()
        if fi == Feats.SILENCE_FLAG:
            return v.unsqueeze(-1)  # keep 0/1
        if fi in (Feats.TIME_BINS, Feats.TAM_TIMES):
            return (v / (v + 10)).unsqueeze(-1)
        return torch.log1p(v).unsqueeze(-1)

    if isinstance(f, list):
        return [_map_one(fi) for fi in f]
    return _map_one(f)


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

        if set(features).issubset(
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
            raise ValueError(
                f"Invalid set of feats. {'-'.join(features)}. Only TAM features are supported."
            )

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
        x_chunk = {}
        for k, v in x.items():
            chunk = v[active_seqs, i * h_detach_period : (i + 1) * h_detach_period]
            if chunk.is_floating_point():
                x_chunk[k] = torch.where(chunk.isfinite(), chunk, 0)
            else:
                x_chunk[k] = torch.where(chunk >= 0, chunk, 0)

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
        send_after_bins: list[int] | None = None,
        delay_duration_bins: list[int] | None = None,
        prob_eps: dict[Actions, float] | float | None = None,
        prefer_wait_bias: float = 0.0,
        enable_delay: bool = False,
        train_env: dict[str, Any] | None = None,
    ):
        super().__init__()

        self.train_env = train_env or {}
        self.enable_delay = bool(enable_delay)

        # Action space:
        self.ACTIONS: list[Actions] = [Actions.SEND_UP, Actions.SEND_DOWN]
        send_count_bins = send_count_bins or [5, 20, 50, 100, 200]
        send_after_bins = send_after_bins or [0, 1, 3, 5, 7, 9]

        # Delay actions:
        if self.enable_delay:
            self.ACTIONS += [Actions.DELAY_UP, Actions.DELAY_DOWN]

        delay_duration_bins = delay_duration_bins or [1, 2, 4, 8]

        ratio = float(max_silence_s) / float(time_step)
        if abs(ratio - round(ratio)) > 1e-8:
            raise ValueError(
                "max_silence_s should be a multiple of time_step."
                + f" Got max_silence_s={max_silence_s}, time_step={time_step}."
            )

        n_send_counts = len(send_count_bins)
        n_decay_times = len(send_after_bins)
        n_delay_durations = len(delay_duration_bins)

        # Buffers obey device...
        self.register_buffer(
            "send_count_bins", torch.tensor(send_count_bins, dtype=torch.long)
        )
        self.register_buffer(
            "send_after_bins", torch.tensor(send_after_bins, dtype=torch.long)
        )
        self.register_buffer(
            "delay_duration_bins", torch.tensor(delay_duration_bins, dtype=torch.long)
        )

        # Time step between feature extractions.
        self.time_step = time_step
        # Maximum silence the model tolerates before acting.
        self.max_silence_s = max_silence_s

        # Exploration prob eps for each action:
        control_actions = self.ACTIONS
        if prob_eps is not None:
            if isinstance(prob_eps, float):
                prob_eps = {a: prob_eps for a in control_actions}
            elif not set(prob_eps.keys()).issuperset(set(control_actions)):
                raise ValueError("prob_eps keys must cover all actions.")
            self.prob_eps = {a: prob_eps[a] for a in control_actions}
        else:
            self.prob_eps = {a: 0.0 for a in control_actions}

        # Input features:
        self.features = [
            Feats.UP_COUNT,
            Feats.DOWN_COUNT,
            Feats.Dt_BINS,
            Feats.TIME_BINS,
            Feats.SILENCE_FLAG,
        ]

        # Architecture params:
        self.num_layers = nlayers
        self.hidden_size = hsize
        nfeat = len(self.features)

        self.scaler = nn.Sequential(nn.Linear(nfeat, nfeat, bias=False), nn.Tanh())
        self.rnn = nn.LSTM(nfeat, hsize, nlayers, batch_first=True, bias=True)

        self.out_norm = nn.LayerNorm(hsize)

        selector_dim = 4 if not self.enable_delay else 7
        actor_heads = {
            AHKs.ACTION_SELECTION: nn.Sequential(nn.Linear(hsize, selector_dim)),
            AHKs.SEND_COUNT_U: nn.Sequential(nn.Linear(hsize, n_send_counts)),
            AHKs.SEND_BYPASS_U: nn.Sequential(nn.Linear(hsize, 2)),
            AHKs.SEND_REPLACE_U: nn.Sequential(nn.Linear(hsize, 2)),
            AHKs.SEND_COUNT_D: nn.Sequential(nn.Linear(hsize, n_send_counts)),
            AHKs.SEND_BYPASS_D: nn.Sequential(nn.Linear(hsize, 2)),
            AHKs.SEND_REPLACE_D: nn.Sequential(nn.Linear(hsize, 2)),
            AHKs.SEND_TIME_U: nn.Sequential(nn.Linear(hsize, n_decay_times)),
            AHKs.SEND_TIME_D: nn.Sequential(nn.Linear(hsize, n_decay_times)),
        }
        if self.enable_delay:
            delay_heads = {
                AHKs.DELAY_BINS_U: nn.Sequential(nn.Linear(hsize, n_delay_durations)),
                AHKs.DELAY_BYPASS_U: nn.Sequential(nn.Linear(hsize, 2)),
                AHKs.DELAY_REPLACE_U: nn.Sequential(nn.Linear(hsize, 2)),
                AHKs.DELAY_BINS_D: nn.Sequential(nn.Linear(hsize, n_delay_durations)),
                AHKs.DELAY_BYPASS_D: nn.Sequential(nn.Linear(hsize, 2)),
                AHKs.DELAY_REPLACE_D: nn.Sequential(nn.Linear(hsize, 2)),
            }
            actor_heads.update(delay_heads)

        self.actor = nn.ModuleDict(actor_heads)

        self.critic = nn.Sequential(
            nn.Linear(hsize, hsize), nn.ReLU(), nn.Linear(hsize, 1)
        )
        self.cond_beta = 1.0

        if prefer_wait_bias != 0.0:
            self._init_action_selection_prefer_wait(prefer_wait_bias=prefer_wait_bias)

    def _init_action_selection_prefer_wait(self, prefer_wait_bias: float) -> None:
        """Initialize selector logits to heavily prefer DO_NOTHING (selector index 0)."""
        if prefer_wait_bias < 0:
            raise ValueError(f"prefer_wait_bias must be >= 0, got {prefer_wait_bias}")

        head = self.actor["action_selection"]
        lin = head[-1]
        if not isinstance(lin, nn.Linear) or lin.out_features not in {4, 5}:
            raise TypeError("action_selection head must end with Linear(..., 4|5)")

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
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        if h_detach_period is not None:
            return _forward_w_detach(x, h, h_detach_period, seq_lens, self.forward)

        # Typically we have time in dim=1, here we always(?)
        # (N, L) x nfeat
        fs = _feature_map(x, self.features, dt=float(self.time_step))

        # (N, L, nfeat)
        inputs = torch.cat(fs, dim=-1)

        # (N, L, nfeat * feat_scale)
        inputs = self.scaler(inputs)

        # (N, L, H) (N, n_hidden, H)
        output, h = self.rnn(inputs, h)

        # (N, L, H)
        output = self.out_norm(output)

        # (N, L)
        state_values = self.critic(output).squeeze(-1)

        out = {k: mod_(output) for k, mod_ in self.actor.items()}
        out[Feats.STATE_VALUE] = state_values

        return out, h

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

        selections, sel_log_probs, sel_entropy, sel_probs = _select_cat_from_logits(
            action_outputs[Actions.SELECTOR], self.prob_eps[Actions.SELECTOR], sample
        )

        send_count_u_idx, send_count_u_logp, suc_entropy, _ = _select_cat_from_logits(
            action_outputs[Actions.SEND_COUNT_UP],
            self.prob_eps[Actions.SEND_COUNT_UP],
            sample,
        )
        send_time_u_idx, send_time_u_logp, sudt_entropy, _ = _select_cat_from_logits(
            action_outputs[Actions.SEND_UP_AFTER_BINS],
            self.prob_eps[Actions.SEND_UP_AFTER_BINS],
            sample,
        )

        send_count_d_idx, send_count_d_logp, sdc_entropy, _ = _select_cat_from_logits(
            action_outputs[Actions.SEND_COUNT_DOWN],
            self.prob_eps[Actions.SEND_COUNT_DOWN],
            sample,
        )
        send_time_d_idx, send_time_d_logp, sddt_entropy, _ = _select_cat_from_logits(
            action_outputs[Actions.SEND_DOWN_AFTER_BINS],
            self.prob_eps[Actions.SEND_DOWN_AFTER_BINS],
            sample,
        )

        delay_dur_logp = torch.zeros_like(sel_log_probs)
        delay_entropy = torch.zeros_like(sel_entropy)
        delay_bins = torch.zeros_like(x[Feats.Dt_BINS]).to(torch.long)
        if self.enable_delay and sel_probs.shape[-1] >= 5:
            delay_idx, delay_dur_logp, delay_entropy, _ = _select_cat_from_logits(
                action_outputs[Actions.DELAY_BINS],
                self.prob_eps[Actions.DELAY_BINS],
                sample,
            )
            delay_bins = self.delay_duration_bins[delay_idx]
        elif self.enable_delay:
            raise ValueError(
                "enable_delay is True but action selector has < 5 outputs."
            )

        send_count_u = self.send_count_bins[send_count_u_idx]
        send_count_d = self.send_count_bins[send_count_d_idx]
        send_time_u = self.send_after_bins[send_time_u_idx]  # int bins
        send_time_d = self.send_after_bins[send_time_d_idx]  # int bins

        up_p = sel_probs[..., 1] + sel_probs[..., 3]
        down_p = sel_probs[..., 2] + sel_probs[..., 3]

        cond_entropy = self.cond_beta * (
            up_p * (sudt_entropy + suc_entropy) + down_p * (sddt_entropy + sdc_entropy)
        )
        if self.enable_delay and sel_probs.shape[-1] >= 5:
            cond_entropy = (
                cond_entropy + self.cond_beta * sel_probs[..., 4] * delay_entropy
            )

        entropies = {
            "selection_entropy": sel_entropy,
            "conditional_entropy": cond_entropy,
        }

        log_probs = torch.zeros_like(sel_log_probs)

        actions: dict[Actions, torch.Tensor] = {
            Actions.SELECTOR: selections.detach().clone(),
            Actions.DO_NOTHING: torch.zeros_like(selections),
            Actions.SEND_COUNT_DOWN: send_count_d.detach().clone(),
            Actions.SEND_COUNT_UP: send_count_u.detach().clone(),
            Actions.SEND_DOWN_AFTER_BINS: send_time_d.detach().clone(),
            Actions.SEND_UP_AFTER_BINS: send_time_u.detach().clone(),
        }
        if self.enable_delay and sel_probs.shape[-1] >= 5:
            actions[Actions.DELAY_BINS] = torch.zeros_like(delay_bins).detach().clone()

        mask = selections == 0
        log_probs[mask] = sel_log_probs[mask]
        actions[Actions.DO_NOTHING][mask] = 1
        actions[Actions.SEND_COUNT_UP][mask] = 0
        actions[Actions.SEND_COUNT_DOWN][mask] = 0
        actions[Actions.SEND_UP_AFTER_BINS][mask] = 0
        actions[Actions.SEND_DOWN_AFTER_BINS][mask] = 0

        if self.enable_delay and sel_probs.shape[-1] >= 5:
            mask = selections == 4
            log_probs[mask] = (
                sel_log_probs[mask] + self.cond_beta * delay_dur_logp[mask]
            )
            actions[Actions.DELAY_BINS][mask] = (
                delay_bins[mask].to(torch.long).clamp(min=1)
            )
            actions[Actions.DO_NOTHING][mask] = 0
            actions[Actions.SEND_COUNT_UP][mask] = 0
            actions[Actions.SEND_COUNT_DOWN][mask] = 0
            actions[Actions.SEND_UP_AFTER_BINS][mask] = 0
            actions[Actions.SEND_DOWN_AFTER_BINS][mask] = 0

        mask = selections == 1
        log_probs[mask] = sel_log_probs[mask] + self.cond_beta * (
            send_count_u_logp[mask] + send_time_u_logp[mask]
        )
        actions[Actions.SEND_COUNT_DOWN][mask] = 0
        actions[Actions.SEND_DOWN_AFTER_BINS][mask] = 0

        mask = selections == 2
        log_probs[mask] = sel_log_probs[mask] + self.cond_beta * (
            send_count_d_logp[mask] + send_time_d_logp[mask]
        )
        actions[Actions.SEND_COUNT_UP][mask] = 0
        actions[Actions.SEND_UP_AFTER_BINS][mask] = 0

        mask = selections == 3
        log_probs[mask] = sel_log_probs[mask] + self.cond_beta * (
            send_count_u_logp[mask]
            + send_time_u_logp[mask]
            + send_count_d_logp[mask]
            + send_time_d_logp[mask]
        )

        values = action_outputs[Feats.STATE_VALUE]
        # x[TIME_BINS] and x[Dt_BINS] are int bins; action time = current bin + dt bins.
        # Invalid entries have -1; result will be negative for those.
        time_bins = x[Feats.TIME_BINS] + x[Feats.Dt_BINS]

        return time_bins, actions, log_probs, sel_probs, values, entropies, h

    def act_step(
        self,
        x: dict[Feats, torch.Tensor],
        h: torch.Tensor | None = None,
        sample: bool = True,
    ) -> tuple[
        torch.Tensor,
        dict[Actions, torch.Tensor],
        torch.Tensor,
        torch.Tensor,
        dict[str, torch.Tensor],
        torch.Tensor,
        torch.Tensor,
    ]:
        """Single-step action API (expects L=1)."""

        for f in self.features:
            v = x[f]
            if v.ndim != 2 or v.shape[1] != 1:
                raise ValueError("act_step expects each feature to be (B,1)")

        return self.act(
            x,
            h=h,
            h_detach_period=None,
            seq_lens=None,
            sample=sample,
        )


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
        fs = _feature_map(x, self.features, dt=float(self.time_step))

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
