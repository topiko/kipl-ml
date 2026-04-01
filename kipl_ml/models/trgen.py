from typing import Any

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical
from torch.nn.utils.rnn import PackedSequence, pack_padded_sequence, pad_packed_sequence

from kipl_ml.logging.logger import get_logger
from kipl_ml.rl.enums import (
    ActDelayDown,
    ActDelayUp,
    ActDoNothing,
    Actions,
    ActSendDown,
    ActSendUp,
    AHKs,
    EntropyKeys,
    StepAction,
    StepActions,
)
from kipl_ml.trace.features import Feats

logger = get_logger(__name__)


def _get_probs(logits: torch.Tensor, eps: float) -> torch.Tensor:
    probs = torch.nn.functional.softmax(logits, dim=-1)
    probs = (1 - eps) * probs + eps / probs.shape[-1]
    return probs


def _select_cat_from_logits(
    logits: torch.Tensor, bins: torch.Tensor, eps: float, sample: bool = True
) -> tuple[np.ndarray, torch.Tensor, torch.Tensor, torch.Tensor]:
    # logits: (bs, L, selector_dim)
    probs = _get_probs(logits, eps)
    if sample:
        # (bs, L, selector_dim)
        dist = Categorical(probs=probs)
        # (bs, L)
        idx = dist.sample()
        # (bs, L)
        value = bins[idx] if bins is not None else idx
        # (bs, L)
        logp = dist.log_prob(idx)
        # (bs, L)
        entropy = dist.entropy()
    else:
        # (bs, L, selector_dim) -> (bs, L)
        idx = logits.argmax(dim=-1)
        # (bs, L)
        value = bins[idx] if bins is not None else idx
        # (bs, L)
        logp = torch.log(
            probs.gather(-1, idx.unsqueeze(-1)).squeeze(-1).clamp(min=1e-12)
        )
        # (bs, L)
        entropy = -(probs * torch.log(probs.clamp(min=1e-12))).sum(dim=-1)

    value = value.detach().cpu().numpy().squeeze(1)  # (bs, )
    return value, logp, entropy, probs


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
        prob_eps: dict[AHKs, float] | float | None = None,
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
        self.register_buffer(
            "send_count_bins", torch.tensor(send_count_bins, dtype=torch.long)
        )
        send_after_bins = send_after_bins or [0, 1, 3, 5, 7, 9]
        self.register_buffer(
            "send_after_bins", torch.tensor(send_after_bins, dtype=torch.long)
        )

        self.ackt_bin_d = {
            AHKs.SEND_COUNT_U: self.send_count_bins,
            AHKs.SEND_TIME_U: self.send_after_bins,
            AHKs.SEND_COUNT_D: self.send_count_bins,
            AHKs.SEND_TIME_D: self.send_after_bins,
            AHKs.SEND_BYPASS_D: None,
            AHKs.SEND_REPLACE_D: None,
            AHKs.SEND_BYPASS_U: None,
            AHKs.SEND_REPLACE_U: None,
            AHKs.ACTION_SELECTION: None,
        }

        delay_duration_bins = delay_duration_bins or [1, 2, 4, 8]

        # Delay actions:
        if self.enable_delay:
            self.ACTIONS += [Actions.DELAY_UP, Actions.DELAY_DOWN]
            self.register_buffer(
                "delay_duration_bins",
                torch.tensor(delay_duration_bins, dtype=torch.long),
            )
            self.ackt_bin_d.update(
                {
                    AHKs.DELAY_BINS_U: self.delay_duration_bins,
                    AHKs.DELAY_BINS_D: self.delay_duration_bins,
                    AHKs.DELAY_BYPASS_U: None,
                    AHKs.DELAY_BYPASS_D: None,
                    AHKs.DELAY_REPLACE_U: None,
                    AHKs.DELAY_REPLACE_D: None,
                }
            )
            n_delay_durations = len(self.delay_duration_bins)
        else:
            n_delay_durations = 0

        ratio = float(max_silence_s) / float(time_step)
        if abs(ratio - round(ratio)) > 1e-8:
            raise ValueError(
                "max_silence_s should be a multiple of time_step."
                + f" Got max_silence_s={max_silence_s}, time_step={time_step}."
            )

        n_send_counts = len(self.send_count_bins)
        n_decay_times = len(self.send_after_bins)

        # Time step between feature extractions.
        self.time_step = time_step
        # Maximum silence the model tolerates before acting.
        self.max_silence_s = max_silence_s

        # Exploration prob eps for each action:
        # Initialize prob_eps for all AHKs (will be set in the block below)
        self.prob_eps: dict[AHKs, float] = {a: 0.0 for a in AHKs}
        if prob_eps is not None:
            for a in AHKs:
                if a not in self.ackt_bin_d:
                    continue
                prob = prob_eps[a] if isinstance(prob_eps, dict) else prob_eps
                if not (0.0 <= prob <= 1.0):
                    raise ValueError(
                        f"Invalid prob_eps for action {a}: {prob}. Must be in [0, 1]."
                    )
                self.prob_eps[a] = prob

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

    def forward(
        self,
        x: dict[Feats, torch.Tensor],
        h: torch.Tensor | None = None,
        h_detach_period: int | None = None,
        seq_lens: torch.Tensor | None = None,
    ) -> tuple[dict[AHKs | Feats, torch.Tensor], torch.Tensor]:
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
        StepActions,
        torch.Tensor,
        torch.Tensor,
        dict[str, torch.Tensor],
        torch.Tensor,
    ]:
        action_outputs, h = self(x, h, h_detach_period, seq_lens)

        values = action_outputs.pop(Feats.STATE_VALUE)  # type: ignore

        if values.shape[1] != 1:
            raise ValueError(f"act expects L=1, got L={values.shape[1]}")

        # dict[AHKs, tuple[value (np.arr), logp, entropy, probs]],
        # where the shapes are (B, L (==1 !))
        act_tup_d = {
            k: _select_cat_from_logits(v, self.ackt_bin_d[k], self.prob_eps[k], sample)
            for k, v in action_outputs.items()
        }

        # Entropy work, be careful w. this one.
        sel_probs = act_tup_d[AHKs.ACTION_SELECTION][3]

        up_p = sel_probs[..., 1] + sel_probs[..., 3]
        down_p = sel_probs[..., 2] + sel_probs[..., 3]

        # Entropy H(A) = H(selector, action_params) = H(selector)
        # + H(action_params | selector) <===
        # NOTE: This computes E[H(params) * P(selector)] which is an approximation of
        # the true conditional entropy H(params | selector). The true conditional
        # entropy would require computing p(params, selector) from the shared trunk
        # output. We hope this approximation is sufficient for exploration as we
        # already have a floor on action probabilities via prob_eps in _get_probs.
        cond_entropy = up_p * (
            act_tup_d[AHKs.SEND_COUNT_U][2]
            + act_tup_d[AHKs.SEND_TIME_U][2]
            + act_tup_d[AHKs.SEND_BYPASS_U][2]
            + act_tup_d[AHKs.SEND_REPLACE_U][2]
        ) + down_p * (
            act_tup_d[AHKs.SEND_COUNT_D][2]
            + act_tup_d[AHKs.SEND_TIME_D][2]
            + act_tup_d[AHKs.SEND_BYPASS_D][2]
            + act_tup_d[AHKs.SEND_REPLACE_D][2]
        )
        if self.enable_delay:
            up_delay_p = sel_probs[..., 4] + sel_probs[..., 6]
            down_delay_p = sel_probs[..., 5] + sel_probs[..., 6]
            cond_entropy = cond_entropy + (
                down_delay_p
                * (
                    act_tup_d[AHKs.DELAY_BINS_D][2]
                    + act_tup_d[AHKs.DELAY_BYPASS_D][2]
                    + act_tup_d[AHKs.DELAY_REPLACE_D][2]
                )
                + up_delay_p
                * (
                    act_tup_d[AHKs.DELAY_BINS_U][2]
                    + act_tup_d[AHKs.DELAY_BYPASS_U][2]
                    + act_tup_d[AHKs.DELAY_REPLACE_U][2]
                )
            )

        selection_entropy = act_tup_d[AHKs.ACTION_SELECTION][2]

        entropies = {
            EntropyKeys.SELECTION_ENTROPY: selection_entropy,
            EntropyKeys.COND_ENTROPY: cond_entropy,
        }

        # (B, )
        selector_idx = act_tup_d[AHKs.ACTION_SELECTION][0]

        # Get log probs
        sel_log_probs = act_tup_d[AHKs.ACTION_SELECTION][1]
        log_probs = torch.zeros_like(sel_log_probs)
        time_bins = x[Feats.TIME_BINS] + x[Feats.Dt_BINS]
        bs = time_bins.shape[0]

        # Build sparse StepActions per batch item
        step_actions: list[StepAction] = []
        for b in range(bs):
            sel = selector_idx[b]
            t = time_bins[b, 0].item()
            if sel == 0:
                sa = StepAction(time=t, _actions={Actions.DO_NOTHING: ActDoNothing()})
                log_probs[b, 0] = sel_log_probs[b, 0]
            if sel == 1:
                sa = StepAction(
                    time=t,
                    _actions={
                        Actions.SEND_UP: ActSendUp(
                            count=act_tup_d[AHKs.SEND_COUNT_U][0][b],
                            after_steps=act_tup_d[AHKs.SEND_TIME_U][0][b],
                            bypass=bool(act_tup_d[AHKs.SEND_BYPASS_U][0][b]),
                            replace=bool(act_tup_d[AHKs.SEND_REPLACE_U][0][b]),
                        )
                    },
                )
                log_probs[b, 0] = (
                    sel_log_probs[b, 0]
                    + act_tup_d[AHKs.SEND_COUNT_U][1][b]
                    + act_tup_d[AHKs.SEND_TIME_U][1][b]
                    + act_tup_d[AHKs.SEND_BYPASS_U][1][b]
                    + act_tup_d[AHKs.SEND_REPLACE_U][1][b]
                )
            elif sel == 2:
                sa = StepAction(
                    time=t,
                    _actions={
                        Actions.SEND_DOWN: ActSendDown(
                            count=act_tup_d[AHKs.SEND_COUNT_D][0][b],
                            after_steps=act_tup_d[AHKs.SEND_TIME_D][0][b],
                            bypass=bool(act_tup_d[AHKs.SEND_BYPASS_D][0][b]),
                            replace=bool(act_tup_d[AHKs.SEND_REPLACE_D][0][b]),
                        )
                    },
                )
                log_probs[b, 0] = (
                    sel_log_probs[b, 0]
                    + act_tup_d[AHKs.SEND_COUNT_D][1][b]
                    + act_tup_d[AHKs.SEND_TIME_D][1][b]
                    + act_tup_d[AHKs.SEND_BYPASS_D][1][b]
                    + act_tup_d[AHKs.SEND_REPLACE_D][1][b]
                )
            elif sel == 3:
                sa = StepAction(
                    time=t,
                    _actions={
                        Actions.SEND_UP: ActSendUp(
                            count=act_tup_d[AHKs.SEND_COUNT_U][0][b],
                            after_steps=act_tup_d[AHKs.SEND_TIME_U][0][b],
                            bypass=bool(act_tup_d[AHKs.SEND_BYPASS_U][0][b]),
                            replace=bool(act_tup_d[AHKs.SEND_REPLACE_U][0][b]),
                        ),
                        Actions.SEND_DOWN: ActSendDown(
                            count=act_tup_d[AHKs.SEND_COUNT_D][0][b],
                            after_steps=act_tup_d[AHKs.SEND_TIME_D][0][b],
                            bypass=bool(act_tup_d[AHKs.SEND_BYPASS_D][0][b]),
                            replace=bool(act_tup_d[AHKs.SEND_REPLACE_D][0][b]),
                        ),
                    },
                )
                log_probs[b, 0] = (
                    sel_log_probs[b, 0]
                    + act_tup_d[AHKs.SEND_COUNT_U][1][b]
                    + act_tup_d[AHKs.SEND_TIME_U][1][b]
                    + act_tup_d[AHKs.SEND_BYPASS_U][1][b]
                    + act_tup_d[AHKs.SEND_REPLACE_U][1][b]
                    + act_tup_d[AHKs.SEND_COUNT_D][1][b]
                    + act_tup_d[AHKs.SEND_TIME_D][1][b]
                    + act_tup_d[AHKs.SEND_BYPASS_D][1][b]
                    + act_tup_d[AHKs.SEND_REPLACE_D][1][b]
                )
            elif sel >= 4:
                if not self.enable_delay:
                    raise ValueError(
                        f"Invalid selector index {sel} for non-delay model."
                    )
                if sel == 4:
                    sa = StepAction(
                        time=t,
                        _actions={
                            Actions.DELAY_UP: ActDelayUp(
                                steps=act_tup_d[AHKs.DELAY_BINS_U][0][b],
                                bypass=bool(act_tup_d[AHKs.DELAY_BYPASS_U][0][b]),
                                replace=bool(act_tup_d[AHKs.DELAY_REPLACE_U][0][b]),
                            )
                        },
                    )
                    log_probs[b, 0] = (
                        sel_log_probs[b, 0]
                        + act_tup_d[AHKs.DELAY_BINS_U][1][b]
                        + act_tup_d[AHKs.DELAY_BYPASS_U][1][b]
                        + act_tup_d[AHKs.DELAY_REPLACE_U][1][b]
                    )
                elif sel == 5:
                    sa = StepAction(
                        time=t,
                        _actions={
                            Actions.DELAY_DOWN: ActDelayDown(
                                steps=act_tup_d[AHKs.DELAY_BINS_D][0][b],
                                bypass=bool(act_tup_d[AHKs.DELAY_BYPASS_D][0][b]),
                                replace=bool(act_tup_d[AHKs.DELAY_REPLACE_D][0][b]),
                            )
                        },
                    )
                    log_probs[b, 0] = (
                        sel_log_probs[b, 0]
                        + act_tup_d[AHKs.DELAY_BINS_D][1][b]
                        + act_tup_d[AHKs.DELAY_BYPASS_D][1][b]
                        + act_tup_d[AHKs.DELAY_REPLACE_D][1][b]
                    )
                elif sel == 6:
                    sa = StepAction(
                        time=t,
                        _actions={
                            Actions.DELAY_UP: ActDelayUp(
                                steps=act_tup_d[AHKs.DELAY_BINS_U][0][b],
                                bypass=bool(act_tup_d[AHKs.DELAY_BYPASS_U][0][b]),
                                replace=bool(act_tup_d[AHKs.DELAY_REPLACE_U][0][b]),
                            ),
                            Actions.DELAY_DOWN: ActDelayDown(
                                steps=act_tup_d[AHKs.DELAY_BINS_D][0][b],
                                bypass=bool(act_tup_d[AHKs.DELAY_BYPASS_D][0][b]),
                                replace=bool(act_tup_d[AHKs.DELAY_REPLACE_D][0][b]),
                            ),
                        },
                    )
                    log_probs[b, 0] = (
                        sel_log_probs[b, 0]
                        + act_tup_d[AHKs.DELAY_BINS_U][1][b]
                        + act_tup_d[AHKs.DELAY_BYPASS_U][1][b]
                        + act_tup_d[AHKs.DELAY_REPLACE_U][1][b]
                        + act_tup_d[AHKs.DELAY_BINS_D][1][b]
                        + act_tup_d[AHKs.DELAY_BYPASS_D][1][b]
                        + act_tup_d[AHKs.DELAY_REPLACE_D][1][b]
                    )
                else:
                    raise ValueError(f"Invalid selector index {sel} for delay model.")

            step_actions.append(sa)

        return time_bins, step_actions, log_probs, sel_probs, values, entropies, h

    def act_step(
        self,
        x: dict[Feats, torch.Tensor],
        h: torch.Tensor | None = None,
        sample: bool = True,
    ) -> tuple[
        torch.Tensor,
        list[StepAction],
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
