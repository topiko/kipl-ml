from __future__ import annotations

from time import perf_counter

import torch

from kipl_ml.defences.models.trgen import AGENT1, _feature_map, _select_cat_from_logits
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
)
from kipl_ml.trace.features import Feats


class ProfiledAGENT1(AGENT1):
    """AGENT1 subclass with per-operation timer instrumentation.

    Accumulates wall-clock timers across calls.  Retrieve with
    ``get_timers(reset=True)``.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._timers: dict[str, float] = {}
        self._steps = 0

    def _acc(self, key: str, dt: float) -> None:
        self._timers[key] = self._timers.get(key, 0.0) + dt

    def get_timers(self, reset: bool = True) -> dict[str, float]:
        out = dict(self._timers)
        out["steps"] = self._steps
        if reset:
            self._timers.clear()
            self._steps = 0
        return out

    # ── forward ──────────────────────────────────────────────────────

    def forward(
        self,
        x: dict[Feats, torch.Tensor],
        h: torch.Tensor | None = None,
        h_detach_period: int | None = None,
        seq_lens: torch.Tensor | None = None,
    ):
        if h_detach_period is not None:
            return super().forward(x, h, h_detach_period, seq_lens)

        t0 = perf_counter()
        fs = _feature_map(x, self.features, dt=float(self.time_step))
        inputs = torch.cat(fs, dim=-1)
        self._acc("fw_feature_map", perf_counter() - t0)

        t0 = perf_counter()
        inputs = self.scaler(inputs)
        self._acc("fw_scaler", perf_counter() - t0)

        t0 = perf_counter()
        output, h = self.rnn(inputs, h)
        self._acc("fw_rnn", perf_counter() - t0)

        t0 = perf_counter()
        output = self.out_norm(output)
        self._acc("fw_out_norm", perf_counter() - t0)

        t0 = perf_counter()
        state_values = self.critic(output).squeeze(-1)
        self._acc("fw_critic", perf_counter() - t0)

        t0 = perf_counter()
        out = {k: mod_(output) for k, mod_ in self.actor.items()}
        self._acc("fw_actor_heads", perf_counter() - t0)

        out[Feats.STATE_VALUE] = state_values
        return out, h

    # ── act ──────────────────────────────────────────────────────────

    def act(
        self,
        x: dict[Feats, torch.Tensor],
        h: torch.Tensor | None = None,
        h_detach_period: int | None = None,
        seq_lens: torch.Tensor | None = None,
        sample: bool = True,
    ):
        t_fw = perf_counter()
        action_outputs, h = self(x, h, h_detach_period, seq_lens)
        self._acc("act_forward", perf_counter() - t_fw)

        values = action_outputs.pop(Feats.STATE_VALUE)

        if values.shape[1] != 1:
            raise ValueError(f"act expects L=1, got L={values.shape[1]}")

        t0 = perf_counter()
        act_tup_d = {
            k: _select_cat_from_logits(v, self.ackt_bin_d[k], self.prob_eps[k], sample)
            for k, v in action_outputs.items()
        }
        self._acc("act_select_action", perf_counter() - t0)

        t0 = perf_counter()
        sel_probs = act_tup_d[AHKs.ACTION_SELECTION][3]

        up_p = sel_probs[..., 1] + sel_probs[..., 3]
        down_p = sel_probs[..., 2] + sel_probs[..., 3]

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

        sel_log_probs = act_tup_d[AHKs.ACTION_SELECTION][1]
        log_probs = torch.zeros_like(sel_log_probs)
        time_bins = x[Feats.TIME_BINS] + x[Feats.Dt_BINS]
        bs = time_bins.shape[0]
        self._acc("act_entropy", perf_counter() - t0)

        t0 = perf_counter()
        selector_idx = act_tup_d[AHKs.ACTION_SELECTION][0]

        step_actions: list[StepAction] = []
        for b in range(bs):
            sel = selector_idx[b]
            t = time_bins[b, 0].item()
            if sel == 0:
                sa = StepAction(
                    time_bin=t, _actions={Actions.DO_NOTHING: ActDoNothing()}
                )
                log_probs[b, 0] = sel_log_probs[b, 0]
            if sel == 1:
                sa = StepAction(
                    time_bin=t,
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
                    time_bin=t,
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
                    time_bin=t,
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
                        time_bin=t,
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
                        time_bin=t,
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
                        time_bin=t,
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

        self._acc("act_step_build", perf_counter() - t0)
        self._steps += 1

        return time_bins, step_actions, log_probs, sel_probs, values, entropies, h

    def act_step(
        self,
        x: dict[Feats, torch.Tensor],
        h: torch.Tensor | None = None,
        sample: bool = True,
    ):
        for f in self.features:
            v = x[f]
            if v.ndim != 2 or v.shape[1] != 1:
                raise ValueError("act_step expects each feature to be (B,1)")

        return self.act(x, h=h, h_detach_period=None, seq_lens=None, sample=sample)
