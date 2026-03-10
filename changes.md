# feature/delay vs dev

This branch adds an interactive (stepwise) obfuscation mode for obsrl so the policy can affect *future* observation windows, and introduces a new exclusive `DELAY` action.

The goal is correctness first (equivalence with the old single-pass pipeline when delay is disabled), then usability (debug plots + forced action patterns), then enough performance to train and sweep configs.

## Desired Behavior

- `DELAY` is a new selector option.
- `DELAY` is exclusive: it cannot co-occur with any send action in the same step.
- Delay duration is fixed to one model time-step (`obs.time_step_s`, e.g. 0.02s). No separate delay-time head.
- Delay semantics: applying delay at time `t0` blocks the window `[t0, t0+dt)`.
  - Packets that would occur inside that interval are clamped to `t0+dt`.
  - Packets outside the interval are unchanged (no global shift of the whole future trace).
- Stable packet ordering: if multiple packets end up with the same timestamp (e.g. after clamping), their relative order must follow original packet order.
- When delay is disabled, the stepwise/streaming rollout must match the old precomputed-window single-pass rollout (up to expected numerical ties).
- TAM reward mapping assumptions are enforced: discriminator TAM bin width must match `obs.time_step_s`.

## What Changed (High-Level)

- Streaming window generation:
  - Added `WindowFeatureStreamer` which yields one observation window at a time.
  - This enables actions (especially `DELAY`) to affect which windows are produced next.

- Stepwise trace execution:
  - Introduced a stepwise execution state that accumulates actions and finalizes once, avoiding repeated re-sorts.

- New `DELAY` action:
  - Added `Actions.DELAY` and expanded `AGENT1(enable_delay=True)` selector head.
  - Delay duration is fixed to `obs.time_step`.

- Reward mapping fixes for TAM:
  - Action-intervals may legitimately contain zero TAM bins (no longer an error).
  - Only fail fast when action times truly exceed disc coverage; tolerate boundary/float noise.

- TIMES padding correctness:
  - Stopped using `0` as a TIMES padding sentinel.
  - Use mask-based filling for TIMES and keep strict sentinel checks only where it is actually a sentinel.

- NN defence (RNNDef) integration:
  - Delay-enabled policies run via streaming rollout.
  - Added inference-mode execution paths to reduce overhead.
  - Fixed DataLoader crash when cached traces contain inference tensors (avoid in-place time shift).

## Files Touched / Added

- `kipl_ml/models/trgen.py`
  - `AGENT1` supports `enable_delay` and outputs `DELAY` with duration `time_step`.
  - Selector mapping updated (`DO_NOTHING` replaces `WAIT` with alias kept for compat).

- `kipl_ml/rl/enums.py`
  - Adds `Actions.DELAY`.
  - Adds `Actions.DO_NOTHING` (with compatibility alias for older `WAIT` naming).

- `kipl_ml/rl/action.py`
  - Implements delay clamping semantics and stable ordering for equal timestamps.
  - Adds `TraceExecState` to accumulate stepwise actions and finalize once.

- `kipl_ml/rl/observation.py`
  - Adds `WindowFeatureStreamer`.
  - Delay-aware cursor logic so future windows reflect applied delays.

- `kipl_ml/rl/simulate.py` (new)
  - Shared rollout utilities used by training and defences:
    - `policy_rollout_single_pass(...)`
    - `policy_rollout_streaming(...)` (delay-aware)
    - Lightweight trace-only helpers for inference/defences.

- `experiment/obsrl/sim.py`
  - Training rollout selects single-pass vs streaming based on `obs.enable_delay`.
  - TAM reward mapping hardened:
    - fail-fast if truly beyond coverage
    - tolerate boundary/rounding overhang
    - reduce warning spam for ~0 overhang

- `experiment/obsrl/sisyphus.py`, `experiment/obsrl/obs_agent_01.py`
  - Assert TAM discriminator bin width matches `obs.time_step_s`.

- `kipl_ml/defences/nndefs.py`
  - `RNNDef` uses streaming rollout for delay-enabled models.
  - Uses inference-mode rollout helpers.

- Debug / verification scripts:
  - `experiment/obsrl/debug_window_streamer_equiv.py`: streamer vs precomputed windows.
  - `experiment/obsrl/debug_rollout_equiv.py`: single-pass vs streaming equivalence, timing, plotting, forced selector patterns.

- Plotting:
  - `kipl_ml/tools/plottr.py` updated to plot `DO_NOTHING` and `DELAY` as spans.

## Notes / Tradeoffs

- Performance:
  - Streaming (stepwise) rollout is substantially slower than single-pass because it evaluates the policy one window at a time.
  - The branch mitigates this in defences with early stopping and inference-only rollouts, but the core stepwise policy eval remains the main cost.

- TAM timeline:
  - With half-open coverage and float rounding, it is common to produce an action exactly at the last covered time; this is now treated as benign.

## Next Simplification Targets

- Reduce duplication between the “full” streaming rollout and the “trace-only” streaming rollout in `kipl_ml/rl/simulate.py`.
- Consolidate rollout selection logic to keep training/defence code paths as small as possible.
- Keep delay semantics in exactly one place (execution + streaming cursor) and ensure tests/scripts cover the invariant.
