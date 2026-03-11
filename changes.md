# feature/delay vs dev

This branch adds an interactive (stepwise) obfuscation mode for obsrl so the policy can affect *future* observation windows, and introduces a new exclusive `DELAY` action.

The goal is correctness first (equivalence with the old single-pass pipeline when delay is disabled), then usability (debug plots + forced action patterns), then enough performance to train and sweep configs.

## Desired Behavior

- `DELAY` is a new selector option.
- `DELAY` is exclusive: it cannot co-occur with any send action in the same step.
- Delay duration is fixed to one model time-step (`obs.time_step_s`, e.g. 0.02s). No separate delay-time head.
- Action times are on the model grid (`k * time_step_s`) and internal execution uses integer bins.
- `send_mode='spread'` is deprecated for obsrl delay training; fixed-offset send mode is used.
- Right-edge semantics: actions chosen from window `[t0, t0+dt)` take effect starting at `t1 = t0+dt`.
- Delay semantics: applying delay at action time `t1` blocks the window `[t1, t1+dt)`.
  - Packets (including padding packets) that would occur inside that interval are clamped to `t1+dt`.
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
  - Streaming delay application uses action times (right edge) to align with `send_exec`.

- Bin-first timing for execution:
  - Delay starts/shifts and send-after offsets are carried as integer bins in execution paths.
  - Action boundaries use boundary binning (`round(t/dt)`), while packet membership uses floor binning.
  - Right-edge synthetic timestamps are nudged to the next representable float to avoid boundary drift into the previous bin.

- Invariant hardening (`row2 == row4 - padding`):
  - Added strict checks for per-window and per-bin equality between obs features (row2) and defended trace TAM bins (row4 minus padding).
  - Added strict no-packets-inside-delay-window checks.
  - Added recompute check (`fd` vs `get_window_feature_dict(X_obs - padding)`).
  - Wired these checks into real-data debug plotting and daily test suite.

- Delay penalty:
  - Optional reward component controlled by `rewards.delay_scale`.
  - Penalizes the number of original packets falling in the delayed window bin (computed from `X_raw` + `action_times`).

- Reward mapping fixes for TAM:
  - Action-intervals may legitimately contain zero TAM bins (no longer an error).
  - Only fail fast when action times truly exceed disc coverage; tolerate boundary/float noise.

- TAM time refactor (root fix):
  - Added integer TAM bins (`Feats.TAM_BINS`) so reward mapping and timeline logic
    can operate in bin-space rather than float seconds.
  - `TAM_TIMES` is now derived from integer bin indices * `window_width_s` to avoid
    float32 spacing jitter.
  - TAM reward mapping uses `TAM_BINS` + `window_width_s` (with a backward-compat
    fallback to derive bins from `TAM_TIMES` if needed).

- TIMES padding correctness:
  - Stopped using `0` as a TIMES padding sentinel.
  - Use mask-based filling for TIMES and keep strict sentinel checks only where it is actually a sentinel.

- NN defence (RNNDef) integration:
  - Delay-enabled policies run via streaming rollout.
  - Added inference-mode execution paths to reduce overhead.
  - Fixed multi-model state-dict handling (`state_dict()` call in list-model path).
  - `simul_kwargs` now propagate to rollout calls (`sample`, `extend_end_s`, `max_packets`).
  - Fixed DataLoader crash when cached traces contain inference tensors (avoid in-place time shift).

- Debug tooling reorg:
  - All obsrl debug scripts moved under `experiment/obsrl/codex_debug/`.
  - Added dedicated NNDef action-application check script.

## Files Touched / Added

- `kipl_ml/models/trgen.py`
  - `AGENT1` supports `enable_delay` and outputs `DELAY` with duration `time_step`.
  - Selector mapping updated (`DO_NOTHING` replaces `WAIT` with alias kept for compat).
  - `send_mode='fixed'` is the supported mode for this path; spread mode is deprecated.
  - Action times are quantized to the `time_step` grid.

- `kipl_ml/rl/enums.py`
  - Adds `Actions.DELAY`.
  - Adds `Actions.DO_NOTHING` (with compatibility alias for older `WAIT` naming).

- `kipl_ml/rl/action.py`
  - Implements bin-first delay/send execution and stable ordering for equal timestamps.
  - Adds `TraceExecState` to accumulate stepwise actions and finalize once.
  - `send_exec` clamps padding packets too when a delay window covers them.
  - Right-edge synthetic timestamps are nudged (`nextafter(..., +inf)`) to preserve intended bin under floor mapping.

- `kipl_ml/rl/observation.py`
  - Adds `WindowFeatureStreamer`.
  - Delay-aware cursor logic so future windows reflect applied delays.
  - Adds optional `Feats.WINDOW_BINS` output for dt-aligned logic.

- `kipl_ml/rl/simulate.py` (new)
  - Shared rollout utilities used by training and defences:
    - `policy_rollout_single_pass(...)`
    - `policy_rollout_streaming(...)` (delay-aware)
    - Lightweight trace-only helpers for inference/defences.
  - Runs the window streamer on CPU when `X` is on CUDA to avoid per-step GPU sync.

- `experiment/obsrl/sim.py`
  - Training rollout selects single-pass vs streaming based on `obs.enable_delay`.
  - TAM reward mapping hardened:
    - fail-fast if truly beyond coverage
    - tolerate boundary/rounding overhang
    - reduce warning spam for ~0 overhang
  - Uses integer TAM bins for reward mapping (`Feats.TAM_BINS`) to avoid float
    boundary/spacing issues.
  - Uses boundary binning for action boundaries in reward mapping to align with execution semantics.
  - Standardizes executed trace naming as `X_obs`.

- `experiment/obsrl/sisyphus.py`, `experiment/obsrl/obs_agent_01.py`
  - Assert TAM discriminator bin width matches `obs.time_step_s`.
  - Add `Feats.TAM_BINS` to disc feature transforms so reward mapping can use it.

- `kipl_ml/trace/enums.py`
  - Adds `Feats.TAM_BINS`.

- `kipl_ml/trace/features.py`
  - TAM transforms use an integer bin grid; adds `TAM_BINS` transform.

- `kipl_ml/defences/nndefs.py`
  - `RNNDef` uses streaming rollout for delay-enabled models.
  - Uses inference-mode rollout helpers.
  - Fixes list-model state-dict loading and forwards `simul_kwargs` to rollout helpers.

- `experiment/obsrl/invariants.py` (new)
  - Shared invariant checks and formatted reports for:
    - row2/row4 equality,
    - fd recomputation equality,
    - delay-window packet leaks.

- `experiment/obsrl/daily_test.sh`
  - Updated to call `codex_debug` scripts.
  - Includes comprehensive row2/row4 checks and NNDef action-application checks.

- Debug / verification scripts:
  - `experiment/obsrl/codex_debug/debug_window_streamer_equiv.py`: streamer vs precomputed windows.
  - `experiment/obsrl/codex_debug/debug_rollout_equiv.py`: single-pass vs streaming equivalence, timing, plotting, forced selector patterns.
  - `experiment/obsrl/codex_debug/debug_delay_exec_semantics.py`: minimal delay execution regression checks.
  - `experiment/obsrl/codex_debug/debug_row2_row4_checks.py`: comprehensive row2/row4 and delay-window invariants.
  - `experiment/obsrl/codex_debug/debug_plot_real_defended.py`: real-trace strict invariant check + plot (rows 1/2/4 TAM-like).
  - `experiment/obsrl/codex_debug/debug_nndef_applies_actions.py`: validates that `RNNDef.simulate(...)` applies forced policy actions.
  - `experiment/obsrl/codex_debug/debug_plot_smoke.py`: smoke fixture kept invariant-consistent.

- Plotting:
  - `kipl_ml/tools/plottr.py` updated to plot `DO_NOTHING` and `DELAY` as spans.
  - `kipl_ml/tools/plottr.py::plot_obs_features` now includes row-2 packet count summary box.

## Notes / Tradeoffs

- Performance:
  - Streaming (stepwise) rollout is substantially slower than single-pass because it evaluates the policy one window at a time.
  - The branch mitigates this in defences with early stopping and inference-only rollouts, but the core stepwise policy eval remains the main cost.

- TAM timeline:
  - With half-open coverage and float rounding, it is common to produce an action exactly at the last covered time; this is now treated as benign.

- Boundary timestamps:
  - Exact float equality on right-edge synthetic timestamps is no longer a reliable correctness criterion.
  - Bin-level invariants are the source of truth.

## Next Simplification Targets

- Reduce duplication between the “full” streaming rollout and the “trace-only” streaming rollout in `kipl_ml/rl/simulate.py`.
- Consolidate rollout selection logic to keep training/defence code paths as small as possible.
- Keep delay semantics in exactly one place (execution + streaming cursor) and ensure tests/scripts cover the invariant.
