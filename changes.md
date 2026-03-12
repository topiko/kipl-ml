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
- Optional long-trace blacklist for obsrl dataset loading:
  - exclude by `time_to_<trace_len>_packets > threshold_s`
  - exclusion applies only to traces with `n_packets >= trace_len`
  - report both `% of all traces` and `% of trace_len-capable traces`

## What Changed (High-Level)

### Int Bins Refactor (Core Change)

- **Integer bins throughout internal computation**: All timing operations (TIMES, Dt, DELAY, SEND_*_AFTER_TIME) now use integer bin indices internally, eliminating float32 boundary drift issues.
- **Float seconds only at boundaries**: Float timestamps appear only at:
  - Final output (`X_obs` from `TraceExecState.finalize()`)
  - Model input boundary (via `_feature_map` conversion)
- **Sentinel change**: `-1` (torch.long) replaces `NaN` for invalid/padding positions in TIMES and Dt. Counts use `0` for padding.
- **UP_COUNT and DOWN_COUNT are always integers**: Packet counts are `torch.long`, not float.

### Streaming Window Generation

- Added `WindowFeatureStreamer` which yields one observation window at a time.
- This enables actions (especially `DELAY`) to affect which windows are produced next.
- Streamer operates entirely in int-bin space.

### Stepwise Trace Execution

- Introduced `TraceExecState` to accumulate actions and finalize once, avoiding repeated re-sorts.
- Added `execute_actions_from_sequence` helper for the common pattern of executing a full action sequence.

### New `DELAY` Action

- Added `Actions.DELAY` and expanded `AGENT1(enable_delay=True)` selector head.
- Delay duration is fixed to `obs.time_step`.
- Streaming delay application uses action times (right edge) to align with execution.

### Invariant Hardening (`row2 == row4 - padding`)

- Added strict checks for per-window and per-bin equality between obs features (row2) and defended trace TAM bins (row4 minus padding).
- Added strict no-packets-inside-delay-window checks.
- Added recompute check (`fd` vs `get_window_feature_dict(X_obs - padding)`).
- Wired these checks into real-data debug plotting and daily test suite.

### Delay Penalty

- Optional reward component controlled by `rewards.delay_scale`.
- Penalizes the number of original packets falling in the delayed window bin (computed from `X_raw` + `action_times`).

### Reward Mapping Fixes for TAM

- Action-intervals may legitimately contain zero TAM bins (no longer an error).
- Only fail fast when action times truly exceed disc coverage; tolerate boundary/float noise.

### NN Defence (RNNDef) Integration

- Delay-enabled policies run via streaming rollout.
- Added inference-mode execution paths to reduce overhead.
- Fixed multi-model state-dict handling (`state_dict()` call in list-model path).
- `simul_kwargs` now propagate to rollout calls (`sample`, `extend_end_s`, `max_packets`).
- Fixed DataLoader crash when cached traces contain inference tensors (avoid in-place time shift).

### Debug Tooling

- All obsrl debug scripts moved under `experiment/obsrl/codex_debug/`.
- Added dedicated NNDef action-application check script.
- All debug scripts updated for int bins (use `>= 0` instead of `isfinite()`).

### Dead Code Removal

- Removed `send_exec` and `_infer_time_step_s` from `action.py` (use `TraceExecState` directly).
- Removed `get_action_seq_lens` (inlined at call sites).
- Removed `_apply_delay_clamp_inplace` (float-based version, use `_apply_delay_clamp_bins_inplace`).
- Removed `_fill_after_seq_end`, `_append_values` from `utils.py`.
- Removed `league_rewards2rewards` from `experiment/obsrl/utils.py`.
- Removed `experiment/trace_gan/` directory entirely.
- Removed `kipl_ml/model_eval/obsfuscator.py`.

### Meta-data and Filtering for Runtime Control

- `conversion.py` now writes `time_to_<X>_packets` columns in seconds.
- Obsrl loaders can blacklist long traces at load time via `metadf` mask.
- Sisyphus defaults were switched to use this filter (`trace_len=5000`, `exclude_longer_than_s=30.0`, no min-packet prefilter).

## Files Touched / Added

- `kipl_ml/models/trgen.py`
  - `AGENT1` supports `enable_delay` and outputs `DELAY` with duration `time_step`.
  - Selector mapping updated (`DO_NOTHING` replaces `WAIT` with alias kept for compat).
  - `send_mode='fixed'` is the supported mode for this path; spread mode is deprecated.
  - Action times are quantized to the `time_step` grid (int bins).
  - `_feature_map` converts int bins to float seconds for model input.

- `kipl_ml/rl/enums.py`
  - Adds `Actions.DELAY`.
  - Adds `Actions.DO_NOTHING` (with compatibility alias for older `WAIT` naming).

- `kipl_ml/rl/action.py`
  - Implements bin-first delay/send execution and stable ordering for equal timestamps.
  - Adds `TraceExecState` to accumulate stepwise actions and finalize once.
  - Adds `execute_actions_from_sequence` helper for executing full action sequences.
  - Removed `send_exec` and `_infer_time_step_s` (use `TraceExecState` directly).
  - Removed `_apply_delay_clamp_inplace` (float-based, use `_apply_delay_clamp_bins_inplace`).

- `kipl_ml/rl/observation.py`
  - Adds `WindowFeatureStreamer` (operates entirely in int-bin space).
  - `get_window_feature_dict` emits int bins for TIMES/Dt with `-1` sentinel.
  - `_TraceWindowCursor` operates in int-bin space.
  - UP_COUNT and DOWN_COUNT are `torch.long` (packet counts are always integers).
  - Delay-aware cursor logic so future windows reflect applied delays.
  - Adds optional `Feats.WINDOW_BINS` output for dt-aligned logic.

- `kipl_ml/rl/utils.py`
  - Consolidated bin conversion functions: `_time_to_bin_idx`, `_boundary_time_to_bin_idx`, `_duration_to_bin_offsets`.
  - Removed `_fill_after_seq_end` and `_append_values` (unused).

- `kipl_ml/rl/simulate.py`
  - Shared rollout utilities used by training and defences:
    - `policy_rollout_single_pass(...)`
    - `policy_rollout_streaming(...)` (delay-aware)
    - Lightweight trace-only helpers for inference/defences.
  - Runs the window streamer on CPU when `X` is on CUDA to avoid per-step GPU sync.
  - Fixed: record policy before max_packets check to handle `base_n >= max_packets` edge case.

- `experiment/obsrl/sim.py`
  - Training rollout selects single-pass vs streaming based on `obs.enable_delay`.
  - TAM reward mapping hardened:
    - fail-fast if truly beyond coverage
    - tolerate boundary/rounding overhang
    - reduce warning spam for ~0 overhang
  - Uses integer TAM bins for reward mapping (`Feats.TAM_BINS`) to avoid float boundary/spacing issues.
  - Uses boundary binning for action boundaries in reward mapping to align with execution semantics.
  - Standardizes executed trace naming as `X_obs`.

- `experiment/obsrl/utils.py`
  - Moved `dl_` from `experiment/trace_gan/data_utils.py` (now deleted).
  - Removed `league_rewards2rewards` (unused).

- `experiment/obsrl/sisyphus.py`, `experiment/obsrl/obs_agent_01.py`
  - Assert TAM discriminator bin width matches `obs.time_step_s`.
  - Add `Feats.TAM_BINS` to disc feature transforms so reward mapping can use it.
  - Pass long-trace exclusion args into dataset loading (`exclude_time_to_packets_n/s`).

- `experiment/obsrl/config/sisyphus.yaml`
  - Defaults updated for runtime filtering experiments:
    - `trace_len: 5000`
    - `min_packets_in_trace: 0`
    - `exclude_longer_than_s: 30.0`
  - Added `disc.train_defence_aug` for trace caching during discriminator training.

- `experiment/obsrl/config/config.yaml`
  - Adds `exclude_longer_than_s` parameter (default `null`).
  - Sets `min_packets_in_trace: 0`.

- `experiment/obsrl/sisysweep.sh`
  - Sweep packet budget updated to `npackets=5000`.

- `kipl_ml/data/conversion.py`
  - Adds/updates `time_to_<X>_packets` metadata columns (`X = 100, 500, 1000..20000`) in **seconds**.
  - For traces with fewer than `X` packets, stores total trace duration (seconds).

- `kipl_ml/data/wf_dataset.py`
  - Adds optional metadata-driven long-trace filter to `get_train_valid_test(...)`.
  - Adds `wipe_cache()` method for clearing trace cache.
  - Filtering is implemented as simple pandas masking and reports:
    - excluded `% of all`,
    - excluded `% among traces with n_packets >= trace_len`.

- `kipl_ml/trace/enums.py`
  - Adds `Feats.TAM_BINS`.

- `kipl_ml/trace/features.py`
  - TAM transforms use an integer bin grid; adds `TAM_BINS` transform.

- `kipl_ml/defences/nndefs.py`
  - `RNNDef` uses streaming rollout for delay-enabled models.
  - Uses inference-mode rollout helpers.
  - Fixes list-model state-dict loading and forwards `simul_kwargs` to rollout helpers.

- `experiment/obsrl/invariants.py`
  - Shared invariant checks and formatted reports for:
    - row2/row4 equality,
    - fd recomputation equality,
    - delay-window packet leaks.
  - Updated for int bins (use `>= 0` instead of `isfinite()`).

- `experiment/obsrl/daily_test.sh`
  - Updated to call `codex_debug` scripts.
  - Includes comprehensive row2/row4 checks and NNDef action-application checks.

- Debug / verification scripts (all updated for int bins):
  - `experiment/obsrl/codex_debug/debug_window_streamer_equiv.py`: streamer vs precomputed windows.
  - `experiment/obsrl/codex_debug/debug_rollout_equiv.py`: single-pass vs streaming equivalence, timing, plotting, forced selector patterns.
  - `experiment/obsrl/codex_debug/debug_delay_exec_semantics.py`: minimal delay execution regression checks.
  - `experiment/obsrl/codex_debug/debug_row2_row4_checks.py`: comprehensive row2/row4 and delay-window invariants.
  - `experiment/obsrl/codex_debug/debug_plot_real_defended.py`: real-trace strict invariant check + plot (rows 1/2/4 TAM-like).
  - `experiment/obsrl/codex_debug/debug_nndef_applies_actions.py`: validates that `RNNDef.simulate(...)` applies forced policy actions.
  - `experiment/obsrl/codex_debug/debug_plot_smoke.py`: smoke fixture kept invariant-consistent.
  - `experiment/obsrl/codex_debug/debug_delay_no_packets_real.py`: verifies no packets observed during forced delay.

- Plotting:
  - `kipl_ml/tools/plottr.py` updated to plot `DO_NOTHING` and `DELAY` as spans.
  - `kipl_ml/tools/plottr.py::plot_obs_features` now includes row-2 packet count summary box.

- Deleted:
  - `experiment/trace_gan/` (entire directory, unused).
  - `kipl_ml/model_eval/obsfuscator.py` (only used by deleted trace_gan scripts).

## Notes / Tradeoffs

- Performance:
  - Streaming (stepwise) rollout is substantially slower than single-pass because it evaluates the policy one window at a time.
  - The branch mitigates this in defences with early stopping and inference-only rollouts, but the core stepwise policy eval remains the main cost.
  - Runtime experiments are ongoing with metadata-driven trace blacklisting (`time_to_5000_packets > 30s`) to reduce pathological long-silence traces.

- Data conversion dependency:
  - The long-trace blacklist relies on `time_to_<X>_packets` columns in `metadf.h5`.
  - After pulling conversion changes, re-run `python kipl_ml/data/conversion.py --dataset <dataset>` before enabling the filter.

- Int bins semantics:
  - All internal timing uses integer bin indices. Float seconds only appear at boundaries (X_obs output, model input).
  - Sentinel for invalid TIMES/Dt is `-1` (torch.long), not NaN.
  - UP_COUNT and DOWN_COUNT are always integers (packet counts).
  - This eliminates float32 boundary drift issues that caused row2/row4 mismatches.

- Boundary timestamps:
  - Exact float equality on right-edge synthetic timestamps is no longer a reliable correctness criterion.
  - Bin-level invariants are the source of truth.

## Completed Simplifications

- Removed `send_exec` and `_infer_time_step_s` (use `TraceExecState` directly).
- Removed `get_action_seq_lens` (inlined at call sites).
- Removed `_apply_delay_clamp_inplace` (float-based version).
- Removed `_fill_after_seq_end` and `_append_values` from utils.py.
- Removed `league_rewards2rewards` from experiment/obsrl/utils.py.
- Removed `experiment/trace_gan/` directory entirely.
- Removed `kipl_ml/model_eval/obsfuscator.py`.
- Consolidated bin conversion functions in `kipl_ml/rl/utils.py`.
- Added `execute_actions_from_sequence` helper to reduce code duplication.
- All debug scripts updated for int bins.

## Next Steps

- Review test coverage in `experiment/obsrl/` for blind spots.
- Revise unittest scheme (`run_test.sh`) for consistency.
