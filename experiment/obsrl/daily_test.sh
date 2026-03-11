#!/usr/bin/env bash
set -euo pipefail

# Daily sanity checks for obsrl delay/streaming behavior.
#
# Optional env vars:
#   DEVICE=cpu|cuda
#   RUNS=1
#   WARMUP=0
#   RUNS_REWARDS=1
#   N_DELAYS=50
#   IDX=0
#   SKIP_REAL_DATA=0|1
#   SKIP_REAL_PLOT=0|1

DEVICE="${DEVICE:-cpu}"
RUNS="${RUNS:-1}"
WARMUP="${WARMUP:-0}"
RUNS_REWARDS="${RUNS_REWARDS:-1}"
N_DELAYS="${N_DELAYS:-50}"
IDX="${IDX:-0}"
SKIP_REAL_DATA="${SKIP_REAL_DATA:-0}"
SKIP_REAL_PLOT="${SKIP_REAL_PLOT:-1}"

run() {
  local title="$1"
  shift
  echo
  echo "=== ${title} ==="
  "$@"
}

run "Window streamer equivalence" \
  python -u experiment/obsrl/debug_window_streamer_equiv.py

run "Delay execution semantics" \
  python -u experiment/obsrl/debug_delay_exec_semantics.py

run "Plot smoke" \
  python -u experiment/obsrl/debug_plot_smoke.py

run "Rollout sanity (do_nothing)" \
  python -u experiment/obsrl/debug_rollout_equiv.py \
    --runs "${RUNS}" --warmup "${WARMUP}" --runs_rewards "${RUNS_REWARDS}" \
    --selector_pattern do_nothing

run "Rollout sanity (send_and_delay_cycle)" \
  python -u experiment/obsrl/debug_rollout_equiv.py \
    --runs "${RUNS}" --warmup "${WARMUP}" --runs_rewards "${RUNS_REWARDS}" \
    --selector_pattern send_and_delay_cycle

if [[ "${SKIP_REAL_DATA}" != "1" ]]; then
  run "Real-data forced-delay invariant" \
    python -u experiment/obsrl/debug_delay_no_packets_real.py \
      --device "${DEVICE}" --n_delays "${N_DELAYS}" --idx "${IDX}"
else
  echo
  echo "=== Real-data forced-delay invariant (skipped) ==="
  echo "Set SKIP_REAL_DATA=0 to enable."
fi

if [[ "${SKIP_REAL_PLOT}" != "1" ]]; then
  run "Real-data plotting + exec consistency" \
    python -u experiment/obsrl/debug_plot_real_defended.py \
      --device "${DEVICE}" --idx "${IDX}" --selector_pattern send_and_delay_cycle
else
  echo
  echo "=== Real-data plotting + exec consistency (skipped) ==="
  echo "Set SKIP_REAL_PLOT=0 to enable."
fi

echo
echo "All daily obsrl sanity checks passed."
