#!/bin/bash
# Daily test runner for kipl-ml
# Run via cron: @daily /home/topiko/Playground/KIPL/projects/kipl-ml/daily_test.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"

DATE=$(date +%Y%m%d)
LOG_FILE="$LOG_DIR/daily_test_$DATE.log"

echo "=== Daily test run: $(date) ===" | tee "$LOG_FILE"
echo | tee -a "$LOG_FILE"

# Run unit tests
echo "=== Unit tests ===" | tee -a "$LOG_FILE"
cd "$SCRIPT_DIR"
./run_tests.sh 2>&1 | tee -a "$LOG_FILE"

echo | tee -a "$LOG_FILE"

# Run obsrl daily tests (quick mode)
echo "=== OBSRL daily tests (quick) ===" | tee -a "$LOG_FILE"
SKIP_REAL_DATA=1 SKIP_REAL_PLOT=1 ROW24_N=3 \
  bash experiment/obsrl/daily_test.sh 2>&1 | tee -a "$LOG_FILE"

echo | tee -a "$LOG_FILE"
echo "=== All tests completed: $(date) ===" | tee -a "$LOG_FILE"
