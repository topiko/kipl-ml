#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EPHEMERAL_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$EPHEMERAL_DIR"

source scripts/defense_lists.sh
nepochs=30
dataset="bigenough"
defaug=1


if [ -n "$1" ]; then
    dataset="$1"
fi

# Optional: name of the run-specific defence folder under config/defence/
# Usage:
#   ./scripts/obs_rl.sh [dataset] [run_name]
rlobs_run_name="${2:-${RLOBS_RUN_NAME:-}}"


netwk_state="infinite"


defences=(front-infinite no_defence ephemeral-pad-inf-sc0.75)

if [ -n "$rlobs_run_name" ]; then
    rl_dir="$EPHEMERAL_DIR/config/defence/$rlobs_run_name"

    if [ ! -d "$rl_dir" ]; then
        echo "Missing defence folder: $rl_dir" >&2
        echo "Expected: experiment/ephemeral_defences/config/defence/$rlobs_run_name/*.yaml" >&2
        exit 1
    fi

    # Collect configs from the folder, sorting numerically by trailing -<step> in filename.
    mapfile -t rlobs_cfgs < <(
        for f in "$rl_dir"/*.yaml; do
            [ -f "$f" ] || continue
            stem="$(basename "$f" .yaml)"
            step="${stem##*-}"
            if [[ "$step" =~ ^[0-9]+$ ]]; then
                printf '%s\t%s\n' "$step" "$stem"
            else
                printf '%s\t%s\n' "999999" "$stem"
            fi
        done | sort -n -k1,1 | cut -f2
    )

    if [ ${#rlobs_cfgs[@]} -eq 0 ]; then
        echo "No configs found for run_name='$rlobs_run_name' under $rl_dir" >&2
        exit 1
    fi

    # Hydra config group is `defence`, so subdir selections use `defence=<folder>/<name>`.
    prefixed_defences=()
    for stem in "${rlobs_cfgs[@]}"; do
        prefixed_defences+=("${rlobs_run_name}/${stem}")
    done

    defences=("${prefixed_defences[@]}" "${defences[@]}")
fi

for defence in "${defences[@]}"
do
	echo $defence
	expr_name="RLobs-$dataset-aug$defaug-$netwk_state"

	common="train.defence_augmentation=$defaug train=fixed-epochs train.n_epochs=$nepochs misc.mlflow.experiment_name=$expr_name network=$netwk_state dataset=$dataset"
	common_lb="$common lr_scheduler.epochs=$nepochs"

	uv run python main.py --config-name=df defence=$defence $common
	uv run python main.py --config-name=rf defence=$defence $common
	uv run python main.py --config-name=df-multi defence=$defence $common_lb
	#uv run python main.py --config-name=laserbeak_wo_attention defence=$defence $common_lb
	#uv run python main.py --config-name=rf_star defence=$defence $common
	#uv run python main.py --config-name=laserbeak defence=$defence $common_lb

done
