#!/bin/bash
#
#
# Exit if anything fails..
set -e

parent_prefix="MORE-audacious-imp-02" #"advanced-smolin" # audacious-imp


for div_z in false true; do
	for i in $(seq 1 100); do
		parent_name="$parent_prefix|divz-$div_z"
		echo "=== $parent_name ==="
		echo "=== Push $i/100 ==="
		uv run python sisyphus.py \
			mlflow.parent_run_name=$parent_name \
			advantages.divide_by_Z=$div_z \
			experiment_name=sisy_sweep
	done
done
