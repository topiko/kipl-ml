#!/bin/bash
#
#
# Exit if anything fails..
set -e

parent_prefix="MORE-audacious-imp-02" #"advanced-smolin" # audacious-imp


for i in $(seq 1 100); do
	parent_name="$parent_prefix"
	echo "=== $parent_name ==="
	echo "=== Push $i/100 ==="
	uv run python sisyphus.py \
		mlflow.parent_run_name=$parent_name \
		experiment_name=sisy_sweep
done
