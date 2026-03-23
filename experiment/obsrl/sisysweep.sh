#!/bin/bash
#
#
# Exit if anything fails..
set -e

adv=mc
trace_len=5000
parent_prefix="audacious-imp4-$trace_len" #"advanced-smolin" # audacious-imp


for enable_delay in true false; do
	for sep_critic in false true; do
		for reuse in true false; do
			for i in $(seq 1 100); do
				parent_name="$parent_prefix|delay_$enable_delay|sep-critic_$sep_critic|reuse_$reuse"
				echo "=== $parent_name ==="
				echo "=== Push $i/100 ==="
				uv run python sisyphus.py \
					mlflow.parent_run_name=$parent_name \
					advantages.type=$adv  \
					obs.enable_delay=$enable_delay \
					obs.separate_critic=$sep_critic \
					obs.reuse_obs_and_critic=$reuse \
					trace.len=$trace_len \
					experiment_name=sisy_sweep
			done
		done
	done
done
