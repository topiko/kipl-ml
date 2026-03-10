#!/bin/bash
#
#
# Exit if anything fails..
set -e

adv=mc
parent_prefix="audacious-imp" #"advanced-smolin" # audacious-imp
npackets=1000


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
					trace_len=$npackets \
					experiment_name=sisy_sweep
			done
		done
	done
done
