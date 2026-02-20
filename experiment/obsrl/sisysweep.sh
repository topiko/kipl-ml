#/bin/bash
#
#
# Exit if anything fails..
set -e

adv=mc

for sep_critic in True False; do
	for reuse in True False; do
		for i in $(seq 1 100); do
			parent_name="sep-critic_$sep_critic|reuse_$reuse"
			echo "=== $parent_name ==="
			echo "=== Push $i/100 ==="
			uv run python sisyphus.py \
				mlflow.parent_run_name=$parent_name \
				advantages.type=$adv  \
				obs.separate_critic=$sep_critic \
				obs.reuse_obs_and_critic=$reuse \
				experiment_name=sisy_sweep
		done
	done
done
