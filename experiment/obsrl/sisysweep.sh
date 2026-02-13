#/bin/bash
#
#
# Exit if anything fails..
set -e

adv=mc

for send_mode in fixed spread
do
	for reuse in True False
	do
		for sep_critic in false true
		do
			uv run python sisyphus.py \
				advantages.type=$adv  \
				obs.separate_critic=$sep_critic \
				obs.reuse_obs_and_critic=$reuse \
				obs.send_mode=$send_mode \
				experiment_name=sisy_sweep_$send_mode
		done
	done
done
