#/bin/bash
#
#
# Exit if anything fails..
set -e

adv=mc


for reuse in True False
do
	for sep_critic in false true
	do
		uv run python sisyphus.py \
			advantages.type=$adv  \
			obs.separate_critic=$sep_critic \
			obs.reuse_obs_and_critic=$reuse \
			experiment_name=sisy_sweep
	done
done
