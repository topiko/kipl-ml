#/bin/bash
#
#
# Exit if anything fails..
set -e

adv=mc
trace_len=6000


for reuse in True False
do
	for sep_critic in true false
	do
		uv run python sisyphus.py \
			advantages.type=$adv  \
			trace_len=$trace_len \
			obs.separate_critic=$sep_critic \
			rewards.padding_scale=$padding_scale \
			rewards.d_clf=$d_clf_scale \
			obs.reuse_obs_and_critic=$reuse \
			experiment_name=sisy_sweep
	done
done
