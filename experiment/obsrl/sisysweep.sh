#/bin/bash
#
#
# Exit if anything fails..
set -e

adv=mc
trace_len=6000
padding_scale=0.01
d_clf_scale=0.0
reuse_obs_and_critic=True

for per in 100 200
do
	for sep_critic in true false
	do
		uv run python sisyphus.py \
			advantages.type=$adv  \
			obs.detach_period=$per \
			obs.critic_detach_period=500 \
			trace_len=$trace_len \
			league_size=10 \
			obs.separate_critic=$sep_critic \
			rewards.padding_scale=$padding_scale \
			rewards.d_clf=$d_clf_scale \
			obs.reuse_obs_and_critic=$reuse_obs_and_critic \
			experiment_name=sisy_sweep
	done
done
