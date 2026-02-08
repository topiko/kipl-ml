#/bin/bash
#
#
# Exit if anything fails..
set -e

adv=mc

for per in 100 300
do
	for sep_critic in true false
	do
		for padding_scale in 0.001 0.002 0.005
		do
			for d_clf_scale in 0.0 0.01 0.05
			do
				uv run python obs_agent_01.py \
					advantages.type=$adv  \
					h_detach_period=$per \
					max_epochs=120 \
					league_size=40 \
					train_subset_frac=0.2 \
					separate_critic=$sep_critic \
					padding_scale=$padding_scale \
					rewards.d_clf=$d_clf_scale \
					experiment_name=obsrl_sweep2
			done
		done
	done
done
