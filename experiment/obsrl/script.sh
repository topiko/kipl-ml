#/bin/bash
#
#
# Exit if anything fails..
set -e

adv=mc
trace_len=10000
figs_period=1

for per in 100 300
do
	for sep_critic in true false
	do
		for padding_scale in 0.005 0.01 0.05
		do
			for d_clf_scale in 0.0 0.01 0.05
			do
				uv run python obs_agent_01.py \
					advantages.type=$adv  \
					h_detach_period=$per \
					max_epochs=120 \
					trace_len=$trace_len \
					league_size=40 \
					train_subset_frac=0.2 \
					separate_critic=$sep_critic \
					padding_scale=$padding_scale \
					figs_period=$figs_period \
					rewards.d_clf=$d_clf_scale \
					experiment_name=obsrl_sweep2
			done
		done
	done
done
