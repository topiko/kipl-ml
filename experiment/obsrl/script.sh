#/bin/bash
#
#


for per in 100 200 50
do
	for rnn_lr_reduction in  1.0 0.5
	do
		for adv in mc mc_w_bootstrap
		do
			uv run python obs_agent_01.py \
				advantages.type=$adv  \
				h_detach_period=$per \
				rnn_lr_reduction=$rnn_lr_reduction \
				max_epochs=100 \
				league_size=50 \
				train_subset_frac=0.2 \
				experiment_name=obsrl_sweep2
		done
	done
done
