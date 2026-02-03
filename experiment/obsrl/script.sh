#/bin/bash
#
#


for adv in mc_w_bootstrap mc
do
	for per in 50 100 200
	do
		for rnn_lr_reduction in  0.33 0.2 0.5 0.8 1.0
		do
			uv run python obs_agent_01.py advantages.type=$adv  h_detach_period=$per rnn_lr_reduction=$rnn_lr_reduction max_epochs=30 experiment_name=obsrl_sweep
		done
	done
done
