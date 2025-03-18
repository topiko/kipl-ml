#!/bin/bash

nepochs=30
expr_name="Ephemeral-Aug-Curve"
ntwork="bottleneck"


for fixed_per_trace in False True
do
	for aug in 1 2 4 6 8 12 0
	do
		common="misc.defence_augmentation=$aug lr_scheduler=plateau train.n_epochs=$nepochs mlflow.experiment_name=$expr_name network=$ntwork"
	        front_ephemeral="$common defence.fixed_per_trace=$fixed_per_trace"

		for model in df rf
		do
			uv run python main.py --config-name=model defence=no_defence $common
			for defence in "front-$ntwork" "ephemeral-$ntwork"
			do
				uv run python main.py --config-name=model defence=$defence $front_ephemeral
				# uv run python main.py --config-name=rf defence=$defence $common
			done
		done
	done
done


