#!/bin/bash

nepochs=30
expr_name="Ephemeral-Aug-Curve"
ntwork="bottleneck"


for model in df-multi df rf
do
	for aug in 1 2 4 6 8 12 0
	do
		for fixed_per_trace in False True
		do
			common="misc.defence_augmentation=$aug lr_scheduler=plateau train.n_epochs=$nepochs mlflow.experiment_name=$expr_name network=$ntwork ignore_existing=True"
	        	front_ephemeral="$common defence.fixed_per_trace=$fixed_per_trace"

			uv run python main.py --config-name=$model defence=no_defence $common
			for defence in "front-$ntwork" "ephemeral-$ntwork"
			do
				uv run python main.py --config-name=$model defence=$defence $front_ephemeral
			done
		done
	done
done


