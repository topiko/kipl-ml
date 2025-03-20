#!/bin/bash

nepochs=30
expr_name="Ephemeral-Aug-Curve"
ntwork="infinite"


for model in df df-multi rf
do
	for fixed_per_trace in False True
	do
		for aug in 1 2 4 8 16 0
		do
			common="misc.defence_augmentation=$aug lr_scheduler=plateau train.n_epochs=$nepochs mlflow.experiment_name=$expr_name network=$ntwork"

			uv run python main.py --config-name=$model defence=no_defence $common

			for defence in "front-$ntwork" # "ephemeral-$ntwork"
			do
				front_ephemeral="$common defence.fixed_per_trace=$fixed_per_trace ignore_existing=True"
				uv run python main.py --config-name=$model defence=$defence $front_ephemeral
			done
		done
	done
done


