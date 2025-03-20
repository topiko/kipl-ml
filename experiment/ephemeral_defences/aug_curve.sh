#!/bin/bash

nepochs=30
expr_name="Ephemeral-Aug-Curve"
ntwork="infinite"


for model in df df-multi rf
do
	for fixed_per_trace in False True
	do
		if [ "$fixed_per_trace" = True ]; then
			fixing_="fixed_per_trace"
		else
			fixing_="no_fixing"
		fi

		for aug in 0 1 2 4 8 16 32
		do
			common="train.defence_augmentation=$aug lr_scheduler=plateau train.n_epochs=$nepochs network=$ntwork"

			uv run python main.py --config-name=$model defence=no_defence $common  misc.mlflow.experiment_name="$expr_name-no_fixing" misc.ignore_existing=False

			for defence in "front-$ntwork" # "ephemeral-$ntwork"
			do
				front_ephemeral="$common defence.fixed_per_trace=$fixed_per_trace misc.mlflow.experiment_name=$expr_name-$fixing_"
				uv run python main.py --config-name=$model defence=$defence $front_ephemeral
			done
		done
	done
done


