#!/bin/bash

nepochs=30
defaug=0
patience=15
fixed_per_trace=false

for netwk_state in infinite bottleneck
do
	for defence in "ephemeral-$netwk_state" # "front-$netwk_state" interspace breakpad "tamaraw-$netwk_state" no_defence
	do
		expr_name="EphemeralAll-inftrain-$netwk_state"
		common="misc.defence_augmentation=$defaug lr_scheduler=plateau mlflow.experiment_name=$expr_name network=$netwk_state train.patience=$patience"


		uv run python main.py --config-name=df defence=$defence $common
		uv run python main.py --config-name=rf defence=$defence $common
		uv run python main.py --config-name=df-multi defence=$defence $common
		# uv run python main.py --config-name=laserbeak defence=$defence $common

	done
done



