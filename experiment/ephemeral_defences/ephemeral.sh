#!/bin/bash

nepochs=30
defaug=1


for netwk_state in infinite bottleneck
do
	for defence in "ephemeral-$netwk_state" # "front-$netwk_state" interspace breakpad "tamaraw-$netwk_state" no_defence
	do
		expr_name="EphemeralAll-Aug$defaug-$netwk_state"
		common="misc.defence_augmentation=$defaug train=fixed-epochs train.n_epochs=$nepochs mlflow.experiment_name=$expr_name network=$netwk_state"
		common_lb="$common lr_scheduler.epochs=$nepochs"

		# uv run python main.py --config-name=df defence=$defence $common
		# uv run python main.py --config-name=rf defence=$defence $common
		# uv run python main.py --config-name=df-multi defence=$defence $common_lb
		uv run python main.py --config-name=laserbeak defence=$defence $common_lb

	done
done



