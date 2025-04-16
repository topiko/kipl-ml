#!/bin/bash

source defence_lists.sh
nepochs=30
defaug=1

for netwk_state in infinite bottleneck
do


	if [[ $netwk_state == "infinite" ]]; then
		list_="defences_inf"
	elif [[ $netwk_state == "bottleneck" ]]; then
		list_="defences_bottle"
	fi

	eval "lst=(\"\${${list_}[@]}\")"

	for defence in "${lst[@]}"
	do
		echo $defence
		expr_name="EphemeralMORE-OFFICIAL-Aug$defaug-$netwk_state"
		common="train.defence_augmentation=$defaug train=fixed-epochs train.n_epochs=$nepochs misc.mlflow.experiment_name=$expr_name network=$netwk_state"
		common_lb="$common lr_scheduler.epochs=$nepochs"

		uv run python main.py --config-name=df defence=$defence $common
		# uv run python main.py --config-name=rf defence=$defence $common
		# uv run python main.py --config-name=df-multi defence=$defence $common_lb
		# uv run python main.py --config-name=laserbeak defence=$defence $common_lb

	done
done



