#!/bin/bash

defaug=0
patience=32
lr_patience=8
fixed_per_trace=false

source defence_lists.sh


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
		expr_name="EphemeralOFFICIAL-inftrain-$netwk_state"


		common="train.defence_augmentation=$defaug lr_scheduler=plateau lr_scheduler.lr_patience=$lr_patience misc.mlflow.experiment_name=$expr_name network=$netwk_state train.patience=$patience train.n_epochs=0"

		uv run python main.py --config-name=df defence=$defence $common
		uv run python main.py --config-name=rf defence=$defence $common
		uv run python main.py --config-name=df-multi defence=$defence $common
		# uv run python main.py --config-name=laserbeak defence=$defence $common

	done
done



