#!/bin/bash

source scripts/defense_lists.sh
defaug=0
patience=32
lr_patience=8
fixed_per_trace=false
dataset="bigenough"
netwk_state="infinite"

if [ -n "$1" ]; then
    dataset="$1"
fi


for xv in "[0,1]" "[2]" "[3]" "[4]"
do


	if [[ $netwk_state == "infinite" ]]; then
		list_="ephemeral_inf"
	elif [[ $netwk_state == "bottleneck" ]]; then
		list_="ephemeral_bottle"
	fi

	eval "lst=(\"\${${list_}[@]}\")"


	for defence in "${lst[@]}"
	do

		for nmachines in 1 5 20 100 1000 10000 50000 100000
		do

			echo $defence
			# expr_name="Ephemeral-$dataset-inftrain-$netwk_state"
			expr_name="Ephemeral-$dataset-machine_count-inftrain-$netwk_state"

			common=("train.defence_augmentation=$defaug"
				"lr_scheduler=plateau"
				"lr_scheduler.lr_patience=$lr_patience"
				"misc.mlflow.experiment_name=$expr_name"
				"network=$netwk_state"
				"train.patience=$patience"
				"train.n_epochs=0"
				"dataset.test_splits=$xv"
				"defence.n_train_machines=$nmachines"
				"misc.ignore_existing=True"
			)

			# uv run python main.py --config-name=df defence=$defence "${$common[@]}"
			uv run python main.py --config-name=df-multi defence=$defence "${common[@]}"
		done
	done
done



