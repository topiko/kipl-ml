#!/bin/bash

defaug=0
patience=32
lr_patience=8
fixed_per_trace=false


defences=(ephemeral-block-bottle-default ephemeral-block-bottle-padding-heavy ephemeral-block-bottle-blocking-heavy)


netwk_state=bottleneck
events_mltp=16




for defence in $defences
do

	if [ "$defence" == "ephemeral-block-bottle-padding-heavy" ]; then
		max_trace_length=300000
	else
		max_trace_length=200000
	fi

	for xv in "[0]" "[1]" "[2]" "[3]" "[4]"
	do
		echo $defence

		for sc in 0.95 0.8 0.75 0.5 0.25 0.99 1.0

		do
			expr_name="Eph-COSTCURVE-$netwk_state"


			common=("train.defence_augmentation=$defaug"
				"lr_scheduler=plateau"
				"lr_scheduler.lr_patience=$lr_patience"
				"misc.mlflow.experiment_name=$expr_name"
				"network=$netwk_state"
				"train.patience=$patience"
				"train.n_epochs=0"
				"dataset.test_splits=$xv"
				"defence.scale=$sc"
				"misc.simul_args.events_multiplier=$events_mltp"
				"misc.simul_args.max_trace_length=$max_trace_length"
			)

			# uv run python main.py --config-name=df defence=$defence $common
			# uv run python main.py --config-name=rf defence=$defence $common
			uv run python main.py --config-name=df-multi defence=$defence "${common[@]}"
			# uv run python main.py --config-name=laserbeak defence=$defence $common
		done
	done
done



