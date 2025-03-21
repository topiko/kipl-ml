#!/bin/bash

ntwork="infinite"
expr_name="Ephemeral-Aug-Curve-$ntwork"

div() {
  if [[ $2 -eq 0 ]]; then
    echo "$1"
  else
    echo $(( $1 / $2 ))
  fi
}


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
			patience=$(div 32 $aug)
			common="train.defence_augmentation=$aug lr_scheduler=plateau network=$ntwork train.patience=$patience train.test_splits=[1,2]"

			uv run python main.py --config-name=$model defence=no_defence $common  misc.mlflow.experiment_name="$expr_name-no_fixing" misc.ignore_existing=False

			for defence in "front-$ntwork" #
			do
				front_ephemeral="$common defence.fixed_per_trace=$fixed_per_trace misc.mlflow.experiment_name=$expr_name-$fixing_"
				uv run python main.py --config-name=$model defence=$defence $front_ephemeral
			done
		done
	done
done


