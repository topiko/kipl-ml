#!/bin/bash

ntwork="infinite"
expr_name="Ephemeral-Aug-Curve-$ntwork"
min_patience=4

div() {
  if [[ $2 -eq 0 ]]; then
    echo "$1"
  else
    result=$(( $1 / $2 ))
    if [[ $result -lt $3 ]]; then
      echo "$3"
    else
      echo "$result"
    fi
  fi
}


for model in march rf df df-multi
do
	for fixed_per_trace in False # True
	do
		if [ "$fixed_per_trace" = True ]; then
			fixing_="fixed_per_trace"
		else
			fixing_="no_fixing"
		fi

		for aug in 0 1 2 4 8 16 32
		do
			patience=$(div 32 $aug $min_patience)
			lr_patience=$(div $patience 4 1)
			common="train.defence_augmentation=$aug lr_scheduler=plateau lr_scheduler.lr_patience=$lr_patience network=$ntwork train.patience=$patience dataset.test_splits=[1,2,3] train.n_epochs=0"


			uv run python main.py --config-name=$model defence=no_defence $common  misc.mlflow.experiment_name="$expr_name-no_fixing" misc.ignore_existing=False

			for defence in "front-$ntwork" "ephemeral-$ntwork" #
			do
				front_ephemeral="$common defence.fixed_per_trace=$fixed_per_trace misc.mlflow.experiment_name=$expr_name-$fixing_ defence.n_train_machines=50000"
				uv run python main.py --config-name=$model defence=$defence $front_ephemeral
			done
		done
	done
done


