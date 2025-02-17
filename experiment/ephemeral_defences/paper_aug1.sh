#!/bin/bash

expr_name="Ephermeral-Aug1"
nepochs=30
defaug=1

for model in df-dirs-only df-tiktok df-multi
do
	common="dataset.defence_augmentation=$defaug train.n_epochs=$nepochs mlflow.experiment_name=$expr_name model.name=$model"

	for defence in front no_defence interspace breakpad
	do
		uv run python main.py --config-name=config defence=$defence $common
		uv run python xv_table.py -en $expr_name
	done


	for nmachines in 1 10 100 1000 10000
	do
		uv run python main.py --config-name=config defence=maybenot defence.n_machines=$nmachines $common
		uv run python xv_table.py -en $expr_name
	done
done
