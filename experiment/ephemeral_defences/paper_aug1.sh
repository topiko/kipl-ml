#!/bin/bash

nepochs=60
defaug=1
scheduler=cosine
expr_name="Ephermeral-Aug$defaug-$scheduler-v3"

for model in df-multi df-dirs-only laserbeak
do
	common="dataset.defence_augmentation=$defaug train=$scheduler train.epochs=$nepochs train.n_epochs=$nepochs mlflow.experiment_name=$expr_name model=$model"

	for defence in no_defence front interspace # breakpad
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
