#!/bin/bash

nepochs=30
defaug=1
expr_name="Ephemeral-Aug$defaug-v4"


common="misc.defence_augmentation=$defaug train.n_epochs=$nepochs mlflow.experiment_name=$expr_name"
common_lb="$common lr_scheduler.epochs=$nepochs"

for defence in no_defence front interspace breakpad
do
	uv run python main.py --config-name=df defence=$defence $common
	uv run python main.py --config-name=rf defence=$defence $common
	uv run python main.py --config-name=df-multi defence=$defence $common_lb
	uv run python main.py --config-name=laserbeak defence=$defence $common_lb

	uv run python xv_table.py -en $expr_name
done


for nmachines in 1 10 100 1000 10000
do
	uv run python main.py --config-name=df defence=maybenot defence.n_machines=$nmachines $common
	uv run python main.py --config-name=rf defence=maybenot defence.n_machines=$nmachines $common
	uv run python main.py --config-name=df-multi defence=maybenot defence.n_machines=$nmachines $common_lb
	uv run python main.py --config-name=laserbeak defence=maybenot defence.n_machines=$nmachines $common_lb

	uv run python xv_table.py -en $expr_name
done
