#!/bin/bash

max_padding_frac=0.5

for defence_aug in 0 1 2 4 8 16
do
	for nmachines in 0 1 10 100 1000 8000
	do
		echo "Running with $nmachines machines"
		python main.py --config-name=config defences.n_machines=$nmachines defences.max_padding_frac=$max_padding_frac dataset.n_train_traces=7000 load_base_model=True dataset.defence_aug=$defence_aug
	done
done
