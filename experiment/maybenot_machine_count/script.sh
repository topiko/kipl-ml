#!/bin/bash

ntrain_traces=7000

python main.py --config-name=config \
	defences.n_machines=0 \
	defences.max_padding_frac=0 \
	dataset.n_train_traces=$ntrain_traces \
	dataset.defence_augmentation=0 \
	load_base_model=True


for defence_aug in 1 2 4 8 16 0
do
	for nmachines in 1 10 100 1000 8000
	do
		echo "Running with $nmachines machines"
		python main.py --config-name=config \
			defences.n_machines=$nmachines \
			dataset.n_train_traces=$ntrain_traces \
			dataset.defence_augmentation=$defence_aug \
			load_base_model=True
	done
done
