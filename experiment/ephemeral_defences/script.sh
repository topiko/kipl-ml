#!/bin/bash


uv run python main.py --config-name=config defence=no_defence dataset.defence_augmentation=6 train.epochs=50

for nmachines in 1 10 100 1000 10000
do
	for defence_aug in 0 1 4 8 16
	do
		echo "Running with $nmachines machines"
		uv run python main.py --config-name=config \
			defence=maybenot \
			defence.n_machines=$nmachines \
			dataset.defence_augmentation=$defence_aug
	done
done
