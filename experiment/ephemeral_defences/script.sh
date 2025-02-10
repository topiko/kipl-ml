#!/bin/bash



for defence_aug in 1 2 4 8 16 0
do
	uv run python main.py --config-name=config defence=no_defence dataset.defence_augmentation=$defence_aug
	for nmachines in 1 10 100 1000 5000 15000
	do
		echo "Running with $nmachines machines"
		uv run python main.py --config-name=config \
			defence=maybenot \
			defence.n_machines=$nmachines \
			dataset.defence_augmentation=$defence_aug
	done
done
