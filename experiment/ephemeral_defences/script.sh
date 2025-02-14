#!/bin/bash


for test_xv in 0 1
do
	uv run python main.py --config-name=config defence=no_defence dataset.defence_augmentation=6 dataset.test_xv=$test_xv

	for nmachines in 1 10 100 1000 10000
	do
		for defence_aug in 0 1 2 4 8 16
		do
			echo "Running with $nmachines machines"
			uv run python main.py --config-name=config \
				defence=maybenot \
				defence.n_machines=$nmachines \
				dataset.defence_augmentation=$defence_aug \
				dataset.test_xv=$test_xv
		done
	done
done
