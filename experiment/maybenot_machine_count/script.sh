#!/bin/bash

max_padding_frac=0.5
ntrain_traces=7000

for defence_aug in 0 1 2 4 8 16
do
	for nmachines in 0 1 10 100 1000 8000
	do
		for netwk_delay in 0 100 500
		do
			echo "Running with $nmachines machines"
				python main.py --config-name=config \
				defences.n_machines=$nmachines \
				defences.max_padding_frac=$max_padding_frac \
				dataset.n_train_traces=$ntrain_traces \
				dataset.defence_augmentation=$defence_aug \
				network.network_delay_millis=$netwk_delay \
				load_base_model=True
		done
	done
done
