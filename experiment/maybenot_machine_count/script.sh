#!/bin/bash


for _ in 0 1 2 3 4 5 6 7 8 9
do
	for nmachines in 0 1 10 100 1000 10000
	do
		echo "Running with $nmachines machines"
		python main.py --config-name=config defences.n_machines=$nmachines dataset.n_train_traces=7000 load_base_model=True
	done
done
