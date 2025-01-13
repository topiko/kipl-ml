#!/bin/bash


for nmachines in 0 1 10 100 1000 10000 100000
do
	python main.py --config-name=config defences.n_machines=$nmachines dataset.n_train_traces=7000
done
