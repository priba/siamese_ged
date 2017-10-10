#!/bin/bash

# Run k times an experiment

mkdir -p ./run/$1

k=10
epochs=2
layers=2
for ((run=0; run<$k; run++))
do
    python train_learn_representation.py /media/priba/PPAP/NeuralMessagePassing/data/IAM/Letter/LOW/ letters -s ./checkpoint/$1/$run/ --nlayers $layers -e $epochs > ./run/$1/$run.txt
done
