#!/bin/bash

# Run k times an experiment

k=10
epochs=2
layers=3
batch_size=128

for dataset in LOW MED HIGH; do
    mkdir -p ./run/$dataset/$1

    for ((run=0; run<$k; run++))
    do
        python train_learn_representation.py /media/priba/PPAP/NeuralMessagePassing/data/IAM/Letter/$dataset/ letters -s ./checkpoint/$dataset/$1/$run/ --nlayers $layers -e $epochs --batch_size $batch_size > ./run/$dataset/$1/$run.txt
    done
done

