#!/bin/bash

# Run k times an experiment

k=10
epochs=300
batch_size=128

for dataset in LOW MED HIGH; do
    for distance in Hd SoftHd; do
        for layers in 2 3; do
            mkdir -p ./run/no_norm/$dataset/$1/$distance/${layers}_layer

            for ((run=0; run<$k; run++))
            do
                python train_learn_representation.py /media/priba/PPAP/NeuralMessagePassing/data/IAM/Letter/$dataset/ letters -s ./checkpoint/no_norm/$dataset/$1/$distance/${layers}_layer/$run/ --nlayers $layers -e $epochs --batch_size $batch_size > ./run/no_norm/$dataset/$1/$distance/${layers}_layer/$run.txt
            done
        done
    done
done


for dataset in LOW MED HIGH; do
    for distance in Hd SoftHd; do
        for layers in 2 3; do
            mkdir -p ./run/norm/$dataset/$1/$distance/${layers}_layer

            for ((run=0; run<$k; run++))
            do
                python train_learn_representation.py /media/priba/PPAP/NeuralMessagePassing/data/IAM/Letter/$dataset/ letters -s ./checkpoint/norm/$dataset/$1/$distance/${layers}_layer/$run/ --nlayers $layers -e $epochs --batch_size $batch_size --normalize > ./run/norm/$dataset/$1/$distance/${layers}_layer/$run.txt
            done
        done
    done
done
