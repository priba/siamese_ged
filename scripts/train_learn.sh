#!/bin/bash

# Run k times an experiment

k=5
epochs=300
batch_size=128
lr=1e-2
pipeline=learn_representation
dataset=$1
distance=SoftHd
    
for layers in 2 3; do
    for edges in adj feat; do
        mkdir -p ./run/$pipeline/$dataset/$distance/${layers}_layer/$edges/
        echo  ./run/$pipeline/$dataset/$distance/${layers}_layer/$edges/

        for ((run=0; run<$k; run++))
        do
            python train_learn_representation.py /media/priba/PPAP/NeuralMessagePassing/data/IAM/Letter/$dataset/ letters -s ./checkpoint/$pipeline/$dataset/$distance/${layers}_layer/$edges/$run/ --log ./log/$pipeline/$dataset/$distance/${layers}_layer/$edges/$run/ -lr $lr --nlayers $layers -e $epochs -b $batch_size --representation $edges --schedule 100 200 250 --distance $distance > ./run/$pipeline/$dataset/$distance/${layers}_layer/$edges/$run.txt
        done
    done
done

