#!/bin/bash

# Run k times an experiment

k=10
epochs=300
batch_size=128
lr=1e-2
pipeline=siamese_net
dataset=$1

for layers in 2 3; do
    for edges in adj feat; do
        mkdir -p ./run/$pipeline/$dataset/no_norm/${layers}_layer/$edges/
        echo  ./run/$pipeline/$dataset/no_norm/${layers}_layer/$edges/

        for ((run=0; run<$k; run++))
        do
            python train_siamese_net.py /media/priba/PPAP/NeuralMessagePassing/data/IAM/Letter/$dataset/ letters -s ./checkpoint/$pipeline/$dataset/no_norm/${layers}_layer/$edges/$run/ --log ./log/$pipeline/$dataset/no_norm/${layers}_layer/$edges/$run/ -lr $lr --nlayers $layers -e $epochs -b $batch_size --representation $edges --schedule $epochs+1 > ./run/$pipeline/$dataset/no_norm/${layers}_layer/$edges/$run.txt
        done
    done
done

for layers in 2 3; do
    for edges in adj feat; do
        mkdir -p ./run/$pipeline/$dataset/norm/${layers}_layer/$edges/
        echo  ./run/$pipeline/$dataset/norm/${layers}_layer/$edges/

        for ((run=0; run<$k; run++))
        do
            python train_siamese_net.py /media/priba/PPAP/NeuralMessagePassing/data/IAM/Letter/$dataset/ letters -s ./checkpoint/$pipeline/$dataset/norm/${layers}_layer/$edges/$run/ --log ./log/$pipeline/$dataset/norm/${layers}_layer/$edges/$run/ -lr $lr --nlayers $layers -e $epochs -b $batch_size --representation $edges --schedule $epochs+1 --normalization > ./run/$pipeline/$dataset/norm/${layers}_layer/$edges/$run.txt
        done
    done
done
