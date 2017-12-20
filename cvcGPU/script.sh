#!/bin/bash

MIN_USER=2 # Min number of self-owned processes in the queue
GLOBAL_MAX=3 # Max number of processes in the queue. If the queue has equal or less processes than this number, it will launch processes until it is accomplished.
WAIT_TIME=60 # Wait time to poll the queue

for i in `seq 0 $2`; do
    while [[ `squeue | grep $USER | wc -l` -ge $MIN_USER && `squeue | tail -n +2 | wc -l` > $GLOBAL_MAX ]]; do # while in use, do nothing
        sleep $WAIT_TIME
        echo 'found' `squeue | grep $USER | wc -l` 'tasks, waiting'
    done
    echo "run $i"
    sbatch --gres=gpu:1 ./scripts/$1.sh $i # script launch, modify if needed
    sleep 1 # to make sure the queue is updated before next iter
done