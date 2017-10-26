#!/bin/bash
#SBATCH --job-name=sdistance
#SBATCH --output=out.sdistance_%A.%a
#SBATCH --error=err.sdistance_%A.%a
#SBATCH --array=1-1
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=16384
#SBATCH --partition=gpu
#SBATCH --qos=job_gpu
#SBATCH --gres=gpu:1080ti:1
srun python ubelix/train_siamese_distance.py $SLURM_ARRAY_TASK_ID
