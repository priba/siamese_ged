#!/bin/bash
#SBATCH --job-name=learn
#SBATCH --output=out.learn_%A.%a
#SBATCH --error=err.learn_%A.%a
#SBATCH --array=1-1
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=16384
#SBATCH --partition=gpu
#SBATCH --qos=job_gpu
#SBATCH --gres=gpu:1080ti:1
srun python ubelix/train_learn.py $SLURM_ARRAY_TASK_ID
