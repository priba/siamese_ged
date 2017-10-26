#!/bin/bash
#SBATCH --job-name=siamese
#SBATCH --output=out.siamese_%A.%a
#SBATCH --error=err.siamese_%A.%a
#SBATCH --array=1-1
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=16384
#SBATCH --partition=gpu
#SBATCH --qos=job_gpu
#SBATCH --gres=gpu:1080ti:1
srun ruby rubyscripts/train_siamese.rb $SLURM_ARRAY_TASK_ID
