#!/bin/bash -l
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:a40:0
#SBATCH --job-name=openmmlab-env
#SBATCH --export=NONE

unset SLURM_EXPORT_ENV

module load python
module load cuda



source activate openmmlab

pip install pytorch torchvision

pip install -U openmim

mim install mmcv-full

pip install mmdet
