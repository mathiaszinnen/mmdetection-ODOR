#!/bin/bash -l
#SBATCH --time=00:10:00
#SBATCH --gres=gpu:a40:1
#SBATCH --job-name=frcnn-swinl
#SBATCH --export=NONE

unset SLURM_EXPORT_ENV

module load python
module load cuda

source activate openmmlab

mkdir ${TMPDIR}/$SLURM_JOB_ID
cd ${TMPDIR}/$SLURM_JOB_ID

cp -r /home/woody/iwi5/iwi5064h/mmdetection-ODOR .

cd mmdetection-ODOR


mkdir -p ./data/ODOR-v3
tar xf /home/janus/iwi5-datasets/odor3/odor3.tar -C ./data/ODOR-v3

GPUS=1
CONFIG=configs/swin/odor3-swinl.py

./tools/dist_train.sh $CONFIG $GPUS 


echo "train done"
