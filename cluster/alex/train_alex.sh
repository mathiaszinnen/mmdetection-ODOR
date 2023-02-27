#!/bin/bash -l
#SBATCH --time=23:59:59
#SBATCH --gres=gpu:a100:4
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

pip install .


mkdir -p ./data/ODOR-v3
tar xf /home/janus/iwi5-datasets/odor3/odor3.tar -C ./data/ODOR-v3

GPUS=4
CONFIG=configs/swin/odor3-swinl.py
WORK_DIR=/home/woody/iwi5/iwi5064h/mmdetection-workdirs/

./tools/dist_train.sh $CONFIG $GPUS --work-dir $WORK_DIR


echo "train done"
