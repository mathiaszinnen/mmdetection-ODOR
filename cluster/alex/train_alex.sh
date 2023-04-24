#!/bin/bash -l
#SBATCH --time=10:59:59
#SBATCH --gres=gpu:a40:8
#SBATCH --job-name=deart_on_odor_head
#SBATCH --export=NONE

unset SLURM_EXPORT_ENV

module load python
module load cuda

echo 'execution started'

source activate openmmlab


mkdir ${TMPDIR}/$SLURM_JOB_ID
cd ${TMPDIR}/$SLURM_JOB_ID

cp -r /home/woody/iwi5/iwi5064h/mmdetection-ODOR .

cd mmdetection-ODOR

echo 'starting pip install'

pip install .


mkdir -p ./data/ODOR-v3
tar xf /home/janus/iwi5-datasets/odor3/odor3.tar -C ./data/ODOR-v3



GPUS=8
CONFIG=$1
WORK_DIR='/home/woody/iwi5/iwi5064h/mmdetection-workdirs/deart_on_odor_head'

echo 'train with ' ${CONFIG}

./tools/dist_train.sh ${CONFIG} ${GPUS}


echo "train done"
