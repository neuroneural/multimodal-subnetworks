#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 24
#SBATCH --mem=100g
#SBATCH -p qTRDGPUH
#SBATCH -t 7200
#SBATCH --gres=gpu:A100:1
#SBATCH -J fm_smart_init
#SBATCH -D /data/users2/jwardell1/multimodal-subnetworks-smart-init
#SBATCH --output=/data/users2/jwardell1/multimodal-subnetworks-smart-init/_out/%j.out
#SBATCH -A psy53c17

sleep 10s
echo "Running on host: $HOSTNAME" >&2
echo "Job ID: $SLURM_JOB_ID" >&2
echo "TMPDIR is: $TMPDIR" >&2
export TMPDIR=/tmp
source /data/users2/jwardell1/miniconda3/bin/activate mmsn312
echo "Using python from: $(which python)"
echo "Conda environment: $CONDA_DEFAULT_ENV"

python3 train_script_rev.py \
    --config-name fbirn_smart_init \
    --config-dir conf

sleep 10s
echo "Job $SLURM_JOB_ID completed"
