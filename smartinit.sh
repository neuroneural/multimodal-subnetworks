#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 10
#SBATCH --mem=100g
#SBATCH -p qTRDGPUH
#SBATCH -t 4440
#SBATCH --gres=gpu:A100:1
#SBATCH -J fbirn_smart_init
#SBATCH -D .
#SBATCH --output=./_out/smart_init-%j.out
#SBATCH -A psy53c17

sleep 10s

echo "Running on host: $HOSTNAME" >&2
echo "Job ID: $SLURM_JOB_ID" >&2

source /data/users2/jwardell1/miniconda3/bin/activate mmsn

echo "Using python from: $(which python)"
echo "Conda environment: $CONDA_DEFAULT_ENV"

python train_script_rev.py \
  --config-name fbirn_smart_init \
  --config-dir conf
