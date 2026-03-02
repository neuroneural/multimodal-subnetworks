#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 10
#SBATCH --mem=128g
#SBATCH -p qTRDGPUH
#SBATCH -t 4440
#SBATCH --gres=gpu:A100:2
#SBATCH -J ukbfalff
#SBATCH -D .
#SBATCH --output=./_out/%j.out
#SBATCH -A psy53c17

sleep 10s

echo "Running on host: $HOSTNAME" >&2
echo "Job ID: $SLURM_JOB_ID" >&2

source /data/users2/ppopov1/miniconda/bin/activate catalyst12

echo "Using python from: $(which python)"
echo "Conda environment: $CONDA_DEFAULT_ENV"

dataset="ukb"


python train_script_rev.py \
  --config-name new_conf \
  --config-dir conf \
  experiment.experiment_name="baselines" \
  experiment.collections=$dataset \
  experiment.dbfields="[falff]" \
  experiment.metafields="[gender_encoded]" \
  model.masked=False 
python train_script_rev.py \
  --config-name new_conf \
  --config-dir conf \
  experiment.experiment_name="masked" \
  experiment.collections=$dataset \
  experiment.dbfields="[falff]" \
  experiment.metafields="[gender_encoded]" \
  model.masked=True 


  

sleep 10s
echo "Job $SLURM_JOB_ID completed"
