#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 10
#SBATCH --mem=100g
#SBATCH -p qTRDGPUH
#SBATCH -t 4440
#SBATCH --gres=gpu:A100:1
#SBATCH -J holoFMS
#SBATCH -D .
#SBATCH --output=./_out/smart-%j.out
#SBATCH -A psy53c17

sleep 10s

echo "Running on host: $HOSTNAME" >&2
echo "Job ID: $SLURM_JOB_ID" >&2

source /data/users2/ppopov1/miniconda/bin/activate catalyst12

echo "Using python from: $(which python)"
echo "Conda environment: $CONDA_DEFAULT_ENV"

dataset="fbirn"

python train_script_rev.py \
  experiment.experiment_name="smart_init" \
  experiment.collections=fbirn \
  experiment.dbfields="[falff,smri,dwi]" \
  experiment.metafields="[gender_encoded]" \
  model.masked=True \
  model.smart_init=True \
  "model.unimodal_model_paths.falff=./logs/baselines_fbirn_('falff',)_('gender_encoded',)_masked_False_sps_0.7/fold_0/checkpoints/best.pth" \
  "model.unimodal_model_paths.smri=./logs/baselines_fbirn_('smri',)_('gender_encoded',)_masked_False_sps_0.7/fold_0/checkpoints/best.pth" \
  "model.unimodal_model_paths.dwi=./logs/baselines_fbirn_('dwi',)_('gender_encoded',)_masked_False_sps_0.7/fold_0/checkpoints/best.pth"


sleep 10s
echo "Job $SLURM_JOB_ID completed"
