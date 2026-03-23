#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 24
#SBATCH --mem=100g
#SBATCH -p qTRDGPUH
#SBATCH -t 7200
#SBATCH --gres=gpu:A100:1
#SBATCH --nodelist=arctrddgxa002
#SBATCH -J ukbf_m
#SBATCH -D /data/users2/jwardell1/multimodal-subnetworks
#SBATCH --output=/data/users2/jwardell1/multimodal-subnetworks/_out/smart-%j.out
#SBATCH -A psy53c17
sleep 10s
echo "Running on host: $HOSTNAME" >&2
echo "Job ID: $SLURM_JOB_ID" >&2
echo "TMPDIR is: $TMPDIR" >&2
export TMPDIR=/tmp
source /data/users2/jwardell1/miniconda3/bin/activate mmsn312
echo "Using python from: $(which python)"
echo "Conda environment: $CONDA_DEFAULT_ENV"
export CUDA_LAUNCH_BLOCKING=1
export HYDRA_FULL_ERROR=1
export PYTHONFAULTHANDLER=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
dataset="ukb"
modality="falff"
python3 train_script_rev.py \
    --config-name new_conf \
    --config-dir conf \
    experiment.experiment_name=${dataset}_${modality}_lz4 \
    experiment.collections=$dataset \
    experiment.dbfields=[falff] \
    experiment.metafields=[gender_encoded] \
    model.masked=True \
    model.model_channels=64
sleep 10s
echo "Job $SLURM_JOB_ID completed"
