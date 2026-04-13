#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 24
#SBATCH --mem=100g
#SBATCH -p qTRDGPUH
#SBATCH -t 7200
#SBATCH --gres=gpu:A100:1
#SBATCH -J fm_snip_fi
#SBATCH -D /data/users2/jwardell1/multimodal-subnetworks-smart-init
#SBATCH --output=/data/users2/jwardell1/multimodal-subnetworks-smart-init/_out/%A_%a.out
#SBATCH -A psy53c17
#SBATCH --array=0-15

sleep 10s
echo "Running on host: $HOSTNAME" >&2
echo "Job ID: $SLURM_JOB_ID, Array Task ID: $SLURM_ARRAY_TASK_ID" >&2
echo "TMPDIR is: $TMPDIR" >&2
export TMPDIR=/tmp
source /data/users2/jwardell1/miniconda3/bin/activate mmsn312
echo "Using python from: $(which python)"
echo "Conda environment: $CONDA_DEFAULT_ENV"

# Read sweep parameters
SPARSITY_VALUES=($(cat slurm_scripts/fbirn/sparsity_values.txt))
SNIP_BATCH_VALUES=($(cat slurm_scripts/fbirn/snip_batch_values.txt))

SPARSITY=${SPARSITY_VALUES[$SLURM_ARRAY_TASK_ID]}
SNIP_BATCH=${SNIP_BATCH_VALUES[$SLURM_ARRAY_TASK_ID]}

echo "Sweep: sparsity=${SPARSITY}, snip_batch_size=${SNIP_BATCH}" >&2

dataset="fbirn"
modality="multimodal"

python3 train_script_rev.py \
    --config-name new_conf \
    --config-dir conf \
    experiment.experiment_name=${dataset}_${modality}_snip_fi_sps${SPARSITY}_sb${SNIP_BATCH} \
    experiment.collections=$dataset \
    experiment.dbfields=[falff,smri,dwi] \
    experiment.metafields=[gender_encoded] \
    model.masked=True \
    model.sparsity=${SPARSITY} \
    model.snip_batch_size=${SNIP_BATCH} \
    model.model_channels=64

sleep 10s
echo "Job $SLURM_JOB_ID array task $SLURM_ARRAY_TASK_ID completed"
