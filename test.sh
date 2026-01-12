#!/bin/bash
#SBATCH -N 1                    # Number of nodes
#SBATCH -n 1                    # Number of tasks (processes)
#SBATCH -c 8                    # CPU cores per task
#SBATCH --mem=64g               # Memory allocation
#SBATCH -p qTRDGPUH             # Partition name
#SBATCH -t 1440                 # Time limit in minutes
#SBATCH --gres=gpu:V100:2            # Single GPU sufficient for ResNet3D
#SBATCH -J holo_test      # Job name reflecting task
#SBATCH -D .                        # adding this means that node starting path is the path from which you run this script
#SBATCH --output=./_out/run-%j.out     # output file name
#SBATCH -A psy53c17                 # elpis project name, can be different for you, check you allocations at https://elpis.rs.gsu.edu/

# Wait for node allocation
sleep 10s

echo "Running on host: $HOSTNAME" >&2
echo "Job ID: $SLURM_JOB_ID" >&2

# Conda environment setup
source /data/users2/ppopov1/miniconda/bin/activate catalyst12

# Verify environment
echo "Using python from: $(which python)"
echo "Conda environment: $CONDA_DEFAULT_ENV"

DATASETS=(
  "fbirn"
  "ukb"
)

MODALITY=(
  "dwi"
  "falff"
  "smri"
)

TASK_ID=${SLURM_ARRAY_TASK_ID:-0}

dataset_id=$(( TASK_ID / ${#MODALITY[@]} ))
modality_id=$(( TASK_ID % ${#MODALITY[@]} ))

echo "Running with dataset: ${DATASETS[$dataset_id]}"
echo "Running with modality: ${MODALITY[$modality_id]}"

python train_script_rev.py \
  --config-name new_conf \
  --config-dir conf \
  experiment.experiment_name="baselines" \
  experiment.collections="${DATASETS[$dataset_id]}" \
  experiment.dbfields="[${MODALITY[$modality_id]}]" \
  experiment.metafields="[gender_encoded]"

## FBIRN modalities
# python train_script_rev.py \
#   --config-name new_conf \
#   --config-dir conf \
#   experiment.experiment_name="baselines" \
#   experiment.collections="${DATASETS[$dataset_id]}" \
#   experiment.dbfields="[dwi]" \
#   experiment.metafields="[gender_encoded]"

# python train_script_rev.py \
#   --config-name new_conf \
#   --config-dir conf \
#   experiment.experiment_name="baselines" \
#   experiment.collections="${DATASETS[$dataset_id]}" \
#   experiment.dbfields="[falff]" \
#   experiment.metafields="[gender_encoded]"

# python train_script_rev.py \
#   --config-name new_conf \
#   --config-dir conf \
#   experiment.experiment_name="baselines" \
#   experiment.collections="${DATASETS[$dataset_id]}" \
#   experiment.dbfields="[smri]" \
#   experiment.metafields="[gender_encoded]"


## TEST
# python train_script_rev.py \
#   --config-name new_conf \
#   --config-dir conf \
#   experiment.experiment_name="test" \
#   experiment.collections="fbirn" \
#   experiment.dbfields="[smri]" \
#   experiment.metafields="[gender_encoded]"
  
# Cleanup
sleep 10s
echo "Job $SLURM_JOB_ID completed"
