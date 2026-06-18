#!/bin/bash
#SBATCH -N 1                    # Number of nodes
#SBATCH -n 1                    # Number of tasks (processes)
#SBATCH -c 20                   # CPU cores per task
#SBATCH --mem=96g               # Memory allocation
#SBATCH -p qTRDGPUH             # Partition name
#SBATCH -t 5760                 # Time limit in minutes (4 days)
#SBATCH --gres=gpu:A100:1       # 1 GPU (masked routes modalities sequentially)
#SBATCH -J fbirn_masked_base    # Job name
#SBATCH -D .                    # node starting path = path you submit from
#SBATCH --output=./_out/run-%j.out
#SBATCH -A psy53c17             # elpis project name

# Wait for node allocation
sleep 10s

echo "Running on host: $HOSTNAME" >&2
echo "Job ID: $SLURM_JOB_ID" >&2

# Conda environment setup
source /data/users2/ppopov1/miniconda/bin/activate catalyst12
echo "Using python from: $(which python)" >&2

export HYDRA_FULL_ERROR=1

# Baseline: masked (SNIP) multimodal with the OLD single shared classifier head
python train_script_rev.py \
  --config-name fbirn_masked_baseline \
  --config-dir conf

# Cleanup
sleep 10s
echo "Job $SLURM_JOB_ID completed"
