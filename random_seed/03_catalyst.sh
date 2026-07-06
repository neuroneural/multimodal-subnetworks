#!/bin/bash
#SBATCH -N 1                    # Number of nodes
#SBATCH -n 1                    # Number of tasks (Catalyst spawns 1 proc/GPU internally)
#SBATCH -c 8                    # CPU cores per task
#SBATCH --mem=16g               # Memory allocation
#SBATCH -p qTRDGPUH             # Partition name
#SBATCH -t 15                   # Time limit in minutes (quick check)
#SBATCH --gres=gpu:A100:2       # OVERRIDE per run: sbatch --gres=gpu:A100:1 03_catalyst.sh
#SBATCH -J seedcheck            # Job name
#SBATCH -D .                    # start in the dir you submit from
#SBATCH --output=seedcheck-%j.out   # .out lands right where you run sbatch
#SBATCH -A psy53c17             # allocation/project name

# ---------------------------------------------------------------------------
# Seed-behavior check under real Catalyst DDP (03_catalyst.py).
# Submit from this (random_seed/) directory, picking the GPU count on the CLI:
#
#     sbatch --gres=gpu:A100:1 03_catalyst.sh
#     sbatch --gres=gpu:A100:2 03_catalyst.sh
#     sbatch --gres=gpu:A100:4 03_catalyst.sh
# ---------------------------------------------------------------------------

echo "Running on host: $HOSTNAME" >&2
echo "Job ID: $SLURM_JOB_ID" >&2
echo "SLURM_GPUS_ON_NODE: $SLURM_GPUS_ON_NODE" >&2

source /data/users2/ppopov1/miniconda/bin/activate catalyst12

python 03_catalyst.py
