#!/bin/bash
#SBATCH -N 1                    # Number of nodes
#SBATCH -n 1                    # Number of tasks (processes)
#SBATCH -c 16                    # CPU cores per task
#SBATCH --mem=64g               # Memory allocation
#SBATCH -p qTRDGPUH             # Partition name
#SBATCH -t 1440                 # Time limit in minutes
#SBATCH --gres=gpu:V100:4            # Single GPU sufficient for ResNet3D
#SBATCH -J resnet3d-gender      # Job name reflecting task
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

# Training configuration
CONFIG_NAME="resnet3d_gender_bn_64base_2.2.2.2_exp01"
CONFIG_PATH="conf"
SCRIPT_NAME="train_resnet3d.py"
LOG_DIR="logs/resnet3d_gender_${SLURM_JOB_ID}"

# Create log directory
mkdir -p $LOG_DIR

# Run training with Hydra
python $SCRIPT_NAME \
    --config-name $CONFIG_NAME \
    --config-dir $CONFIG_PATH 


# Cleanup
sleep 10s
echo "Job $SLURM_JOB_ID completed"
