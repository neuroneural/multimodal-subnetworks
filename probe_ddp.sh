#!/bin/bash
#SBATCH -N 1                    # Number of nodes
#SBATCH -n 1                    # Number of tasks (Catalyst spawns 1 proc/GPU internally)
#SBATCH -c 20                   # CPU cores per task
#SBATCH --mem=96g               # Memory allocation
#SBATCH -p qTRDGPUH             # Partition name
#SBATCH -t 60                   # Time limit in minutes (probe is quick)
#SBATCH --gres=gpu:A100:2       # OVERRIDE per run: sbatch --gres=gpu:A100:1 probe_ddp.sh
#SBATCH -J ddp_probe            # Job name
#SBATCH -D .                    # start in the dir you submit from
#SBATCH --output=./_out/probe-%j.out
#SBATCH -A psy53c17             # allocation/project name

# ---------------------------------------------------------------------------
# DDP sampler probe launcher (Layer 2).
#
# Run the SAME probe at 1, 2 and 4 GPUs to compare sharding/seed/world-size:
#     sbatch --gres=gpu:A100:1 probe_ddp.sh current
#     sbatch --gres=gpu:A100:2 probe_ddp.sh current
#     sbatch --gres=gpu:A100:4 probe_ddp.sh current
# And the OLD (5fb4306) sampler for comparison:
#     sbatch --gres=gpu:A100:2 probe_ddp.sh old
#
# Tunables via env:
#     PROBE_N (default 200), PROBE_BS (default 4), PROBE_EPOCHS (default 2)
# ---------------------------------------------------------------------------

sleep 5s
echo "Running on host: $HOSTNAME" >&2
echo "Job ID: $SLURM_JOB_ID" >&2
echo "SLURM_GPUS_ON_NODE: $SLURM_GPUS_ON_NODE" >&2

SAMPLER="${1:-current}"          # current | old
N="${PROBE_N:-200}"
BS="${PROBE_BS:-4}"
EPOCHS="${PROBE_EPOCHS:-2}"
LOGDIR="./_probe_out/job_${SLURM_JOB_ID}_gpus${SLURM_GPUS_ON_NODE}_${SAMPLER}"

# Conda environment setup (same env as the real training job)
source /data/users2/ppopov1/miniconda/bin/activate catalyst12

mkdir -p "$LOGDIR"

python probe_ddp_catalyst.py \
  --logdir "$LOGDIR" \
  --n "$N" \
  --batch-size "$BS" \
  --epochs "$EPOCHS" \
  --sampler "$SAMPLER"

echo "=========== ANALYSIS ($SAMPLER, ${SLURM_GPUS_ON_NODE} GPUs) ==========="
# the runner nests one more level (sampler_n.._bs..); analyze that leaf dir(s)
for d in "$LOGDIR"/*/; do
  echo ">>> $d"
  python probe_analyze.py "$d" --data-size "$N"
done

sleep 5s
echo "Job $SLURM_JOB_ID completed"
