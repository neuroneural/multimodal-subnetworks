#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 12
#SBATCH --mem=128g
#SBATCH -p qTRDGPUH
#SBATCH -t 02:00:00
#SBATCH --gres=gpu:1
#SBATCH -J fm_disjoint_test
#SBATCH -D /data/users2/jwardell1/multimodal-subnetworks
#SBATCH --output=/data/users2/jwardell1/multimodal-subnetworks/_out/%x_%j.out
#SBATCH -A psy53c17
#SBATCH --exclude=arctrddgxa001

sleep 10s
echo "Running on host: $HOSTNAME" >&2
echo "Job ID: $SLURM_JOB_ID" >&2
echo "TMPDIR is: $TMPDIR" >&2
export TMPDIR=/tmp
export WANDB_X_STATS_SAMPLING_INTERVAL=2

source /data/users2/jwardell1/miniconda3/bin/activate mmsn312
echo "Using python from: $(which python)"
echo "Conda environment: $CONDA_DEFAULT_ENV"

dataset="fbirn"

python3 train_script_rev.py \
    --config-name new_conf \
    --config-dir conf \
    experiment.experiment_name=${dataset}_disjoint_mask_test \
    experiment.collections=$dataset \
    experiment.dbfields=[falff,smri,dwi] \
    experiment.metafields=[gender_encoded] \
    experiment.cv_folds=10 \
    experiment.max_folds=1 \
    model.masked=True \
    model.disjoint_mask_init=True \
    model.sparsity=0.7 \
    model.snip_batch_size=20 \
    model.model_channels=64 \
    model.model_init_seed=1997 \
    experiment.numvolumes=4 \
    experiment.num_workers=6 \
    experiment.prefetches=2 \
    experiment.prefetch_factor=2 \
    experiment.train_num_workers=6 \
    experiment.train_prefetches=2 \
    experiment.train_prefetch_factor=2 \
    experiment.train_persistent_workers=True \
    experiment.eval_num_workers=6 \
    experiment.eval_prefetches=2 \
    experiment.eval_prefetch_factor=2 \
    experiment.eval_persistent_workers=True \
    experiment.profile_timings=False \
    experiment.timing_sync_cuda=False \
    experiment.cudnn_benchmark=False \
    experiment.lr_scale=0.005 \
    experiment.epochs=10

sleep 10s
echo "Job $SLURM_JOB_ID completed"
