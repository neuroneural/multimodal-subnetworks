#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 24
#SBATCH --mem=100g
#SBATCH -p qTRDGPUH
#SBATCH -t 7200
#SBATCH --gres=gpu:A100:1
#SBATCH -J fc_opt1_masked
#SBATCH -D /data/users2/jwardell1/multimodal-subnetworks-fc-option1
#SBATCH --output=/data/users2/jwardell1/multimodal-subnetworks-fc-option1/_out/%j.out
#SBATCH -A psy53c17

sleep 10s
echo "Running on host: $HOSTNAME"
echo "Job ID: $SLURM_JOB_ID"
export TMPDIR=/tmp
source /data/users2/jwardell1/miniconda3/bin/activate mmsn312

python3 train_script_rev.py \
    --config-name new_conf \
    --config-dir conf \
    experiment.experiment_name=fbirn_multimodal_masked_fc_option1 \
    experiment.collections=fbirn \
    experiment.dbfields=[falff,smri,dwi] \
    experiment.metafields=[gender_encoded] \
    experiment.epochs=60 \
    experiment.cv_folds=10 \
    experiment.num_workers=16 \
    model.masked=True \
    model.sparsity=0.7 \
    model.snip_batch_size=20 \
    model.smart_init=False \
    model.model_channels=64 \
    model.init_weights_path=/data/users2/jwardell1/multimodal-subnetworks-fc-option1/init_weights_seed1997_fc_option1.pth

echo "Job $SLURM_JOB_ID completed"
