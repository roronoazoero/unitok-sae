#!/bin/bash
#SBATCH --partition=gpu-a100-80g
#SBATCH --gres=gpu:a100:4
#SBATCH --cpus-per-task=4
#SBATCH --mem=35G
#SBATCH --time=4:00:00
#SBATCH --job-name=sae_k64

cd $WRKDIR/UniTok
python sae/train_sae.py --k 64 --output_dir sae/checkpoints/k64/ --wandb --wandb_run sae_k64_pm1
