#!/bin/bash
#SBATCH --job-name=smoke_lora
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=smoke_lora-%j.out
#SBATCH --error=smoke_lora-%j.err

module load miniforge/24.7.1
cd $SCRATCH/repos/llama-peft-project
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate llama-peft

scripts/cluster/run_job.sh --config configs/llama3_8b_lora_l40_protocol_v2_smoke_seed42.yaml
