#!/bin/bash
# The interpreter used to execute the script

#“#SBATCH” directives that convey submission options:
#SBATCH --job-name=sft_s1
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=180g
#SBATCH --gpus=8
#SBATCH --time=48:00:00
#SBATCH -A qdj_project_owned3
#SBATCH -p spgpu2

# The application(s) to execute along with its input arguments and options:

module load cuda/12.6.3
module load gcc/13.2.0
source ~/.bashrc
conda activate 360-llama-factory
#python3.11-anaconda/2024.02

export HF_HOME=""  # set your HuggingFace model cache
export FORCE_TORCHRUN=1

llamafactory-cli train llama-factory_omr_qwen3_4b_skillmix_omr_5K_temp1.yaml
llamafactory-cli train llama-factory_omr_qwen3_4b_skillmix_5K_temp1_w_skillmix.yaml

llamafactory-cli train llama-factory_omr_qwen3_4b_skillmix_omr_10K_temp1.yaml
llamafactory-cli train llama-factory_omr_qwen3_4b_skillmix_10K_temp1_w_skillmix.yaml

llamafactory-cli train llama-factory_omr_qwen3_4b_skillmix_omr_20K_temp1.yaml
llamafactory-cli train llama-factory_omr_qwen3_4b_skillmix_20K_temp1_w_skillmix.yaml