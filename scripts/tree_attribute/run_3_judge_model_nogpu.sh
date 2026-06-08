#!/bin/bash
# The interpreter used to execute the script

#“#SBATCH” directives that convey submission options:
#SBATCH --job-name=run_eval_qwen
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128g
#SBATCH --gpus=0
#SBATCH --time=24:00:00
#SBATCH -p standard

# The application(s) to execute along with its input arguments and options:

module load cuda/12.6.3
module load gcc/11.2.0
source ~/.bashrc
conda activate prompting
#python3.11-anaconda/2024.02

export HF_HOME=""  # set your HuggingFace model cache

# pip install math-verify[antlr4_13_2]
# pip install swifter

python judge_qwen_math.py \
    --attempt_model "saves/qwen3_4B/sft_omr_1k_w_skillmix_temp2/checkpoint-625" \
    --judge_model "cpu" \
    --batch_size 1000 \
    --attempt_path "./omr_filtered_10K_w_custom_2.jsonl"