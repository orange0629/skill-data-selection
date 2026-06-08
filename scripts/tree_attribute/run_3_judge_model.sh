#!/bin/bash
# The interpreter used to execute the script

#“#SBATCH” directives that convey submission options:
#SBATCH --job-name=run_eval_qwen
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64g
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH -p spgpu

# The application(s) to execute along with its input arguments and options:

module load cuda/12.6.3
module load gcc/11.2.0
source ~/.bashrc
conda activate prompting
#python3.11-anaconda/2024.02

export HF_HOME=""  # set your HuggingFace model cache


python judge_qwen_math.py --attempt_model "Qwen/Qwen2.5-Math-7B" --judge_model "Qwen/Qwen2.5-7B-Instruct" --batch_size 1000 --attempt_path "mapped_qa_chains_attempt_2.jsonl"