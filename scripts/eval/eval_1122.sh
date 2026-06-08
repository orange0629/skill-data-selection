#!/bin/bash
# The interpreter used to execute the script

#“#SBATCH” directives that convey submission options:
#SBATCH --job-name=run_eval_ckpt_lora
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64g
#SBATCH --gpus=4
#SBATCH --time=48:00:00
#SBATCH -p spgpu

# The application(s) to execute along with its input arguments and options:

module load cuda/12.6.3
module load gcc/11.2.0
source ~/.bashrc
conda activate prompting
#python3.11-anaconda/2024.02

export HF_HOME=""  # set your HuggingFace model cache


CUDA_VISIBLE_DEVICES=0 python eval_checkpoints.py --model_dir "saves/qwen3_4B/sft_limo/checkpoint-510" --base_model "Qwen/Qwen3-4B" --benchmark_name "aime" &
CUDA_VISIBLE_DEVICES=4 python eval_checkpoints.py --model_dir "saves/qwen3_4B/sft_s1k-1.1/checkpoint-625" --base_model "Qwen/Qwen3-4B" --benchmark_name "aime" &
sleep 300
CUDA_VISIBLE_DEVICES=1 python eval_checkpoints.py --model_dir "saves/qwen3_4B/sft_limo/checkpoint-510" --base_model "Qwen/Qwen3-4B" --benchmark_name "amc23" &
CUDA_VISIBLE_DEVICES=2 python eval_checkpoints.py --model_dir "saves/qwen3_4B/sft_limo/checkpoint-510" --base_model "Qwen/Qwen3-4B" --benchmark_name "aime25" &
CUDA_VISIBLE_DEVICES=3 python eval_checkpoints.py --model_dir "saves/qwen3_4B/sft_limo/checkpoint-510" --base_model "Qwen/Qwen3-4B" --benchmark_name "mathl5" &
CUDA_VISIBLE_DEVICES=5 python eval_checkpoints.py --model_dir "saves/qwen3_4B/sft_s1k-1.1/checkpoint-625" --base_model "Qwen/Qwen3-4B" --benchmark_name "amc23" &
CUDA_VISIBLE_DEVICES=6 python eval_checkpoints.py --model_dir "saves/qwen3_4B/sft_s1k-1.1/checkpoint-625" --base_model "Qwen/Qwen3-4B" --benchmark_name "aime25" &
CUDA_VISIBLE_DEVICES=7 python eval_checkpoints.py --model_dir "saves/qwen3_4B/sft_s1k-1.1/checkpoint-625" --base_model "Qwen/Qwen3-4B" --benchmark_name "mathl5" &

wait
saves/qwen3/sft_omr_qwen3_4b_skillmix_5K_temp1_w_skillmix