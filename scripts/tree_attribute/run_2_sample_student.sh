#!/bin/bash
# The interpreter used to execute the script

#“#SBATCH” directives that convey submission options:
#SBATCH --job-name=run_eval_qwen
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128g
#SBATCH --gpus=8
#SBATCH --time=48:00:00
#SBATCH -p spgpu

# The application(s) to execute along with its input arguments and options:

module load cuda/12.6.3
module load gcc/13.2.0
source ~/.bashrc
conda activate prompting
#python3.11-anaconda/2024.02

export HF_HOME=""  # set your HuggingFace model cache

# rm ~/.cache/vllm/torch_compile_cache

python sample_from_llm.py --model_name "deepseek-ai/DeepSeek-R1-Distill-Llama-8B" --input_path "omr_filtered_100K.jsonl" --output_path "omr_filtered_100K_w_llama_8b.jsonl" --batch_size 10000 --gpus "0,1,2,3,4,5,6,7"
# python judge_qwen_math.py