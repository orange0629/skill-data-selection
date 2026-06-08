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

module load cuda/12.6.3
module load gcc/13.2.0
source ~/.bashrc
conda activate prompting
#python3.11-anaconda/2024.02

export HF_HOME=""  # set your HuggingFace model cache

python evaluate_by_skill_tree.py \
  --skill_tree_path "tree_data/skill_mix/skill_mix.json" \
  --qa_jsonl_path ./omr_filtered_100K.jsonl \
  --mapping_jsonl_path tree_data/skill_mix/omr_mapped_qa_chains.jsonl \
  --output_dir outputs/eval_vllm \
  --model_name Qwen/Qwen3-4B
