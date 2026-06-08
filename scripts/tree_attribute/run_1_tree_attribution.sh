#!/bin/bash
# The interpreter used to execute the script

#“#SBATCH” directives that convey submission options:
#SBATCH --job-name=tree_attribution
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128g
#SBATCH --time=48:00:00
#SBATCH -p standard

# The application(s) to execute along with its input arguments and options:

# module load cuda/12.6.3
module load gcc/11.2.0
source ~/.bashrc
conda activate prompting
#python3.11-anaconda/2024.02

export HF_HOME=""  # set your HuggingFace model cache

# python attribute_question_to_tree.py \
#     --input_type "s1_59K" \
#     --output_path "tree_data/skill_mix/mapped_qa_chains.jsonl" \
#     --skill_tree_path "tree_data/skill_mix/skill_mix.json" \
#     --api_url "http://gl1518.arc-ts.umich.edu:2341/v1/chat/completions" \
#     --model_name "Qwen/Qwen2.5-32B-Instruct"

python attribute_question_to_tree.py \
    --input_type "omr_filtered_100K.jsonl" \
    --output_path "tree_data/evaltree/omr_mapped_qa_chains.jsonl" \
    --skill_tree_path "tree_data/evaltree/evaltree_math_full.json" \
    --api_url "http://gl1700.arc-ts.umich.edu:2341/v1/chat/completions;http://gl1700.arc-ts.umich.edu:2342/v1/chat/completions;http://gl1700.arc-ts.umich.edu:2343/v1/chat/completions;http://gl1700.arc-ts.umich.edu:2344/v1/chat/completions;http://gl1723.arc-ts.umich.edu:2341/v1/chat/completions;http://gl1723.arc-ts.umich.edu:2342/v1/chat/completions;http://gl1723.arc-ts.umich.edu:2343/v1/chat/completions;http://gl1723.arc-ts.umich.edu:2344/v1/chat/completions" \
    --model_name "Qwen/Qwen2.5-32B-Instruct" \
    --num_proc 512