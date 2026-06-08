#!/bin/bash
# The interpreter used to execute the script

#“#SBATCH” directives that convey submission options:
#SBATCH --job-name=tree_attribution
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128g
#SBATCH --time=108:00:00
#SBATCH -p standard

# The application(s) to execute along with its input arguments and options:

module load cuda/12.6.3
module load gcc/11.2.0
source ~/.bashrc
conda activate prompting
#python3.11-anaconda/2024.02

export HF_HOME=""  # set your HuggingFace model cache

# pip install jsonlines

# python reverse_mapping_with_score.py \
#     --skill_tree_path "tree_data/skill_mix/skill_mix.json" \
#     --qa_jsonl_path "omr_filtered_100K.jsonl" \
#     --tree_mapping_path "tree_data/skill_mix/omr_mapped_qa_chains.jsonl" \
#     --output_path "tree_data/skill_mix/omr_qwen7b" \
#     --sample_size 1000 \
#     --num_instances_to_show 100000 \
#     --attempt_model "Qwen/Qwen2.5-Math-7B-Instruct"

# python reverse_mapping_with_score.py \
#     --skill_tree_path "tree_data/skill_mix/skill_mix.json" \
#     --qa_jsonl_path "omr_filtered_100K.jsonl" \
#     --tree_mapping_path "tree_data/skill_mix/omr_mapped_qa_chains.jsonl" \
#     --output_path "tree_data/skill_mix/omr_qwen3_temp3" \
#     --sample_size 1000 \
#     --num_instances_to_show 100000 \
#     --attempt_model "Qwen/Qwen3-4B" \
#     --temperature 3

# python reverse_mapping_with_score.py \
#     --skill_tree_path "tree_data/evaltree/evaltree_math_full.json" \
#     --qa_jsonl_path "omr_filtered_100K.jsonl" \
#     --tree_mapping_path "tree_data/evaltree/omr_mapped_qa_chains.jsonl" \
#     --output_path "tree_data/evaltree/omr_qwen3_temp1_depth15" \
#     --sample_size 1000 \
#     --num_instances_to_show 100000 \
#     --attempt_model "Qwen/Qwen3-4B" \
#     --filtered_instance_file "omr_filtered_10K.jsonl" \
#     --max_depth 15 \
#     --temperature 1

# python reverse_mapping_with_score.py \
#     --skill_tree_path "tree_data/skill_mix/skill_mix.json" \
#     --qa_jsonl_path "omr_filtered_100K.jsonl" \
#     --tree_mapping_path "tree_data/skill_mix/omr_mapped_qa_chains.jsonl" \
#     --output_path "tree_data/skill_mix/omr_qwen3_temp1" \
#     --sample_size 5000 \
#     --num_instances_to_show 100000 \
#     --attempt_model "Qwen/Qwen3-4B" \
#     --temperature 1

# python reverse_mapping_with_score.py \
#     --skill_tree_path "tree_data/skill_mix/skill_mix.json" \
#     --qa_jsonl_path "omr_filtered_100K.jsonl" \
#     --tree_mapping_path "tree_data/skill_mix/omr_mapped_qa_chains.jsonl" \
#     --output_path "tree_data/skill_mix/omr_qwen3_temp2_correct0.5" \
#     --sample_size 1000 \
#     --num_instances_to_show 100000 \
#     --attempt_model "Qwen/Qwen3-4B" \
#     --temperature 2 \
#     --correct_portion 0.5

# python reverse_mapping_with_score.py \
#     --skill_tree_path "tree_data/skill_mix/skill_mix.json" \
#     --qa_jsonl_path "omr_filtered_100K.jsonl" \
#     --tree_mapping_path "tree_data/skill_mix/omr_mapped_qa_chains.jsonl" \
#     --output_path "tree_data/skill_mix/omr_qwen3_temp2_correct0_sequence_only" \
#     --sample_size 1000 \
#     --num_instances_to_show 100000 \
#     --attempt_model "Qwen/Qwen3-4B" \
#     --temperature 2 \
#     --correct_portion 0 \
#     --sample_particular_skill "sequence and series analysis skills" \
#     --filtered_instance_file "omr_filtered_10K.jsonl"

# python reverse_mapping_with_score.py \
#     --skill_tree_path "tree_data/skill_mix/skill_mix.json" \
#     --qa_jsonl_path "omr_filtered_100K.jsonl" \
#     --tree_mapping_path "tree_data/skill_mix/omr_mapped_qa_chains.jsonl" \
#     --output_path "tree_data/skill_mix/omr_qwen3_temp2_max_depth_1" \
#     --sample_size 1000 \
#     --num_instances_to_show 100000 \
#     --attempt_model "Qwen/Qwen3-4B" \
#     --temperature 2 \
#     --filtered_instance_file "omr_filtered_10K.jsonl" \
#     --max_depth 1

# python reverse_mapping_with_score.py \
#     --skill_tree_path "tree_data/skill_mix/skill_mix.json" \
#     --qa_jsonl_path "omr_filtered_100K.jsonl" \
#     --tree_mapping_path "tree_data/skill_mix/omr_mapped_qa_chains.jsonl" \
#     --output_path "tree_data/skill_mix/omr_qwen3_temp1_correct0_forced" \
#     --sample_size 1000 \
#     --num_instances_to_show 1000 \
#     --attempt_model "Qwen/Qwen3-4B" \
#     --temperature 1 \
#     --correct_portion 0 \
#     --filtered_instance_file "omr_filtered_10K.jsonl" \
#     --forced_distribution

# python reverse_mapping_with_score.py \
#     --skill_tree_path "tree_data/skill_mix/skill_mix.json" \
#     --qa_jsonl_path "omr_filtered_100K_w_qwen3_8b_final.jsonl" \
#     --tree_mapping_path "tree_data/skill_mix/omr_mapped_qa_chains.jsonl" \
#     --output_path "tree_data/skill_mix/omr_qwen3_8b_temp1" \
#     --sample_size 1000 \
#     --num_instances_to_show 100000 \
#     --attempt_model "Qwen/Qwen3-8B" \
#     --filtered_instance_file "omr_filtered_10K.jsonl" \
#     --temperature 1

# python reverse_mapping_with_score.py \
#     --skill_tree_path "tree_data/evaltree/evaltree_math_full.json" \
#     --qa_jsonl_path "omr_filtered_100K_w_qwen3_8b_final.jsonl" \
#     --tree_mapping_path "tree_data/evaltree/omr_mapped_qa_chains.jsonl" \
#     --output_path "tree_data/evaltree/omr_qwen3_8b_temp1" \
#     --sample_size 1000 \
#     --num_instances_to_show 100000 \
#     --attempt_model "Qwen/Qwen3-8B" \
#     --filtered_instance_file "omr_filtered_10K.jsonl" \
#     --temperature 1

# python reverse_mapping_with_score.py \
#     --skill_tree_path "tree_data/skill_mix/skill_mix.json" \
#     --qa_jsonl_path "omr_filtered_100K_final.jsonl" \
#     --tree_mapping_path "tree_data/skill_mix/omr_mapped_qa_chains.jsonl" \
#     --output_path "tree_data/skill_mix/omr_r1_llama_8b_temp1" \
#     --sample_size 1000 \
#     --num_instances_to_show 100000 \
#     --attempt_model "deepseek-ai/DeepSeek-R1-Distill-Llama-8B" \
#     --filtered_instance_file "omr_filtered_10K.jsonl" \
#     --temperature 1

# python reverse_mapping_with_score.py \
#     --skill_tree_path "tree_data/evaltree/evaltree_math_full.json" \
#     --qa_jsonl_path "omr_filtered_100K_final.jsonl" \
#     --tree_mapping_path "tree_data/evaltree/omr_mapped_qa_chains.jsonl" \
#     --output_path "tree_data/evaltree/omr_r1_llama_8b_temp1" \
#     --sample_size 1000 \
#     --num_instances_to_show 100000 \
#     --attempt_model "deepseek-ai/DeepSeek-R1-Distill-Llama-8B" \
#     --filtered_instance_file "omr_filtered_10K.jsonl" \
#     --temperature 1


# Below is for rebuttal

# # ============================================================================
# # 5K samples
# # ============================================================================
# python reverse_mapping_with_score.py \
#     --skill_tree_path "tree_data/skill_mix/skill_mix.json" \
#     --qa_jsonl_path "omr_filtered_100K_final.jsonl" \
#     --tree_mapping_path "tree_data/skill_mix/omr_mapped_qa_chains.jsonl" \
#     --output_path "tree_data/skill_mix/omr_qwen3_temp1" \
#     --sample_size 5000 \
#     --num_instances_to_show 100000 \
#     --attempt_model "Qwen/Qwen3-4B" \
#     --temperature 1

# # Prepare data for LLaMA-Factory (without skills)
# python prepare_data_and_yaml.py \
#     --input_path "tree_data/skill_mix/omr_qwen3_temp1/mapped_qa_chains_attempt_evaluated_5K.jsonl" \
#     --dataset_name "omr_qwen3_4b_skillmix_5K_temp1"

# # Prepare data for LLaMA-Factory (with skills)
# python prepare_data_and_yaml.py \
#     --input_path "tree_data/skill_mix/omr_qwen3_temp1/mapped_qa_chains_attempt_evaluated_5K.jsonl" \
#     --with_skill \
#     --skill_file_path "tree_data/skill_mix/omr_mapped_qa_chains.jsonl" \
#     --dataset_name "omr_qwen3_4b_skillmix_5K_temp1_w_skillmix"

# # ============================================================================
# # 10K samples
# # ============================================================================
# python reverse_mapping_with_score.py \
#     --skill_tree_path "tree_data/skill_mix/skill_mix.json" \
#     --qa_jsonl_path "omr_filtered_100K_final.jsonl" \
#     --tree_mapping_path "tree_data/skill_mix/omr_mapped_qa_chains.jsonl" \
#     --output_path "tree_data/skill_mix/omr_qwen3_temp1" \
#     --sample_size 10000 \
#     --num_instances_to_show 100000 \
#     --attempt_model "Qwen/Qwen3-4B" \
#     --temperature 1

# # Prepare data for LLaMA-Factory (without skills)
# python prepare_data_and_yaml.py \
#     --input_path "tree_data/skill_mix/omr_qwen3_temp1/mapped_qa_chains_attempt_evaluated_10K.jsonl" \
#     --dataset_name "omr_qwen3_4b_skillmix_10K_temp1"

# # Prepare data for LLaMA-Factory (with skills)
# python prepare_data_and_yaml.py \
#     --input_path "tree_data/skill_mix/omr_qwen3_temp1/mapped_qa_chains_attempt_evaluated_10K.jsonl" \
#     --with_skill \
#     --skill_file_path "tree_data/skill_mix/omr_mapped_qa_chains.jsonl" \
#     --dataset_name "omr_qwen3_4b_skillmix_10K_temp1_w_skillmix"

# # ============================================================================
# # 20K samples
# # ============================================================================
# python reverse_mapping_with_score.py \
#     --skill_tree_path "tree_data/skill_mix/skill_mix.json" \
#     --qa_jsonl_path "omr_filtered_100K_final.jsonl" \
#     --tree_mapping_path "tree_data/skill_mix/omr_mapped_qa_chains.jsonl" \
#     --output_path "tree_data/skill_mix/omr_qwen3_temp1" \
#     --sample_size 20000 \
#     --num_instances_to_show 100000 \
#     --attempt_model "Qwen/Qwen3-4B" \
#     --temperature 1

# # Prepare data for LLaMA-Factory (without skills)
# python prepare_data_and_yaml.py \
#     --input_path "tree_data/skill_mix/omr_qwen3_temp1/mapped_qa_chains_attempt_evaluated_20K.jsonl" \
#     --dataset_name "omr_qwen3_4b_skillmix_20K_temp1"

# # Prepare data for LLaMA-Factory (with skills)
# python prepare_data_and_yaml.py \
#     --input_path "tree_data/skill_mix/omr_qwen3_temp1/mapped_qa_chains_attempt_evaluated_20K.jsonl" \
#     --with_skill \
#     --skill_file_path "tree_data/skill_mix/omr_mapped_qa_chains.jsonl" \
#     --dataset_name "omr_qwen3_4b_skillmix_20K_temp1_w_skillmix"

# ============================================================================
# Random baseline samples (no tree-attribute selection)
# ============================================================================

# 5K random sample
python -c "
import pandas as pd
df = pd.read_json('omr_filtered_100K_final.jsonl', lines=True)
df.sample(n=5000, random_state=42).to_json('tree_data/omr_random_5K.jsonl', lines=True, orient='records')
"
python prepare_data_and_yaml.py \
    --input_path "tree_data/omr_random_5K.jsonl" \
    --dataset_name "omr_random_5K"
python prepare_data_and_yaml.py \
    --input_path "tree_data/omr_random_5K.jsonl" \
    --with_skill \
    --skill_file_path "tree_data/skill_mix/omr_mapped_qa_chains.jsonl" \
    --dataset_name "omr_random_5K_w_skillmix"

# 10K random sample
python -c "
import pandas as pd
df = pd.read_json('omr_filtered_100K_final.jsonl', lines=True)
df.sample(n=10000, random_state=42).to_json('tree_data/omr_random_10K.jsonl', lines=True, orient='records')
"
python prepare_data_and_yaml.py \
    --input_path "tree_data/omr_random_10K.jsonl" \
    --dataset_name "omr_random_10K"
python prepare_data_and_yaml.py \
    --input_path "tree_data/omr_random_10K.jsonl" \
    --with_skill \
    --skill_file_path "tree_data/skill_mix/omr_mapped_qa_chains.jsonl" \
    --dataset_name "omr_random_10K_w_skillmix"

# 20K random sample
python -c "
import pandas as pd
df = pd.read_json('omr_filtered_100K_final.jsonl', lines=True)
df.sample(n=20000, random_state=42).to_json('tree_data/omr_random_20K.jsonl', lines=True, orient='records')
"
python prepare_data_and_yaml.py \
    --input_path "tree_data/omr_random_20K.jsonl" \
    --dataset_name "omr_random_20K"
python prepare_data_and_yaml.py \
    --input_path "tree_data/omr_random_20K.jsonl" \
    --with_skill \
    --skill_file_path "tree_data/skill_mix/omr_mapped_qa_chains.jsonl" \
    --dataset_name "omr_random_20K_w_skillmix"