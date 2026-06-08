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
#SBATCH -p spgpu2

# The application(s) to execute along with its input arguments and options:

module load cuda/12.6.3
module load gcc/13.2.0
source ~/.bashrc
conda activate prompting
#python3.11-anaconda/2024.02

export HF_HOME=""  # set your HuggingFace model cache

# python sample_from_llm.py --model_name "saves/qwen3_4B/sft_omr_1k_seed0/checkpoint-625" --input_path "omr_filtered_10K_w_custom_3_1.jsonl" --output_path "omr_filtered_10K_w_custom_3_2.jsonl" --batch_size 10000 --gpus "0,1,2,3"
# # python judge_qwen_math.py
# python sample_from_llm.py --model_name "saves/qwen3_4B/sft_omr_1k_seed88/checkpoint-625" --input_path "omr_filtered_10K_w_custom_3_2.jsonl" --output_path "omr_filtered_10K_w_custom_3_3.jsonl" --batch_size 10000 --gpus "0,1,2,3"

# python sample_from_llm.py --model_name "saves/qwen3_8B/sft_omr_1k/checkpoint-625" --input_path "omr_filtered_10K_w_custom_3_3.jsonl" --output_path "omr_filtered_10K_w_custom_3_4.jsonl" --batch_size 10000 --gpus "0,1,2,3"
# python sample_from_llm.py --model_name "saves/qwen3_8B/sft_omr_skillmix_1k_temp1/checkpoint-625" --input_path "omr_filtered_10K_w_custom_3_4.jsonl" --output_path "omr_filtered_10K_w_custom_3_5.jsonl" --batch_size 10000 --gpus "0,1,2,3"
# python sample_from_llm.py --model_name "saves/qwen3_8B/sft_omr_skillmix_1k_temp1_w_skillmix/checkpoint-625" --input_path "omr_filtered_10K_w_custom_3_5.jsonl" --output_path "omr_filtered_10K_w_custom_3_6.jsonl" --batch_size 10000 --gpus "0,1,2,3"

# python sample_from_llm.py --model_name "saves/qwen3_4B/sft_omr_1k_w_skillmix/checkpoint-625" --input_path "omr_filtered_10K_w_custom_3_6.jsonl" --output_path "omr_filtered_10K_w_custom_3_7.jsonl" --batch_size 10000 --gpus "0,1,2,3"
# python sample_from_llm.py --model_name "saves/qwen3_4B/sft_omr_1k_w_skillmix_seed0/checkpoint-625" --input_path "omr_filtered_10K_w_custom_3_7.jsonl" --output_path "omr_filtered_10K_w_custom_3_8.jsonl" --batch_size 10000 --gpus "0,1,2,3"
# python sample_from_llm.py --model_name "saves/qwen3_4B/sft_omr_1k_w_skillmix_seed88/checkpoint-625" --input_path "omr_filtered_10K_w_custom_3_8.jsonl" --output_path "omr_filtered_10K_w_custom_3_9.jsonl" --batch_size 10000 --gpus "0,1,2,3"

# python sample_from_llm.py --model_name "saves/qwen3_8B/sft_omr_1k_w_skillmix/checkpoint-625" --input_path "omr_filtered_10K_w_custom_3_9.jsonl" --output_path "omr_filtered_10K_w_custom_3_10.jsonl" --batch_size 10000 --gpus "0,1,2,3"
# python sample_from_llm.py --model_name "saves/qwen3_8B/sft_omr_1k_w_skillmix_seed0/checkpoint-625" --input_path "omr_filtered_10K_w_custom_3_10.jsonl" --output_path "omr_filtered_10K_w_custom_3_11.jsonl" --batch_size 10000 --gpus "0,1,2,3"
# python sample_from_llm.py --model_name "saves/qwen3_8B/sft_omr_1k_w_skillmix_seed88/checkpoint-625" --input_path "omr_filtered_10K_w_custom_3_11.jsonl" --output_path "omr_filtered_10K_w_custom_3_12.jsonl" --batch_size 10000 --gpus "0,1,2,3"


# python sample_from_llm.py --model_name "saves/qwen3_4B/sft_omr_1k_w_skillmix_seed0/checkpoint-625" --input_path "tree_data/skill_mix/math500_mapped_qa_chains.jsonl" --output_path "tree_data/skill_mix/math500_mapped_qa_chains_2.jsonl" --batch_size 10000 --gpus "0,1,2,3"
# python sample_from_llm.py --model_name "saves/qwen3_4B/sft_omr_1k_seed0/checkpoint-625" --input_path "tree_data/skill_mix/math500_mapped_qa_chains_2.jsonl" --output_path "tree_data/skill_mix/math500_mapped_qa_chains_3.jsonl" --batch_size 10000 --gpus "0,1,2,3"
# python sample_from_llm.py --model_name "Qwen/Qwen3-4B" --input_path "tree_data/skill_mix/math500_mapped_qa_chains_2.jsonl" --output_path "tree_data/skill_mix/math500_mapped_qa_chains_3.jsonl" --batch_size 10000 --gpus "0,1,2,3"
# python sample_from_llm.py --model_name "saves/qwen3_4B/sft_omr_1k_w_skillmix_temp1/checkpoint-625" --input_path "tree_data/skill_mix/math500_mapped_qa_chains_3.jsonl" --output_path "tree_data/skill_mix/math500_mapped_qa_chains_4.jsonl" --batch_size 10000 --gpus "0,1,2,3"
# python sample_from_llm.py --model_name "saves/qwen3_4B/sft_skillmix_omr_1k_temp1_w_skillmix/checkpoint-625" --input_path "tree_data/skill_mix/math500_mapped_qa_chains_4.jsonl" --output_path "tree_data/skill_mix/math500_mapped_qa_chains_5.jsonl" --batch_size 10000 --gpus "0,1,2,3"

python sample_from_llm.py --model_name "saves/qwen3_8B/sft_omr_1k_w_skillmix_seed0/checkpoint-625" --input_path "tree_data/skill_mix/math500_mapped_qa_chains.jsonl" --output_path "tree_data/skill_mix/math500_mapped_qa_chains_2.jsonl" --batch_size 10000 --gpus "0,1,2,3"
python sample_from_llm.py --model_name "saves/qwen3_8B/sft_omr_1k_seed0/checkpoint-625" --input_path "tree_data/skill_mix/math500_mapped_qa_chains_2.jsonl" --output_path "tree_data/skill_mix/math500_mapped_qa_chains_3.jsonl" --batch_size 10000 --gpus "0,1,2,3"
python sample_from_llm.py --model_name "Qwen/Qwen3-8B" --input_path "tree_data/skill_mix/math500_mapped_qa_chains_2.jsonl" --output_path "tree_data/skill_mix/math500_mapped_qa_chains_3.jsonl" --batch_size 10000 --gpus "0,1,2,3"
python sample_from_llm.py --model_name "saves/qwen3_8B/sft_omr_skillmix_1k_temp1/checkpoint-625" --input_path "tree_data/skill_mix/math500_mapped_qa_chains_3.jsonl" --output_path "tree_data/skill_mix/math500_mapped_qa_chains_4.jsonl" --batch_size 10000 --gpus "0,1,2,3"
python sample_from_llm.py --model_name "saves/qwen3_8B/sft_omr_skillmix_1k_temp1_w_skillmix/checkpoint-625" --input_path "tree_data/skill_mix/math500_mapped_qa_chains_4.jsonl" --output_path "tree_data/skill_mix/math500_mapped_qa_chains_5.jsonl" --batch_size 10000 --gpus "0,1,2,3"
