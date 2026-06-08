from datasets import load_dataset
from tqdm import tqdm
import json
import pandas as pd

min_shown_depth = 0
max_shown_depth = 5
train_file_path = "../scripts/tree_attribute/tree_data/evaltree/omr_qwen3_temp1/mapped_qa_chains_attempt_evaluated_1K.jsonl"
skill_file_path = "../scripts/tree_attribute/tree_data/evaltree/omr_mapped_qa_chains.jsonl"
# output_file_path = train_file_path.replace(".jsonl", "_w_skillmix_shown_depth_0_1_llama_factory.jsonl")
output_file_path = train_file_path.replace(".jsonl", "_w_evaltree_depth_0_5_llama_factory.jsonl")
# full_tree_path = "../scripts/tree_attribute/tree_data/skill_mix/skill_mix.json"
full_tree_path = "../scripts/tree_attribute/tree_data/evaltree/evaltree_math_full.json"
is_show_tree = False
dataset_name = "omr_qwen3_4b_evaltree_1K_temp1_w_evaltree_depth_0_5"

# train_file_path = "../scripts/tree_attribute/tree_data/skill_mix/qwen7b/mapped_qa_chains_attempt_evaluated_1K.jsonl"
# skill_file_path = "../scripts/tree_attribute/tree_data/skill_mix/mapped_qa_chains.jsonl"
# output_file_path = "../../360-LLaMA-Factory/skillmix_qwen7b_1K_w_skillmix.jsonl"
# dataset_name = "skillmix_qwen7b_1K_w_skillmix"

try:
    dataset = load_dataset(train_file_path)
except:
    dataset = load_dataset("json", data_files=train_file_path)

dataset_df = dataset["train"].to_pandas()
skill_df = pd.read_json(skill_file_path, lines=True)

if is_show_tree:
    with open(full_tree_path, "r") as f:
        full_tree = json.load(f)

if "mapped_chains" in dataset_df.columns:
    dataset_df_w_skill = dataset_df.drop(columns=["mapped_chains"]).merge(skill_df[["question", "unique_id", "mapped_chains"]], on="question", how="inner")
else:
    dataset_df_w_skill = dataset_df.merge(skill_df[["question", "unique_id", "mapped_chains"]], on="question", how="inner")

# def apply_chat_template(example):
#     cot_prefix = "Please reason step by step, and put your final answer within \\boxed{}."
#     return {
#         "messages": [
#             #{"role": "system", "content": cot_prefix},
#             {"role": "user", "content": f"{cot_prefix}\n\nQuestion: {example['question']}"},
#             {"role": "assistant", "content": f"<|im_start|>think\n{example['deepseek_thinking_trajectory']}\n<|im_start|>answer\n{example['deepseek_attempt']}\n<|im_end|>"}
#         ]
#     }

# def apply_chat_template(example):
#     cot_prefix = "Please reason step by step, and put your final answer within \\boxed{}."
#     skill_prefix = "Skills:\n" + '\n'.join(['->'.join(sub_chain) for sub_chain in example['mapped_chains']])
#     return [
#             #{"role": "system", "content": cot_prefix},
#             {"role": "user", "content": f"{cot_prefix}\n\nQuestion: {example['question']}"},
#             {"role": "assistant", "content": f"<|im_start|>think\n{skill_prefix}\n{example['thinking_trajectories'][0]}\n<|im_start|>answer\n{example['attempt']}\n<|im_end|>"}
#         ]

# need to delete in the future
with open("../scripts/tree_attribute/tree_data/evaltree/capability_to_distinction.json", "r") as f:
    capability_to_distinction = json.load(f)

def apply_chat_template(example):
    skill_tree_str = ""
    if is_show_tree:
        # show the tree structure
        skill_tree_str = "Here is a skill tree:\n" + json.dumps(full_tree, indent=2) + "\n\n"
    cot_prefix = "Please reason step by step, and put your final answer within \\boxed{}."
    if max_shown_depth is not None and min_shown_depth is not None:
        if capability_to_distinction is not None:
            sub_chain_shown = ['->'.join([capability_to_distinction[tmp][0] if tmp in capability_to_distinction else tmp for tmp in sub_chain[min_shown_depth:max_shown_depth]]) for sub_chain in example['mapped_chains'] if len(sub_chain) >= max_shown_depth]
        else:
            sub_chain_shown = ['->'.join(sub_chain[min_shown_depth:max_shown_depth]) for sub_chain in example['mapped_chains'] if len(sub_chain) >= max_shown_depth]
        sub_chain_shown = list(set(sub_chain_shown))
        skill_prefix = "Skills:\n" + '\n'.join(sub_chain_shown)
    else:
        skill_prefix = "Skills:\n" + '\n'.join(['->'.join(sub_chain) for sub_chain in example['mapped_chains']])
    return [
            #{"role": "system", "content": cot_prefix},
            {"role": "user", "content": f"{skill_tree_str}{cot_prefix}\n\nQuestion: {example['question']}"},
            {"role": "assistant", "content": f"{skill_prefix}\n{example['attempt']}"}
        ]
dataset_df_w_skill["messages"] = dataset_df_w_skill.apply(apply_chat_template, axis=1)

dataset_df_w_skill.to_json(output_file_path, lines=True, orient="records")

print("Length of dataset_df_w_skill:", len(dataset_df_w_skill))

# register this data
with open("../../360-LLaMA-Factory/data/dataset_info.json", "r") as f:
    dataset_info = json.load(f)

# Add new dataset info
dataset_info[dataset_name] = {
    "file_name": output_file_path,
    "formatting": "sharegpt", 
    "columns": {
        "messages": "messages"
    },
    "tags": {
        "role_tag": "role",
        "content_tag": "content", 
        "user_tag": "user",
        "assistant_tag": "assistant"
    }
}

# Write back updated dataset_info.json
with open("data/dataset_info.json", "w") as f:
    json.dump(dataset_info, f, indent=2)
