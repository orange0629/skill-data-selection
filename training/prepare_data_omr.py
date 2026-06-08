from datasets import load_dataset
from tqdm import tqdm
import json

# train_file_path = "simplescaling/s1K-1.1"
# output_file_path = "./s1K-1.1.jsonl"
# train_file_path = "../scripts/tree_attribute/tree_data/skill_mix/omr_qwen3_temp1_correct0_forced/mapped_qa_chains_attempt_evaluated_1K.jsonl"
# output_file_path = train_file_path.replace(".jsonl", "_llama_factory.jsonl")
# llama_factory_name = "omr_qwen3_4b_skillmix_1K_temp1_correct0_forced"

train_file_path = "../scripts/tree_attribute/tree_data/evaltree/omr_r1_llama_8b_temp1/mapped_qa_chains_attempt_evaluated_1K.jsonl"
output_file_path = train_file_path.replace(".jsonl", "_llama_factory.jsonl")
llama_factory_name = "omr_r1_llama_8b_evaltree_1K_temp1"

try:
    dataset = load_dataset(train_file_path)
except:
    dataset = load_dataset("json", data_files=train_file_path)

# def apply_chat_template(example):
#     cot_prefix = "Please reason step by step, and put your final answer within \\boxed{}."
#     return {
#         "messages": [
#             #{"role": "system", "content": cot_prefix},
#             {"role": "user", "content": f"{cot_prefix}\n\nQuestion: {example['question']}"},
#             {"role": "assistant", "content": f"<|im_start|>think\n{example['deepseek_thinking_trajectory']}\n<|im_start|>answer\n{example['deepseek_attempt']}\n<|im_end|>"}
#         ]
#     }

def apply_chat_template(example):
    cot_prefix = "Please reason step by step, and put your final answer within \\boxed{}."
    return {
        "messages": [
            #{"role": "system", "content": cot_prefix},
            {"role": "user", "content": f"{cot_prefix}\n\nQuestion: {example['question']}"},
            {"role": "assistant", "content": f"{example['attempt']}"}
        ]
    }
dataset = dataset.map(apply_chat_template)

dataset["train"].to_json(output_file_path, lines=True)

# register this data
with open("../../360-LLaMA-Factory/data/dataset_info.json", "r") as f:
    dataset_info = json.load(f)

# Add new dataset info
dataset_info[llama_factory_name] = {
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
