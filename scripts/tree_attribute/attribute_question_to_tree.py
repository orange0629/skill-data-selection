import os
# Set HF_HOME via environment variable before running, e.g.:
#   export HF_HOME=/path/to/your/model/cache

import json
import jsonlines
from tqdm import tqdm
import multiprocessing
from modeling import vllm_server_model
from functools import partial
import argparse

# Load and rebuild tree from structured JSON
def rebuild_tree(node):
    return {
        "name": node["name"],
        "children": {child["name"]: rebuild_tree(child) for child in node.get("children", [])}
    }

# Per-instance Q&A mapper
def map_qa_to_chains(tree_root, qa_input, model, max_depth=20):
    """
    Map a Q&A input to multiple skill chains by expanding all relevant branches.
    """
    all_chains = []

    def dfs(current_node, current_chain, depth):
        if depth >= max_depth:
            all_chains.append(current_chain)
            return

        children = current_node.get("children", {})
        if not children:
            all_chains.append(current_chain)
            return

        child_names = list(children.keys())

        prompt = f"""
Given the following Math problem:

Q&A: {qa_input}

Which of the following skills are involved to understanding or solving the problem? Even the most basic skills such as simple addition and subtraction must be taken into account. You can select multiple options if needed. Just return a list of skill names.

Skills:
{chr(10).join([f"- {name}" for name in child_names])}

Answer as a Python list of strings.
"""
        response = model.generate(prompt, temperature=0.3).strip()
        try:
            selected_skills = eval(response[response.find('['):response.rfind(']') + 1])
        except Exception as e:
            return

        matched = [s for s in selected_skills if s in children]

        if not matched:
            # all_chains.append(current_chain)
            return

        for skill in matched:
            dfs(children[skill], current_chain + [skill], depth + 1)

    dfs(tree_root, [], 0)
    return all_chains



def process_instance(obj, tree_root_dict, model):
    try:
        qa_input = obj.get("question", "") + "\n" + obj.get("attempt", "").split("</think>")[-1][:20000]
        chains = map_qa_to_chains(tree_root_dict, qa_input, model)
        return {
            "unique_id": obj.get("unique_id"),
            "mapped_chains": chains,
            "question": obj.get("question", ""),
            "attempt": obj.get("attempt", "")
        }
    except Exception as e:
        return {
            "unique_id": obj.get("unique_id"),
            "error": str(e)
        }

def load_cached_ids(output_path):
    cached_ids = set()
    if os.path.exists(output_path):
        with jsonlines.open(output_path, mode="r") as reader:
            for obj in reader:
                uid = obj.get("unique_id")
                if uid:
                    cached_ids.add(uid)
    return cached_ids


# 顶层函数：不能是 lambda 或嵌套函数
def process_wrapper(obj, tree_root, model):
    return process_instance(obj, tree_root, model)

def run_parallel_mapping(input_type, output_path, skill_tree_path, model, num_proc=4):
    # Step 1: Load tree
    with open(skill_tree_path, "r") as f:
        visual_tree = json.load(f)
    tree_root = rebuild_tree(visual_tree)

    # Step 2: Read input Q&A
    input_data = []
    if input_type == "s1_59K":
        from datasets import load_dataset
        input_data = load_dataset("simplescaling/data_ablation_full59K")["train"]
        input_data = input_data.map(lambda example, idx: {"unique_id": f"s1_59K_{idx}"}, with_indices=True).filter(lambda example: example["cot_type"] == "math")
    else:
        with jsonlines.open(input_type) as reader:
            for i, obj in enumerate(reader):
                input_data.append(obj)

    # Step 3: Load existing cache
    cached_ids = load_cached_ids(output_path)
    print(f"Found {len(cached_ids)} cached entries. Skipping these...")

    # Step 4: Filter input
    input_data = [obj for obj in input_data if obj.get("unique_id") not in cached_ids]

    # Step 5: Streamed + parallel process
    print(f"Launching {num_proc} workers on {len(input_data)} new examples...")

    # Partial function to include fixed arguments
    wrapped_func = partial(process_wrapper, tree_root=tree_root, model=model)

    with multiprocessing.Pool(num_proc) as pool, jsonlines.open(output_path, mode='a') as writer:
        for result in tqdm(pool.imap(wrapped_func, input_data), total=len(input_data)):
            writer.write(result)

    print(f"Finished. Appended {len(input_data)} new chains to {output_path}")


if __name__ == "__main__":
    # argparse as input
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_type", type=str, default="s1_59K")
    parser.add_argument("--output_path", type=str, default="tree_data/skill_mix/mapped_qa_chains.jsonl")
    parser.add_argument("--skill_tree_path", type=str, default="tree_data/skill_mix/skill_mix.json")
    parser.add_argument("--api_url", type=str, default="http://gl1518.arc-ts.umich.edu:2341/v1/chat/completions")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-32B-Instruct")
    parser.add_argument("--num_proc", type=int, default=4)
    args = parser.parse_args()

    model = vllm_server_model(api_url=args.api_url,
                              model_name=args.model_name)
    
    run_parallel_mapping(
        input_type=args.input_type,
        output_path=args.output_path,
        skill_tree_path=args.skill_tree_path,
        model=model,
        num_proc=args.num_proc,
    )