import sys
sys.path.append("../lib")
import os
import argparse
from tqdm import tqdm
import multiprocessing
from vllm import LLM, SamplingParams
import json
from datasets import load_dataset, concatenate_datasets, DatasetDict, Dataset
import pandas as pd
from vllm.lora.request import LoRARequest
from datetime import datetime
from eval.math_equivalence import is_equiv
import re

def extract_boxed_answer(text):
    start_token = r"\boxed{"
    start_idx = text.rfind(start_token)
    if start_idx == -1:
        return ''
    i = start_idx + len(start_token)
    brace_depth = 1
    content = []
    while i < len(text):
        if text[i] == '{':
            brace_depth += 1
        elif text[i] == '}':
            brace_depth -= 1
            if brace_depth == 0:
                break
        content.append(text[i])
        i += 1
    return ''.join(content).strip() if brace_depth == 0 else ''


def apply_chat_template(example):
    cot_prefix = "Please reason step by step, and put your final answer within \\boxed{}."
    skill_tree_str = ""
    if args.input_skill_tree:
        # show the tree structure
        skill_tree_str = "Here is a skill tree:\n" + json.dumps(skill_tree, indent=2) + "\n\n"
    return {
        "input_text_only": [
            #{"role": "system", "content": cot_prefix},
            {"role": "user", "content": f"{skill_tree_str}{cot_prefix}\n\nQuestion: {example['Problem']}"},
        ]
    }


def main(args):
    # 1. Read jsonl
    sample_times = 8
    if args.benchmark_name == "aime":
        bench_df = pd.read_parquet("aime2024/aime_2024_problems.parquet")
        question_list = list(bench_df["Problem"])
        true_label_list = list(bench_df["Answer"])
    elif args.benchmark_name == "aime25":
        bench_df = pd.read_parquet("aime2025/aime_2025_problems.parquet")
        question_list = list(bench_df["Problem"])
        true_label_list = list(bench_df["Answer"])
    elif args.benchmark_name == "amc23":
        bench_df = pd.read_parquet("amc23/amc23.parquet")
        question_list = list(bench_df["question"])
        true_label_list = list(bench_df["answer"])
    elif args.benchmark_name == "math500":
        sample_times = 1
        bench_df = pd.read_json("math500/math500.jsonl", lines=True)
        question_list = list(bench_df["problem"])
        true_label_list = list(bench_df["answer"])
    elif args.benchmark_name == "mathl5":
        bench_df = pd.read_json("mathl5/test.jsonl", lines=True)
        question_list = list(bench_df["question"])
        true_label_list = list(bench_df["answer"])
    elif args.benchmark_name == "gpqa":
        sample_times = 4
        bench_df = pd.read_parquet("gpqa/gpqa_diamond.parquet")
        question_list = list(bench_df["question"])
        true_label_list = list(bench_df["answer"])
    elif args.benchmark_name == "olympiadbench":
        sample_times = 1
        bench_df = pd.read_parquet("olympiadbench/olympiadbench.parquet")
        bench_df = bench_df[bench_df["is_multiple_answer"] == False]
        question_list = list(bench_df["question"])
        true_label_list = list(bench_df["final_answer"])
        true_label_list = [a[0] for a in true_label_list]
    else:
        raise ValueError(f"Benchmark name {args.benchmark_name} not supported")

    if args.input_skill_tree:
        with open(args.input_skill_tree, "r") as f:
            skill_tree = json.load(f)

    if "qwen3" in args.model_dir.lower():
        from transformers import AutoTokenizer
        if "checkpoint-0" in args.model_dir:
            tokenizer = AutoTokenizer.from_pretrained(args.base_model)
        else:
            tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
        input_template = tokenizer.apply_chat_template(
            [
                {"role": "user", "content": "Please reason step by step, and put your final answer within \\boxed{}.\n\nQuestion: {user_prompt}"}
            ],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True  # True is the default value for enable_thinking
        )
        print(input_template)
    else:
        # input_template = "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\nPlease reason step by step, and put your final answer within \\boxed{}.\n\nQuestion: {user_prompt}<|im_end|>\n<|im_start|>assistant\n<|im_start|>think\n"
        from transformers import AutoTokenizer
        if "checkpoint-0" in args.model_dir:
            tokenizer = AutoTokenizer.from_pretrained(args.base_model)
        else:
            tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
        input_template = tokenizer.apply_chat_template(
            [
                {"role": "user", "content": "Please reason step by step, and put your final answer within \\boxed{}.\n\nQuestion: {user_prompt}"}
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        print(input_template)
    input_prompt_list = [input_template.replace("{user_prompt}", q) for q in question_list]

    # 初始化记录结构
    records = [{
        "question_id": idx,
        "question": q,
        "true_answer": a,
        "predictions": [],
        "boxed_answers": [],
        "is_correct_list": [],
        "outputs_length": [],
        "correct_count": 0,
        "pass@8": 0.0
    } for idx, (q, a) in enumerate(zip(question_list, true_label_list))]

    sampling_params = SamplingParams(max_tokens=16384, temperature=0.6, skip_special_tokens=False)
    if "checkpoint-0" in args.model_dir:
        os.makedirs(args.model_dir, exist_ok=True)
        llm = LLM(model=args.base_model, tensor_parallel_size=args.num_gpus)
    elif args.lora:
        llm = LLM(model=args.base_model, enable_lora=True, max_lora_rank=256, tensor_parallel_size=args.num_gpus)
    else:
        llm = LLM(model=args.model_dir, tensor_parallel_size=args.num_gpus)

    # 输出路径
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    result_path = os.path.join(args.model_dir, f"{args.benchmark_name}_records_{timestamp}.json")
    score_path = f"{args.benchmark_name}_score.txt"

    # 多轮采样
    for i in range(sample_times):
        print(f"Sampling {i+1}/{sample_times}...")
        vllm_outputs = llm.generate(input_prompt_list, sampling_params=sampling_params)
        outputs = [o.outputs[0].text for o in vllm_outputs]
        outputs_length = [len(o.outputs[0].token_ids) for o in vllm_outputs]

        for idx, output in enumerate(outputs):
            boxed = extract_boxed_answer(output)
            if args.benchmark_name == "gpqa":
                correct = False
                pattern = r'\b[A-Z]\b'
                match = re.search(pattern, boxed, re.MULTILINE)
                if match:
                    correct = (match.group(0) == true_label_list[idx])
            else:
                correct = is_equiv(str(true_label_list[idx]), boxed)
            rec = records[idx]
            rec["predictions"].append(output)
            rec["outputs_length"].append(outputs_length[idx])
            rec["boxed_answers"].append(boxed)
            rec["is_correct_list"].append(correct)
            rec["correct_count"] += int(correct)
            rec[f"pass@{sample_times}"] = rec["correct_count"] / (i + 1)

        # 每轮保存一次，覆盖旧文件
        pd.DataFrame(records).to_json(result_path, orient="records", indent=4)

        # 同时打印并覆盖 score 文件
        avg_score = sum(r[f"pass@{sample_times}"] for r in records) / len(records)
        avg_outputs_length = sum(sum(r["outputs_length"])/len(r["outputs_length"]) for r in records) / len(records)
        score_str = f"[{timestamp}] {args.model_dir} Iter {i+1}/{sample_times} Pass@{sample_times}: {avg_score:.4f} on {len(records)} problems, avg outputs length: {avg_outputs_length:.2f}"
        print(score_str)
        with open(score_path, "a") as f:
            f.write(score_str + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, help="Path to the base model checkpoint directory")
    parser.add_argument("--model_dir", type=str, help="Path to the model checkpoint directory")
    parser.add_argument('--lora', action = 'store_true', help = 'Use Huggingface Transformer', default=False)
    parser.add_argument('--num_gpus', type=int, default=1, help='Tensor parallelism degree')
    parser.add_argument('--benchmark_name', type=str, default="aime", help='Benchmark name')
    parser.add_argument('--input_skill_tree', type=str, default=None, help='Path to the skill tree file')
    args = parser.parse_args()
    main(args)