import os
os.sys.path.append("../")
from lib.utils import math_equal

from vllm import LLM, SamplingParams
import json
import re
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Judge math answers using Qwen model')
    parser.add_argument('--attempt_model', type=str, default="Qwen/Qwen2.5-1.5B-Instruct",
                       help='Name or path of the model to evaluate')
    parser.add_argument('--attempt_path', type=str, default="mapped_qa_chains_attempt_2.jsonl",
                        help='Path of the attempt model output')
    parser.add_argument('--judge_model', type=str, default="Qwen/Qwen2.5-7B-Instruct",
                       help='Name or path of the judge model')
    parser.add_argument('--batch_size', type=int, default=1000,
                       help='Batch size for processing')   
    return parser.parse_args()

# 提取 boxed{} 内容
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

def build_judge_prompt(question, solution, model_answer):
    return f"""<|im_start|>system
You are a helpful assistant and a strict math grader. Given a student's final answer and a reference solution, you must decide if the student's answer is correct. Your decision must be based on the reference solution, so if the reference solution is incomplete or unclear, you must answer "unknown".
<|im_end|>
<|im_start|>user
Question:

{question}

Reference solution:

{solution}

Student's answer:

{model_answer}

Is the student's final answer correct? Answer with only one word: "yes", "no", or "unknown".
<|im_end|>
<|im_start|>assistant
"""

### ---- 配置 ---- ###
args = parse_args()
attempt_field = f"{args.attempt_model}_attempt"
model_attempt_path = args.attempt_path
output_path = model_attempt_path.replace(".jsonl", "_evaluated.jsonl")
judge_model_name = args.judge_model
batch_size = args.batch_size

if "s1" in args.attempt_path:

    ### ---- 加载并合并数据集 ---- ###
    print("Loading original dataset...")
    input_data = load_dataset("simplescaling/data_ablation_full59K")["train"]
    input_data = input_data.map(lambda x, i: {"unique_id": f"s1_59K_{i}"}, with_indices=True)
    input_data = input_data.filter(lambda x: x["cot_type"] == "math")
    input_df = input_data.to_pandas()[["unique_id", "solution"]]

    print("Loading model attempt outputs...")
    model_outputs = []
    with open(model_attempt_path, "r") as f:
        for line in f:
            model_outputs.append(json.loads(line))
    output_df = pd.DataFrame(model_outputs)

    print("Merging by unique_id...")
    merged_df = pd.merge(output_df, input_df, on="unique_id", how="inner")
    records = merged_df.to_dict(orient="records")

    ### ---- 准备评审模型 ---- ###
    print(f"Loading LLM-as-judge model: {judge_model_name}")
    judge_model = LLM(judge_model_name)
    sampling_params = SamplingParams(temperature=0.3, max_tokens=1024)



    ### ---- 批量判断并写入 ---- ###
    print("Running judgment...")
    with open(output_path, "w", encoding="utf-8") as fout:
        for i in tqdm(range(0, len(records), batch_size), desc="Judging"):
            batch = records[i:i+batch_size]
            prompts = []
            for item in batch:
                model_answer = extract_boxed_answer(item.get(attempt_field, ""))
                prompt = build_judge_prompt(item["question"], item["solution"], model_answer)
                prompts.append(prompt)

            outputs = judge_model.generate(prompts, sampling_params=sampling_params)
            completions = [o.outputs[0].text.strip().lower() for o in outputs]

            for item, verdict in zip(batch, completions):
                if "yes" in verdict:
                    verdict = "yes"
                elif "no" in verdict:
                    verdict = "no"
                else:
                    verdict = "unknown"
                item[f"{args.attempt_model}_judge"] = verdict
                fout.write(json.dumps(item, ensure_ascii=False) + "\n")
else:
    records = []
    output_df = pd.read_json(model_attempt_path, lines=True)
    import swifter
    from tqdm.auto import tqdm
    tqdm.pandas()  # 注册tqdm到pandas

    # 加速并显示进度条
    output_df[attempt_field+"_extracted"] = output_df[attempt_field].apply(extract_boxed_answer)
    output_df[f"{args.attempt_model}_judge"] = output_df.progress_apply(
        lambda row: math_equal(row["expected_answer"], row[attempt_field+"_extracted"]), axis=1
    )
    output_df.to_json(output_path, orient="records", lines=True)


print(f"Finished. Results saved to: {output_path}")
