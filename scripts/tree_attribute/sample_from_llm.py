import os
os.sys.path.append("../")
from lib.utils import math_equal, extract_boxed_answer

import pandas as pd
import json
import argparse
from vllm import LLM, SamplingParams
from tqdm.auto import tqdm
tqdm.pandas() 
import multiprocessing as mp
import time


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate using Qwen model with VLLM')
    parser.add_argument('--model_name', type=str, default="Qwen/Qwen2.5-1.5B-Instruct",
                        help='Name or path of the model to use')
    parser.add_argument('--input_path', type=str, default="mapped_qa_chains.jsonl",
                        help='Path to input JSONL file')
    parser.add_argument('--output_path', type=str, default=None,
                        help='Path to output JSONL file. If not specified, will be derived from input path')
    parser.add_argument('--batch_size', type=int, default=1000,
                        help='Batch size for processing')
    parser.add_argument('--gpus', type=str, default="0,1,2,3")
    return parser.parse_args()


def worker(gpu_id, task_queue, result_queue, model_name):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    llm = LLM(model_name)
    sampling_params = SamplingParams(max_tokens=8192, temperature=0.6, skip_special_tokens=False)
    result_queue.put((gpu_id, "ready"))
    
    while True:
        task = task_queue.get()
        if task is None:
            break
        sub_batch_idx, sub_batch = task
        outputs = llm.generate(sub_batch, sampling_params)
        responses = [o.outputs[0].text for o in outputs]
        result_queue.put((sub_batch_idx, responses))


def main():
    args = parse_args()
    gpu_ids = [int(x) for x in args.gpus.split(',')]
    num_gpus = len(gpu_ids)
    print(f"Using {num_gpus} GPUs")
    # Set output path if not specified
    output_path = args.output_path
    if output_path is None:
        output_path = args.input_path.replace(".jsonl", f"_attempt.jsonl")

    task_queue = mp.Queue()
    result_queue = mp.Queue()
    workers = []

    for gpu_id in gpu_ids:
        p = mp.Process(target=worker, args=(gpu_id, task_queue, result_queue, args.model_name))
        p.start()
        workers.append(p)
        if gpu_id == gpu_ids[0]:
            time.sleep(480)
    
    # Wait for all workers to be ready
    for _ in gpu_ids:
        result_queue.get()

    # Prompt template
    if "qwen3" in args.model_name.lower():
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        input_template = tokenizer.apply_chat_template(
            [
                {"role": "user", "content": "Please reason step by step, and put your final answer within \\boxed{}.\n\nQuestion: {user_prompt}"}
            ],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True  # True is the default value for enable_thinking
        )
        print(input_template)
    elif "deepseek" in args.model_name.lower():
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        input_template = tokenizer.apply_chat_template(
            [
                {"role": "user", "content": "Please reason step by step, and put your final answer within \\boxed{}.\n\nQuestion: {user_prompt}"}
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        print(input_template)
    else:
        input_template = (
            "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant."
            "<|im_end|>\n<|im_start|>user\n{user_prompt}\n"
            "Please reason step by step, and put your final answer within \\boxed{}."
            "<|im_end|>\n<|im_start|>assistant\n"
        )

    # Read JSONL data
    with open(args.input_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]

    # Process in batches and write to new file
    with open(output_path, 'a', encoding='utf-8') as f_out:
        for i in tqdm(range(0, len(data), args.batch_size), desc="Processing"):
            batch = data[i:i+args.batch_size]
            input_prompts = [
                input_template.replace("{user_prompt}", item["question"]) for item in batch
            ]

            # Split into sub-batches
            sub_batch_size = args.batch_size // num_gpus
            sub_batches = [input_prompts[j:j+sub_batch_size] for j in range(0, len(input_prompts), sub_batch_size)]
            for sub_batch_idx, sub_batch in enumerate(sub_batches):
                task_queue.put((sub_batch_idx, sub_batch))
            
            results = []
            for _ in range(len(sub_batches)):
                results.append(result_queue.get())
            results.sort(key=lambda x: x[0])
            responses = []
            for x in results:
                responses += x[1]

            
            tmp_df = pd.DataFrame(batch)
            tmp_df[f"{args.model_name}_attempt"] = responses
            tmp_df[f"{args.model_name}_attempt_extracted"] = tmp_df[f"{args.model_name}_attempt"].apply(extract_boxed_answer)
            tmp_df[f"{args.model_name}_judge"] = tmp_df.progress_apply(
                lambda row: math_equal(row["expected_answer"], row[f"{args.model_name}_attempt_extracted"]), axis=1
            )

            # contiune writing to jsonl
            for _, row in tmp_df.iterrows():
                f_out.write(json.dumps(row.to_dict(), ensure_ascii=False) + '\n')

            # # Write JSON lines with model_name_attempt field
            # for item, response in zip(batch, responses):
            #     item[f"{args.model_name}_attempt"] = response
            #     f_out.write(json.dumps(item) + '\n')

    # Terminate workers
    for _ in range(num_gpus):
        task_queue.put(None)
    for p in workers:
        p.join()

if __name__ == "__main__":
    main()
