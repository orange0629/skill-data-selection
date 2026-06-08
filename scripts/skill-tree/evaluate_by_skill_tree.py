import os
os.sys.path.append("../")
from lib.utils import math_equal

import json
import random
import argparse
import pandas as pd
from tqdm import tqdm
from vllm import LLM, SamplingParams

from scipy.stats import norm
from math import sqrt


def parse_args():
    parser = argparse.ArgumentParser(description="Skill Tree Evaluation with vLLM")
    parser.add_argument('--skill_tree_path', type=str, required=True, help='Path to skill tree JSON file')
    parser.add_argument('--qa_jsonl_path', type=str, required=True, help='Path to QA JSONL file with question and answer')
    parser.add_argument('--mapping_jsonl_path', type=str, required=True, help='Path to JSONL with mapped_chains')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for results')
    parser.add_argument('--model_name', type=str, required=True, help='Model name or path to load in vLLM')
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--min_samples', type=int, default=30)
    parser.add_argument('--max_samples', type=int, default=200)
    parser.add_argument('--ci_tol', type=float, default=0.05)
    parser.add_argument('--alpha', type=float, default=0.05)
    parser.add_argument('--temperature', type=float, default=0.6)
    parser.add_argument('--max_tokens', type=int, default=10240)
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


def rebuild_tree(node):
    return {
        "name": node["name"],
        "children": {child["name"]: rebuild_tree(child) for child in node.get("children", [])},
    }


def iter_nodes(node, path=None):
    if path is None:
        path = []
    cur_path = path + [node["name"]]
    if not node.get("children"):
        yield " → ".join(cur_path), node
    for child in node.get("children", {}).values():
        yield from iter_nodes(child, cur_path)


def sanitize_filename(s: str) -> str:
    for ch in '<>:"/\\|?*':
        s = s.replace(ch, "_")
    return s


def wilson_ci(p_hat, n, alpha):
    z = norm.ppf(1 - alpha / 2)
    denom = 1 + (z**2)/n
    center = (p_hat + z**2 / (2*n)) / denom
    half = z * sqrt((p_hat * (1 - p_hat) / n + z**2 / (4*n**2))) / denom
    return center - half, center + half


class VLLMLocalEngine:
    def __init__(self, model_name, temperature=0.6, max_tokens=10240):
        self.llm = LLM(model=model_name)
        self.sampling_params = SamplingParams(temperature=temperature, max_tokens=max_tokens)

    def generate_batch(self, prompts):
        outputs = self.llm.generate(prompts, self.sampling_params)
        return [o.outputs[0].text.strip() if o.outputs else "" for o in outputs]


def evaluate_skill_node(
    node_path, uid_pool, qa_df, prompt_template, output_dir,
    llm_engine, batch_size=10, min_samples=30, max_samples=200,
    ci_tol=0.05, alpha=0.05, seed=42
):
    rng = random.Random(seed)
    total, correct = 0, 0
    evaluated = set()
    details = []

    available_uids = [uid for uid in uid_pool if uid in set(qa_df["unique_id"])]
    rng.shuffle(available_uids)

    def converged():
        if total < min_samples:
            return False
        acc = correct / total
        lo, hi = wilson_ci(acc, total, alpha)
        return hi - lo <= ci_tol

    while not converged() and total < max_samples and available_uids:
        current_batch = []
        current_rows = []

        for _ in range(min(batch_size, len(available_uids))):
            uid = available_uids.pop()
            if uid in evaluated:
                continue
            row = qa_df[qa_df["unique_id"] == uid].iloc[0]
            prompt = prompt_template.format(question=row["question"])
            current_batch.append(prompt)
            current_rows.append(row)
            evaluated.add(uid)

        predictions = llm_engine.generate_batch(current_batch)

        for row, pred in zip(current_rows, predictions):
            uid = row["unique_id"]
            gold = row.get("answer") or row.get("gold")
            is_correct = math_equal(pred, gold)
            total += 1
            correct += int(is_correct)
            details.append({
                "uid": uid,
                "question": row["question"],
                "prediction": pred,
                "gold": gold,
                "correct": is_correct
            })

    # Save details
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, f"{sanitize_filename(node_path)}.jsonl"), "w") as f:
        for d in details:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    acc = correct / total if total > 0 else None
    lo, hi = wilson_ci(acc, total, alpha) if total > 0 else (None, None)
    return {
        "node_path": node_path,
        "num_total": total,
        "num_correct": correct,
        "accuracy": acc,
        "ci_low": lo,
        "ci_high": hi
    }


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Load
    with open(args.skill_tree_path, "r") as f:
        tree_root = rebuild_tree(json.load(f))
    qa_df = pd.read_json(args.qa_jsonl_path, lines=True)
    mapping_df = pd.read_json(args.mapping_jsonl_path, lines=True)

    # Build skill -> uids
    uid_to_chains = dict(zip(mapping_df["question"], mapping_df["mapped_chains"]))
    path_to_uids = {}

    for _, row in qa_df.iterrows():
        q = row["question"]
        uid = row["unique_id"]
        chains = uid_to_chains.get(q, [])
        for chain in chains:
            path = [tree_root["name"]]
            path_to_uids.setdefault(" → ".join(path), set()).add(uid)
            cur = tree_root
            for node in chain:
                if node not in cur.get("children", {}):
                    break
                cur = cur["children"][node]
                path.append(node)
                path_to_uids.setdefault(" → ".join(path), set()).add(uid)

    skill_to_uids = {k: list(v) for k, v in path_to_uids.items()}

    # Initialize model
    llm_engine = VLLMLocalEngine(args.model_name, temperature=args.temperature, max_tokens=args.max_tokens)

    prompt_template = "Answer the following math problem:\n\n{question}\n\nAnswer:"

    # Evaluate each skill node
    summary = []
    per_node_dir = os.path.join(args.output_dir, "per_node")
    for path, _ in tqdm(iter_nodes(tree_root), desc="Evaluating skill nodes"):
        if path not in skill_to_uids or len(skill_to_uids[path]) == 0:
            continue
        result = evaluate_skill_node(
            node_path=path,
            uid_pool=skill_to_uids[path],
            qa_df=qa_df,
            prompt_template=prompt_template,
            output_dir=per_node_dir,
            llm_engine=llm_engine,
            batch_size=args.batch_size,
            min_samples=args.min_samples,
            max_samples=args.max_samples,
            ci_tol=args.ci_tol,
            alpha=args.alpha,
            seed=args.seed,
        )
        summary.append(result)

    pd.DataFrame(summary).to_csv(os.path.join(args.output_dir, "summary.csv"), index=False)
    print(f"✅ Evaluation completed. Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
