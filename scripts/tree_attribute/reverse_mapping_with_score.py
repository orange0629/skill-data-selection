import json
from collections import defaultdict
import jsonlines
import argparse
import os
import pandas as pd
import numpy as np
from collections import Counter
from tqdm import tqdm

parser = argparse.ArgumentParser(description="Skill tree Q&A mapping and visualization")
parser.add_argument('--skill_tree_path', type=str, default="tree_data/skill_mix.json", help='Path to skill tree JSON')
parser.add_argument('--qa_jsonl_path', type=str, default="mapped_qa_chains_attempt_evaluated.jsonl", help='Path to Q&A JSONL file')
parser.add_argument('--tree_mapping_path', type=str, default="tree_data/skill_mix/omr_mapped_qa_chains.jsonl", help='Path to skill tree mapping JSONL file')
parser.add_argument('--num_instances_to_show', type=int, default=100000, help='Number of Q&A instances to process')
parser.add_argument('--output_path', type=str, required=True, help='Directory to store all output files')
parser.add_argument('--sample_size', type=int, default=1000, help='Number of weak instances to sample')
parser.add_argument('--attempt_model', type=str, default="Qwen/Qwen2.5-Math-7B-Instruct", help='Attempt model name')
parser.add_argument('--temperature', type=float, default=0.5, help='Temperature for sampling')
parser.add_argument('--correct_portion', type=float, default=None, help='Portion of correct instances to sample')
parser.add_argument('--sample_particular_skill', type=str, default=None, help='Particular skill to sample')
parser.add_argument('--filtered_instance_file', type=str, default=None, help='File to save filtered instance uids')
parser.add_argument('--max_depth', type=int, default=None, help='Max depth of the tree')
parser.add_argument('--forced_distribution', action='store_true', help='Use skill accuracy-based distribution to sample questions')
args = parser.parse_args()

os.makedirs(args.output_path, exist_ok=True)
output_tree_path = os.path.join(args.output_path, "skill_tree_with_accuracy_and_verdicts.json")
output_html_path = os.path.join(args.output_path, "skill_tree_visualization.html")
output_weak_jsonl_path = os.path.join(args.output_path, f"mapped_qa_chains_attempt_evaluated_{args.sample_size//1000}K.jsonl")

# Load and rebuild tree from structured JSON
def rebuild_tree(node, max_depth=None):
    if max_depth is not None and max_depth <= 0:
        print(f"Max depth reached for {node['name']}")
    return {
        "name": node["name"],
        "children": {child["name"]: rebuild_tree(child, max_depth=max_depth-1 if max_depth is not None else None) for child in node.get("children", []) if max_depth is None or max_depth > 0},
        "related_instances": set()
    }

with open(args.skill_tree_path, "r") as f:
    visual_tree = json.load(f)
tree_root = rebuild_tree(visual_tree, max_depth=args.max_depth)

filtered_file_ids = set()
if args.filtered_instance_file is not None:
    filtered_file_df = pd.read_json(args.filtered_instance_file, lines=True)
    filtered_file_ids = set(filtered_file_df["unique_id"])
    print(f"Filtered {len(filtered_file_ids)} instances")

tree_mapping_df = pd.read_json(args.tree_mapping_path, lines=True)
tmp_df = pd.read_json(args.qa_jsonl_path, lines=True)
tree_mapping_df = tmp_df.merge(tree_mapping_df[["question", "mapped_chains"]], on="question", how="left")
tree_mapping_df.to_json(args.tree_mapping_path+"_tmp", orient="records", lines=True)
tree_mapping_relation = {u: v for u, v in zip(tree_mapping_df["unique_id"], tree_mapping_df["mapped_chains"])}

# 读取 jsonl 文件并构建映射
jsonl_path = args.qa_jsonl_path
qa_data = []
with open(jsonl_path, 'r', encoding='utf-8') as f:
    for line in f:
        item = json.loads(line)
        uid = item["unique_id"]
        item["mapped_chains"] = tree_mapping_relation[uid]
        if uid in filtered_file_ids:
            continue
        qa_data.append(item)
        all_chains = item["mapped_chains"]
        for chain in all_chains:
            current_node = tree_root
            current_node["related_instances"].add(uid)
            for node_name in chain:
                children = current_node.get("children", {})
                if node_name not in children:
                    break
                current_node = children[node_name]
                # 添加到 related_instances 中
                current_node["related_instances"].add(uid)

# 获取 skill_path → related_instance_uids 的映射
def collect_skill_instance_mapping(node, path_prefix=[]):
    results = {}
    full_path = path_prefix + [node["name"]]
    path_str = " → ".join(full_path)

    if node.get("related_instances"):
        results[path_str] = list(node["related_instances"])  # 转成 list 便于 JSON 化

    for child in node.get("children", {}).values():
        results.update(collect_skill_instance_mapping(child, full_path))

    return results

skill_to_instances = collect_skill_instance_mapping(tree_root)

def convert_sets_to_lists(node):
    if isinstance(node, dict):
        for key, value in node.items():
            if key == "related_instances" and isinstance(value, set):
                node[key] = list(value)
            else:
                convert_sets_to_lists(value)
    elif isinstance(node, list):
        for item in node:
            convert_sets_to_lists(item)
convert_sets_to_lists(tree_root)

# Build verdict mapping
verdict_map = {item["unique_id"]: item[f"{args.attempt_model}_judge"] for item in qa_data}

# 更新树节点：添加 verdict_dict 和 accuracy
def compute_accuracy_and_verdicts(node):
    verdict_dict = {}
    for uid in node.get("related_instances", []):
        if uid in verdict_map:
            verdict_dict[uid] = verdict_map[uid]
    yes = sum(1 for v in verdict_dict.values() if v == True)
    no = sum(1 for v in verdict_dict.values() if v == False)
    total = yes + no

    node["verdict_accuracy"] = yes / total if total > 0 else None
    node["verdict_dict"] = verdict_dict
    node["verdict_num_yes"] = yes
    node["verdict_num_total"] = total

    for child in node.get("children", {}).values():
        compute_accuracy_and_verdicts(child)

compute_accuracy_and_verdicts(tree_root)

def get_leaf_skill_accuracy_map(tree_node, path_prefix=[], acc_map=None):
    if acc_map is None:
        acc_map = {}
    path = path_prefix + [tree_node["name"]]
    path_str = " → ".join(path)

    # only collect if leaf
    if not tree_node.get("children") and tree_node.get("verdict_accuracy") is not None:
        acc_map[path_str] = tree_node["verdict_accuracy"]

    for child in tree_node.get("children", {}).values():
        get_leaf_skill_accuracy_map(child, path, acc_map)

    return acc_map

def build_skill_question_matrix(qa_data, skill_list):
    skill_to_index = {s: i for i, s in enumerate(skill_list)}
    M = np.zeros((len(skill_list), len(qa_data)))

    for q_idx, item in enumerate(qa_data):
        for chain in item["mapped_chains"]:
            full_path = " → ".join(chain)
            if full_path in skill_list:
                M[skill_to_index[full_path], q_idx] = 1
    return M

def compute_inverse_accuracy_distribution(skill_accuracy_map, skill_list, temperature):
    accs = []
    for s in skill_list:
        acc = skill_accuracy_map.get(s, 0.0)
        if acc > 0:
            weight = min((1 / acc) ** temperature, 1000 ** temperature)
        else:
            weight = 1000 ** temperature
        accs.append(weight)
    accs = np.array(accs)
    return accs / accs.sum()

def compute_actual_distribution(sampled_uids, qa_data, skill_list):
    skill_counter = Counter()
    uid_set = set(sampled_uids)
    for item in qa_data:
        if item["unique_id"] not in uid_set:
            continue
        for chain in item["mapped_chains"]:
            path = " → ".join(chain)
            if path in skill_list:
                skill_counter[path] += 1
    total = sum(skill_counter.values())
    return {k: v / total for k, v in skill_counter.items()} if total > 0 else {}


# 保存更新后的树
with open(output_tree_path, "w", encoding="utf-8") as f:
    json.dump(tree_root, f, ensure_ascii=False, indent=4)

import random
random.seed(42)

def collect_leaf_node_with_wrong_items(node):
    if not node.get("children"):
        verdict_dict = node.get("verdict_dict", {})
        acc = node.get("verdict_accuracy")
        if acc is None:
            return []
        wrong_ids = [uid for uid, verdict in verdict_dict.items() if verdict == False]
        correct_ids = [uid for uid, verdict in verdict_dict.items() if verdict == True]
        all_ids = [uid for uid in verdict_dict.keys()]
        if args.correct_portion is not None:
            num_total_correct = len(correct_ids) // args.correct_portion if args.correct_portion > 0 else float('inf')
            num_total_incorrect = len(wrong_ids) // (1 - args.correct_portion) if args.correct_portion < 1 else float('inf')
            min_num_total = int(min(num_total_correct, num_total_incorrect))
            sampled_correct = random.sample(correct_ids, int(min_num_total * args.correct_portion))
            sampled_wrong = random.sample(wrong_ids, int(min_num_total * (1 - args.correct_portion)))
            all_ids = sampled_correct + sampled_wrong
            random.shuffle(all_ids)
        # weight = 1.0 - acc  # 用错误率作为权重
        weight = min((1/acc) ** args.temperature, 1000 ** args.temperature) if acc > 0 else 1000 ** args.temperature
        print(f"Node '{node['name']}' (accuracy: {acc:.3f}): sampling weight = {weight:.2f}")
        # return [(wrong_ids, weight)]
        if args.sample_particular_skill is not None:
            if args.sample_particular_skill == node["name"]:
                return [(all_ids, weight)]
            else:
                return []
        return [(all_ids, weight)]

    results = []
    for child in node["children"].values():
        results.extend(collect_leaf_node_with_wrong_items(child))
    return results

def sample_weak_instances_per_node(tree_root, sample_size=1000):
    node_pool = collect_leaf_node_with_wrong_items(tree_root)
    if not node_pool:
        return []

    sampled = set()
    max_iter = 1000000
    while len(sampled) < sample_size and max_iter > 0:
        max_iter -= 1
        uids_lists, weights = zip(*node_pool)
        chosen_node_idx = random.choices(range(len(node_pool)), weights=weights, k=1)[0]
        candidates = uids_lists[chosen_node_idx]
        if not candidates:
            continue
        uid = random.choice(candidates)
        if uid not in sampled:
            sampled.add(uid)
        if len(sampled) >= sample_size:
            break

    return list(sampled)

def greedy_projection(M, target_dist, sample_size, verdict_by_index=None,
                      desired_correct=None, desired_wrong=None):
    N = M.shape[1]
    selected = set()
    skill_count = np.zeros(M.shape[0], dtype=float)

    need_quota = (desired_correct is not None) and (desired_wrong is not None)
    cur_correct = 0
    cur_wrong = 0

    for _ in range(sample_size):
        best_q = None
        best_score = float('inf')

        # Determine which class is still allowed (if quotas are on)
        prefer_correct_only = False
        prefer_wrong_only = False
        if need_quota:
            if cur_correct >= desired_correct and cur_wrong >= desired_wrong:
                break  # quotas met
            elif cur_correct >= desired_correct:
                prefer_wrong_only = True
            elif cur_wrong >= desired_wrong:
                prefer_correct_only = True

        for j in range(N):
            if j in selected:
                continue
            if need_quota:
                v = verdict_by_index.get(j, None)
                if not isinstance(v, bool):
                    continue
                if prefer_correct_only and v is not True:
                    continue
                if prefer_wrong_only and v is not False:
                    continue

            add = M[:, j]
            if add.sum() == 0:
                continue
            proj = skill_count + add
            proj_norm = proj / proj.sum()
            score = np.linalg.norm(proj_norm - target_dist)

            if score < best_score:
                best_score = score
                best_q = j

        if best_q is None:
            print("[Info] No feasible candidate at this step; stopping early.")
            break

        selected.add(best_q)
        skill_count += M[:, best_q]
        if need_quota:
            if verdict_by_index[best_q] is True:
                cur_correct += 1
            else:
                cur_wrong += 1

    return selected


# 采样并保存
if args.forced_distribution:
    print(f"\n=== Matching Skill Distribution Mode ===")
    print(f"Using inverse accuracy as target distribution (leaf nodes only) with temperature = {args.temperature}")

    # Step 1: 获取叶节点 skill 的 accuracy
    skill_accuracy_map = {}
    for child in tree_root.get("children", {}).values():
        get_leaf_skill_accuracy_map(child, [], skill_accuracy_map)
    skill_list = sorted(skill_accuracy_map.keys())

    # Step 2: 构建 skill-question 矩阵
    M = build_skill_question_matrix(qa_data, skill_list)

    # Step 3: 构建目标分布
    target_dist = compute_inverse_accuracy_distribution(skill_accuracy_map, skill_list, args.temperature)

    # ---- Correctness quota setup for forced distribution ----
    # Map question index -> boolean verdict
    verdict_by_index = {j: verdict_map.get(item["unique_id"], None) for j, item in enumerate(qa_data)}

    # Keep only questions with a boolean verdict
    valid_indices = [j for j, v in verdict_by_index.items() if isinstance(v, bool)]
    if len(valid_indices) < args.sample_size:
        print(f"[Warn] Not enough judged items. Target={args.sample_size}, Available={len(valid_indices)}.")
        args.sample_size = len(valid_indices)

    if args.correct_portion is not None:
        desired_correct = int(round(args.sample_size * args.correct_portion))
        desired_wrong = args.sample_size - desired_correct

        correct_pool = [j for j in valid_indices if verdict_by_index[j] is True]
        wrong_pool   = [j for j in valid_indices if verdict_by_index[j] is False]

        # Clip quotas to availability
        desired_correct = min(desired_correct, len(correct_pool))
        desired_wrong   = min(desired_wrong, len(wrong_pool))

        # Adjust total if clipped
        capped_total = desired_correct + desired_wrong
        if capped_total < args.sample_size:
            print(f"[Warn] Quotas clipped by availability. Sample size reduced to {capped_total}.")
            args.sample_size = capped_total
    else:
        desired_correct = None
        desired_wrong = None

    # Step 4: 贪心采样
    selected_indices = greedy_projection(
        M,
        target_dist,
        sample_size=args.sample_size,
        verdict_by_index=verdict_by_index,
        desired_correct=desired_correct,
        desired_wrong=desired_wrong
    )
    sampled_weak_ids = [qa_data[i]["unique_id"] for i in selected_indices]
    # ---- Correctness ratio sanity check ----
    if args.correct_portion is not None and selected_indices:
        sel_correct = sum(verdict_by_index[i] is True for i in selected_indices)
        sel_total = len(selected_indices)
        print("\n=== Correctness Ratio Check ===")
        print(f"Target correct portion: {args.correct_portion:.3f}")
        print(f"Actual correct portion: {sel_correct/sel_total:.3f} ({sel_correct}/{sel_total})")

    # Step 5: 打印目标 vs 实际 skill 分布
    actual_dist = compute_actual_distribution(sampled_weak_ids, qa_data, skill_list)
    all_dist = compute_actual_distribution([item["unique_id"] for item in qa_data], qa_data, skill_list)
    print("\n=== Skill Distribution Comparison (Leaf Only) ===")
    print(f"{'Skill Path':60} | {'Target %':>9} | {'Actual %':>9} | {'Original %':>9}")
    print("-" * 85)
    for skill in skill_list:
        target_p = target_dist[skill_list.index(skill)] * 100
        actual_p = actual_dist.get(skill, 0.0) * 100
        original_p = all_dist.get(skill, 0.0) * 100
        print(f"{skill:60} | {target_p:9.2f} | {actual_p:9.2f} | {original_p:9.2f}")

else:
    print(f"\n=== Sampling Statistics ===")
    print(f"Temperature: {args.temperature}")
    print(f"Sample size: {args.sample_size}")
    if args.correct_portion is not None:
        print(f"Correct portion: {args.correct_portion}")
    if args.sample_particular_skill is not None:
        print(f"Sampling particular skill: {args.sample_particular_skill}")
    print("=" * 30)
    sampled_weak_ids = sample_weak_instances_per_node(tree_root, sample_size=args.sample_size)

# filter qa data by weak_ids
filtered_qa_data = [item for item in qa_data if item["unique_id"] in sampled_weak_ids]

# save filtered qa data
with open(output_weak_jsonl_path, "w", encoding="utf-8") as f:
    for item in filtered_qa_data:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")


qa_display_data = random.sample(qa_data, min(len(qa_data), 1000))
display_uids = set(item["unique_id"] for item in qa_display_data)

skill_to_instances_display = {
    k: [uid for uid in v if uid in display_uids]
    for k, v in skill_to_instances.items()
    if any(uid in display_uids for uid in v)
}
verdict_map_display = {
    uid: verdict_map[uid] for uid in display_uids if uid in verdict_map
}


html_output = f"""
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>Skill Tree with Q&A Filter</title>
  <script src="https://d3js.org/d3.v7.min.js"></script>
  <script>
    window.MathJax = {{
      tex: {{ inlineMath: [['$', '$'], ['\\\\(', '\\\\)']] }}
    }};
  </script>
  <script id="MathJax-script" async
          src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
  <style>
    body {{ font-family: sans-serif; display: flex; }}
    #tree-container {{ width: 30%; max-height: 90vh; overflow: auto; padding: 10px; border-right: 1px solid #ccc; }}
    #qa-container {{ width: 70%; padding: 10px; }}
    .card {{ border: 1px solid #ccc; border-radius: 8px; margin: 1em 0; padding: 1em; }}
    .toggle-btn {{ cursor: pointer; color: #007acc; text-decoration: underline; }}
    .content {{ display: none; margin-top: 1em; white-space: pre-wrap; }}
  </style>
</head>
<body>
  <div id="tree-container"></div>
  <div id="qa-container"><h2>Select a skill to view related Q&A</h2></div>

  <script>
    const treeData = {json.dumps(tree_root)};
    const skillToInstances = {json.dumps(skill_to_instances_display)};
    const qaData = {json.dumps(qa_display_data)};
    const verdictMap = {json.dumps(verdict_map_display)};


    const treeContainer = d3.select("#tree-container");
    function renderTree(data, container, path=[]) {{
      const ul = container.append("ul");
      const li = ul.append("li");
      const currentPath = [...path, data.name];
      const pathStr = currentPath.join(" → ");
      const counts = data.verdict_num_yes !== undefined && data.verdict_num_total !== undefined 
        ? ` [${{data.verdict_num_yes}}/${{data.verdict_num_total}}]` 
        : "";
      const accuracy = data.verdict_accuracy !== null 
        ? ` (✅ ${{(data.verdict_accuracy * 100).toFixed(1)}}%)${{counts}}` 
        : "";
  
      li.append("span")
        .style("cursor", "pointer")
        .style("color", skillToInstances[pathStr] ? "#007acc" : "#000")
        .text(data.name + accuracy)
        .on("click", () => {{
          const container = document.getElementById("qa-container");
          container.innerHTML = "<h2>Q&A related to: " + pathStr + "</h2>";
          const uids = skillToInstances[pathStr] || [];
          const matched = qaData.filter(item => uids.includes(item.unique_id));
          matched.forEach(item => {{
            const verdict = verdictMap[item.unique_id] || "unknown";
            const verdictSymbol = verdict === True ? "✔️" : (verdict === False ? "❌" : "❓");

            const card = document.createElement("div");
            card.className = "card";
            const header = document.createElement("div");
            header.innerHTML = `<strong>${{item.unique_id}}</strong> — ${{verdictSymbol}} ${{item.question.slice(0, 100)}}... <span class="toggle-btn">[Show/Hide]</span>`;
            const content = document.createElement("div");
            content.className = "content";
            content.innerHTML = `
              <p><strong>Question:</strong><br>${{item.question}}</p>
              <p><strong>Mapped Skills:</strong><br>${{item.mapped_chains.map(chain => chain.join(" → ")).join("<br>")}}</p>
              <p><strong>Solution:</strong><br>${{item.attempt}}</p>
              <p><strong>Qwen Attempt:</strong><br>${{item["Qwen/Qwen2.5-1.5B-Instruct_attempt"] || "N/A"}}</p>
              <p><strong>Verdict:</strong> ${{verdict}}</p>`;
            card.appendChild(header);
            card.appendChild(content);
            header.querySelector(".toggle-btn").addEventListener("click", () => {{
              content.style.display = content.style.display === "none" ? "block" : "none";
              MathJax.typeset();
            }});
            container.appendChild(card);
          }});
          MathJax.typeset();
        }});

      if (data.children && Object.keys(data.children).length > 0) {{
        Object.values(data.children).forEach(child =>
          renderTree(child, li.append("div"), currentPath)
        );
      }}
    }}

    renderTree(treeData, treeContainer);
  </script>
</body>
</html>
"""


# 写入 HTML 文件
with open(output_html_path, "w", encoding="utf-8") as f:
    f.write(html_output)

print(f"✅ HTML 生成完成：{output_html_path}")
