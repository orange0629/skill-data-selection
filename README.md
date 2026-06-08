<div align="center">

# Skill-Aware Data Selection and Fine-Tuning<br>for Data-Efficient Reasoning Distillation

[![ACL 2026](https://img.shields.io/badge/ACL-2026%20Main-red?style=flat-square)](https://arxiv.org/abs/2601.10109)
[![arXiv](https://img.shields.io/badge/arXiv-2601.10109-b31b1b?style=flat-square&logo=arxiv)](https://arxiv.org/abs/2601.10109)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue?style=flat-square&logo=python)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

**Lechen Zhang · Yunxiang Zhang · Wei Hu · Lu Wang**

*University of Michigan*

</div>

---

## Abstract

Large reasoning models such as DeepSeek-R1 and their distilled variants achieve strong performance on complex reasoning tasks. Yet, distilling these models often demands large-scale data for supervised fine-tuning (SFT), motivating the pursuit of data-efficient training methods. To address this, we propose a **skill-centric distillation framework** that efficiently transfers reasoning ability to weaker models with two components: **(1) Skill-based data selection**, which prioritizes examples targeting the student model's weaker skills, and **(2) Skill-aware fine-tuning**, which encourages explicit skill decomposition during problem solving. With only **1,000 training examples** selected from a 100K teacher-generated corpus, our method surpasses random SFT baselines by **+1.6% on Qwen3-4B** and **+1.4% on Qwen3-8B** across five mathematical reasoning benchmarks. Further analysis confirms that these gains concentrate on skills emphasized during training, highlighting the effectiveness of skill-centric training for efficient reasoning distillation.

---

## Method

Our framework has three stages, illustrated in Figure 1 of the paper:

```
┌─────────────────────────────────────────────────────────────────────┐
│                    100K Teacher-Generated Pool                       │
│                    (OpenMathReasoning + DeepSeek-R1)                 │
└────────────────────────────┬────────────────────────────────────────┘
                             │
             ┌───────────────▼───────────────┐
             │   Stage 1: Skill Attribution   │
             │                               │
             │  Each Q&A is mapped to a path  │
             │  through a hierarchical skill  │
             │  tree (e.g., Math → Prob →     │
             │  Bayes) via top-down LLM-based │
             │  traversal with Qwen2.5-32B    │
             └───────────────┬───────────────┘
                             │  Q&A + skill chain
             ┌───────────────▼───────────────┐
             │   Stage 2: Skill-Based         │
             │   Sampling                     │
             │                               │
             │  Evaluate student model per    │
             │  skill node → sample weighted  │
             │  toward weakest skill areas    │
             └───────────────┬───────────────┘
                             │  ~1K selected examples
             ┌───────────────▼───────────────┐
             │   Stage 3: Skill-Aware SFT     │
             │                               │
             │  Prepend skill chain to each   │
             │  training prompt (shown in     │
             │  red in paper) for explicit    │
             │  skill decomposition during    │
             │  fine-tuning                   │
             └───────────────────────────────┘
```


## Repository Structure

```
skill-data-selection-public/
├── scripts/
│   ├── tree_attribute/              # Core pipeline scripts
│   │   ├── startserver.sh           # Pre-step: Launch vLLM servers
│   │   ├── run_1_tree_attribution.sh      # Step 1: Map Q&A → skill chains
│   │   ├── run_2_sample_student.sh        # Step 2: Student model inference
│   │   ├── run_3_judge_model.sh           # Step 3: Judge answer correctness
│   │   ├── run_3_judge_model_nogpu.sh     # Step 3 (CPU variant)
│   │   ├── run_4_reverse_mapping.sh       # Step 4: Skill-aware data selection
│   │   ├── attribute_question_to_tree.py  # DFS skill attribution
│   │   ├── reverse_mapping_with_score.py  # Weak-skill data selection
│   │   ├── prepare_data_and_yaml.py       # Format data for LLaMA-Factory
│   │   ├── modeling.py                    # Async vLLM API client
│   │   ├── sample_from_llm.py             # Student model inference
│   │   ├── judge_qwen_math.py             # Math solution judging
│   │   └── tree_data/
│   │       ├── evaltree/                  # EvalTree + supporting data
│   │       └── skill_mix/                 # Instruct-SkillMix tree
│   ├── baselines/
│   │   ├── random_sampling.py             # Random selection baseline
│   │   └── difficulty_sampling.py         # Difficulty-filtered baseline
│   ├── skill-tree/                        # Skill tree construction
│   ├── eval/                              # Evaluation scripts & benchmark data
│   └── lib/                              # Shared utilities & math graders
├── training/                             # LLaMA-Factory configs
│   ├── dataset_info.json
│   ├── prepare_data_omr.py
│   ├── prepare_data_w_skill.py
│   └── yaml/                             # Training YAML configs for all experiments
├── requirements.txt
└── README.md
```

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Install LLaMA-Factory

We use [360-LLaMA-Factory](https://github.com/Qihoo360/360-LLaMA-Factory), which extends LLaMA-Factory with Sequence Parallelism for efficient full-parameter SFT:

```bash
git clone https://github.com/Qihoo360/360-LLaMA-Factory.git
cd 360-LLaMA-Factory
pip install -e ".[torch,metrics]"
```

### 3. Configure environment

```bash
export HF_HOME=/path/to/your/model/cache
```

---

## Pipeline

All core pipeline scripts are in `scripts/tree_attribute/`. The numbered shell scripts contain the exact commands from the paper and serve as the primary entry points.

### Pre-step: Start vLLM inference servers

Skill attribution requires a large LLM (`Qwen/Qwen2.5-32B-Instruct`) served via vLLM. The script launches 4 server instances across 8 GPUs (2 GPUs each, tensor-parallel):

```bash
cd scripts/tree_attribute
# Edit startserver.sh to set: SBATCH account/partition, HF_HOME
sbatch startserver.sh
```

Servers start on ports 2341–2344. You can run multiple nodes in parallel and pass all endpoint URLs to Step 1.

### Step 1: Skill Tree Attribution

Map every Q&A pair in the 100K pool to a path through the skill tree:

```bash
# Edit run_1_tree_attribution.sh: set --api_url to your running vLLM servers
sbatch run_1_tree_attribution.sh
```

Internally calls `attribute_question_to_tree.py` with 512 parallel workers. Uses DFS + LLM to walk the skill tree and find the best-matching skill chain for each problem.

**Output**: `tree_data/evaltree/omr_mapped_qa_chains.jsonl`
Each record has the original Q&A plus `mapped_chains` (list of skill paths through the tree).

### Step 2: Generate Student Model Attempts

Run the student model on the 100K pool to collect per-example answers:

```bash
# Edit run_2_sample_student.sh: set --model_name to your student model
sbatch run_2_sample_student.sh
```

Calls `sample_from_llm.py` using local vLLM inference.

**Output**: e.g., `omr_filtered_100K_w_llama_8b.jsonl` with student-generated attempts.

### Step 3: Judge Answer Correctness

Score the student model's attempts using a math-specialized judge:

```bash
# Runs judge_qwen_math.py with Qwen2.5-Math-7B + Qwen2.5-7B-Instruct
sbatch run_3_judge_model.sh
```

Adds an `is_correct` field to each record.

### Step 4: Skill-Aware Reverse Mapping & Data Selection

Build a reverse index (skill → instances), compute per-skill accuracy, and sample from the weakest skill nodes:

```bash
# Uncomment the relevant python call in run_4_reverse_mapping.sh
sbatch run_4_reverse_mapping.sh
```

Or run manually:

```bash
cd scripts/tree_attribute

python reverse_mapping_with_score.py \
    --skill_tree_path "tree_data/evaltree/evaltree_math_full.json" \
    --qa_jsonl_path "omr_filtered_100K.jsonl" \
    --tree_mapping_path "tree_data/evaltree/omr_mapped_qa_chains.jsonl" \
    --output_path "tree_data/evaltree/omr_qwen3_temp1" \
    --sample_size 1000 \
    --attempt_model "Qwen/Qwen3-4B" \
    --temperature 1
```

**Output**: `tree_data/evaltree/omr_qwen3_temp1/mapped_qa_chains_attempt_evaluated_1K.jsonl`

### Step 5: Format Data for LLaMA-Factory

```bash
cd scripts/tree_attribute

# Standard SFT (no skill prefix):
python prepare_data_and_yaml.py \
    --input_path "tree_data/evaltree/omr_qwen3_temp1/mapped_qa_chains_attempt_evaluated_1K.jsonl" \
    --dataset_name "omr_qwen3_4b_evaltree_1K" \
    --llama_factory_dir /path/to/360-LLaMA-Factory

# Skill-aware SFT (prepend skill chain — best results):
python prepare_data_and_yaml.py \
    --input_path "tree_data/evaltree/omr_qwen3_temp1/mapped_qa_chains_attempt_evaluated_1K.jsonl" \
    --with_skill \
    --skill_file_path "tree_data/evaltree/omr_mapped_qa_chains.jsonl" \
    --dataset_name "omr_qwen3_4b_evaltree_1K_w_skill" \
    --llama_factory_dir /path/to/360-LLaMA-Factory
```

This registers the dataset in `dataset_info.json` and auto-generates a training YAML config.

### Step 6: Fine-tune

```bash
cd /path/to/360-LLaMA-Factory
llamafactory-cli train llama-factory_omr_qwen3_4b_evaltree_1K_w_skill.yaml
```

Pre-built YAML configs for all paper experiments (Qwen3-4B/8B, R1-Llama-8B; 1K/5K/10K/20K; with/without skill prefix) are in `training/yaml/`.

---

## Data

| Resource | Link |
|---|---|
| OpenMathReasoning (OMR 100K pool) | [nvidia/OpenMathReasoning](https://huggingface.co/datasets/nvidia/OpenMathReasoning) |
| EvalTree skill tree | `scripts/tree_attribute/tree_data/evaltree/evaltree_math_full.json` |
| Instruct-SkillMix tree | `scripts/tree_attribute/tree_data/skill_mix/skill_mix.json` |

---

## Models

| Role | Model |
|---|---|
| Skill attribution LLM | [Qwen/Qwen2.5-32B-Instruct](https://huggingface.co/Qwen/Qwen2.5-32B-Instruct) |
| Student (4B) | [Qwen/Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B) |
| Student (8B) | [Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) |
| Student (8B, distilled) | [deepseek-ai/DeepSeek-R1-Distill-Llama-8B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-8B) |
| Math judge | [Qwen/Qwen2.5-Math-7B](https://huggingface.co/Qwen/Qwen2.5-Math-7B) |

---

## Citation

```bibtex
@inproceedings{zhang2026skill,
  title     = {Skill-Aware Data Selection and Fine-Tuning for Data-Efficient Reasoning Distillation},
  author    = {Zhang, Lechen and Zhang, Yunxiang and Hu, Wei and Wang, Lu},
  booktitle = {Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers)},
  year      = {2026}
}
```

---

## Acknowledgments

This work builds on:
- [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) by Yaowei Zheng et al.
- [360-LLaMA-Factory](https://github.com/Qihoo360/360-LLaMA-Factory) for Sequence Parallelism
- [OpenMathReasoning](https://huggingface.co/datasets/nvidia/OpenMathReasoning) dataset by NVIDIA
- [vLLM](https://github.com/vllm-project/vllm) for efficient LLM serving
