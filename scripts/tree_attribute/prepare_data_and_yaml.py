#!/usr/bin/env python3
"""
Combined script to prepare data for LLaMA-Factory training.
Combines functionality from prepare_data_omr.py and prepare_data_w_skill.py.

This script:
1. Loads the evaluated QA chains from reverse_mapping_with_score.py
2. Formats them for LLaMA-Factory (with optional skill information)
3. Registers the dataset in dataset_info.json
4. Auto-generates a corresponding YAML training config

Usage:
    # Basic usage (no skills)
    python prepare_data_and_yaml.py --input_path /path/to/mapped_qa_chains_attempt_evaluated_1K.jsonl

    # With skills
    python prepare_data_and_yaml.py --input_path /path/to/mapped_qa_chains_attempt_evaluated_1K.jsonl \
        --with_skill --skill_file_path /path/to/mapped_qa_chains.jsonl

    # Full options
    python prepare_data_and_yaml.py \
        --input_path /path/to/mapped_qa_chains_attempt_evaluated_1K.jsonl \
        --with_skill \
        --skill_file_path /path/to/mapped_qa_chains.jsonl \
        --full_tree_path /path/to/skill_mix.json \
        --min_shown_depth 0 \
        --max_shown_depth 5 \
        --is_show_tree \
        --model_name "Qwen/Qwen3-4B" \
        --template "qwen3" \
        --dataset_name "my_custom_dataset"
"""

import argparse
import json
import os
import re
from pathlib import Path

import pandas as pd
from datasets import load_dataset

# ============================================================================
# CONFIGURABLE CONSTANTS
# ============================================================================
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LLAMA_FACTORY_DIR = None  # Set via --llama_factory_dir (path to your LLaMA-Factory clone)
CAPABILITY_TO_DISTINCTION_PATH = os.path.join(
    _SCRIPT_DIR, "tree_data/evaltree/capability_to_distinction.json"
)
# ============================================================================


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare data and YAML for LLaMA-Factory training")

    # Input/output paths
    parser.add_argument("--input_path", type=str, required=True,
                        help="Path to the evaluated JSONL file (e.g., mapped_qa_chains_attempt_evaluated_1K.jsonl)")
    parser.add_argument("--output_path", type=str, default=None,
                        help="Output path for the prepared JSONL. Default: input path with _llama_factory suffix")
    parser.add_argument("--dataset_name", type=str, default=None,
                        help="Name for the dataset in dataset_info.json. Auto-generated if not provided")

    # Skill-related options
    parser.add_argument("--with_skill", action="store_true",
                        help="Include skill information in the data")
    parser.add_argument("--skill_file_path", type=str, default=None,
                        help="Path to the skill mapping file (required if --with_skill)")
    parser.add_argument("--full_tree_path", type=str, default=None,
                        help="Path to the full skill tree JSON (optional, for showing tree structure)")
    parser.add_argument("--is_show_tree", action="store_true",
                        help="Include full tree structure in the prompt")
    parser.add_argument("--min_shown_depth", type=int, default=0,
                        help="Minimum depth for skill chain display")
    parser.add_argument("--max_shown_depth", type=int, default=5,
                        help="Maximum depth for skill chain display")
    parser.add_argument("--capability_to_distinction_path", type=str,
                        default=CAPABILITY_TO_DISTINCTION_PATH,
                        help="Path to capability_to_distinction.json for evaltree")

    # YAML generation options
    parser.add_argument("--generate_yaml", action="store_true", default=True,
                        help="Generate a training YAML config (default: True)")
    parser.add_argument("--no_yaml", action="store_true",
                        help="Do not generate a training YAML config")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-4B",
                        help="Model name for the YAML config")
    parser.add_argument("--template", type=str, default="qwen3",
                        help="Template name for the YAML config")
    parser.add_argument("--cutoff_len", type=int, default=8192,
                        help="Cutoff length for the YAML config")
    parser.add_argument("--num_train_epochs", type=int, default=5,
                        help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=1.0e-5,
                        help="Learning rate")
    parser.add_argument("--per_device_train_batch_size", type=int, default=1,
                        help="Per device train batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4,
                        help="Gradient accumulation steps")

    # LLaMA-Factory paths
    parser.add_argument("--llama_factory_dir", type=str, default=LLAMA_FACTORY_DIR,
                        help="Path to LLaMA-Factory directory (required for dataset registration and YAML output)")

    return parser.parse_args()


def infer_dataset_name(input_path, with_skill, min_depth=None, max_depth=None):
    """Infer dataset name from input path."""
    path = Path(input_path)

    # Extract components from path
    # Example: .../tree_data/skill_mix/omr_qwen3_temp1/mapped_qa_chains_attempt_evaluated_1K.jsonl
    parts = path.parts

    # Find tree type (skill_mix or evaltree)
    tree_type = None
    model_info = None
    for i, part in enumerate(parts):
        if part in ["skill_mix", "evaltree"]:
            tree_type = part
            if i + 1 < len(parts):
                model_info = parts[i + 1]
            break

    # Extract sample size from filename (e.g., "1K", "5K", "10K")
    filename = path.stem
    sample_match = re.search(r'_(\d+K)', filename)
    sample_size = sample_match.group(1) if sample_match else "1K"

    # Build dataset name
    if tree_type and model_info:
        # Parse model_info like "omr_qwen3_temp1" or "omr_r1_llama_8b_temp1"
        base_name = f"omr_{model_info.replace('omr_', '')}_{tree_type}_{sample_size}"
    else:
        base_name = filename.replace("mapped_qa_chains_attempt_evaluated_", "omr_")

    # Add skill suffix if applicable
    if with_skill:
        skill_suffix = f"_w_{tree_type if tree_type else 'skill'}"
        if min_depth is not None and max_depth is not None:
            skill_suffix += f"_depth_{min_depth}_{max_depth}"
        base_name += skill_suffix

    return base_name


def apply_chat_template_simple(example):
    """Apply chat template without skills."""
    cot_prefix = "Please reason step by step, and put your final answer within \\boxed{}."
    return {
        "messages": [
            {"role": "user", "content": f"{cot_prefix}\n\nQuestion: {example['question']}"},
            {"role": "assistant", "content": f"{example['attempt']}"}
        ]
    }


def apply_chat_template_with_skill(row, min_shown_depth, max_shown_depth,
                                    full_tree=None, is_show_tree=False,
                                    capability_to_distinction=None):
    """Apply chat template with skill information."""
    skill_tree_str = ""
    if is_show_tree and full_tree:
        skill_tree_str = "Here is a skill tree:\n" + json.dumps(full_tree, indent=2) + "\n\n"

    cot_prefix = "Please reason step by step, and put your final answer within \\boxed{}."

    if max_shown_depth is not None and min_shown_depth is not None:
        if capability_to_distinction is not None:
            sub_chain_shown = [
                '->'.join([
                    capability_to_distinction[tmp][0] if tmp in capability_to_distinction else tmp
                    for tmp in sub_chain[min_shown_depth:max_shown_depth]
                ])
                for sub_chain in row['mapped_chains']
                if len(sub_chain) >= max_shown_depth
            ]
        else:
            sub_chain_shown = [
                '->'.join(sub_chain[min_shown_depth:max_shown_depth])
                for sub_chain in row['mapped_chains']
                if len(sub_chain) >= max_shown_depth
            ]
        sub_chain_shown = list(set(sub_chain_shown))
        skill_prefix = "Skills:\n" + '\n'.join(sub_chain_shown)
    else:
        skill_prefix = "Skills:\n" + '\n'.join(['->'.join(sub_chain) for sub_chain in row['mapped_chains']])

    return [
        {"role": "user", "content": f"{skill_tree_str}{cot_prefix}\n\nQuestion: {row['question']}"},
        {"role": "assistant", "content": f"{skill_prefix}\n{row['attempt']}"}
    ]


def generate_yaml_config(args, dataset_name, output_yaml_path):
    """Generate YAML training config."""
    # Infer output directory name from dataset name
    output_dir_name = f"sft_{dataset_name}"

    yaml_content = f"""### model
model_name_or_path: {args.model_name}
template: {args.template}

### method
stage: sft
do_train: true
finetuning_type: full
deepspeed: examples/deepspeed/ds_z3_config.json

### dataset
dataset: {dataset_name}
cutoff_len: {args.cutoff_len}
max_samples: 100000
overwrite_cache: true
preprocessing_num_workers: 16
dataloader_num_workers: 0

### output
output_dir: saves/{args.template}/{output_dir_name}
logging_steps: 1
save_steps: 600
plot_loss: true
overwrite_output_dir: true
save_only_model: true
report_to: wandb

### train
per_device_train_batch_size: {args.per_device_train_batch_size}
gradient_accumulation_steps: {args.gradient_accumulation_steps}
learning_rate: {args.learning_rate}
num_train_epochs: {args.num_train_epochs}
lr_scheduler_type: cosine
warmup_ratio: 0.1
bf16: true
ddp_timeout: 180000000
resume_from_checkpoint: null

### sequence parallel
sequence_parallel_size: 4
sequence_parallel_mode: "zigzag-ring"
"""

    with open(output_yaml_path, 'w') as f:
        f.write(yaml_content)

    print(f"Generated YAML config: {output_yaml_path}")


def main():
    args = parse_args()

    # Validate arguments
    if args.with_skill and not args.skill_file_path:
        raise ValueError("--skill_file_path is required when using --with_skill")
    if args.llama_factory_dir is None:
        raise ValueError(
            "--llama_factory_dir is required. Point it to your LLaMA-Factory installation "
            "(e.g., --llama_factory_dir /path/to/LLaMA-Factory)"
        )

    # Determine output path (always use absolute path)
    if args.output_path:
        output_file_path = os.path.abspath(args.output_path)
    else:
        if args.with_skill:
            suffix = f"_w_skill_depth_{args.min_shown_depth}_{args.max_shown_depth}_llama_factory.jsonl"
        else:
            suffix = "_llama_factory.jsonl"
        output_file_path = os.path.abspath(args.input_path.replace(".jsonl", suffix))

    # Determine dataset name
    if args.dataset_name:
        dataset_name = args.dataset_name
    else:
        dataset_name = infer_dataset_name(
            args.input_path,
            args.with_skill,
            args.min_shown_depth if args.with_skill else None,
            args.max_shown_depth if args.with_skill else None
        )

    print(f"Input path: {args.input_path}")
    print(f"Output path: {output_file_path}")
    print(f"Dataset name: {dataset_name}")
    print(f"With skill: {args.with_skill}")

    # Load dataset
    try:
        dataset = load_dataset(args.input_path)
    except:
        dataset = load_dataset("json", data_files=args.input_path)

    if args.with_skill:
        # Load skill data and merge
        dataset_df = dataset["train"].to_pandas()
        skill_df = pd.read_json(args.skill_file_path, lines=True)

        # Load capability_to_distinction if exists
        capability_to_distinction = None
        if os.path.exists(args.capability_to_distinction_path):
            with open(args.capability_to_distinction_path, "r") as f:
                capability_to_distinction = json.load(f)

        # Load full tree if needed
        full_tree = None
        if args.is_show_tree and args.full_tree_path:
            with open(args.full_tree_path, "r") as f:
                full_tree = json.load(f)

        # Merge with skill data
        if "mapped_chains" in dataset_df.columns:
            dataset_df_w_skill = dataset_df.drop(columns=["mapped_chains"]).merge(
                skill_df[["question", "unique_id", "mapped_chains"]],
                on="question",
                how="inner"
            )
        else:
            dataset_df_w_skill = dataset_df.merge(
                skill_df[["question", "unique_id", "mapped_chains"]],
                on="question",
                how="inner"
            )

        # Apply chat template with skills
        dataset_df_w_skill["messages"] = dataset_df_w_skill.apply(
            lambda row: apply_chat_template_with_skill(
                row,
                args.min_shown_depth,
                args.max_shown_depth,
                full_tree,
                args.is_show_tree,
                capability_to_distinction
            ),
            axis=1
        )

        # Save
        dataset_df_w_skill.to_json(output_file_path, lines=True, orient="records")
        print(f"Length of dataset: {len(dataset_df_w_skill)}")
    else:
        # Simple mode without skills
        dataset = dataset.map(apply_chat_template_simple)
        dataset["train"].to_json(output_file_path, lines=True)
        print(f"Length of dataset: {len(dataset['train'])}")

    # Register dataset in dataset_info.json
    dataset_info_path = os.path.join(args.llama_factory_dir, "data", "dataset_info.json")
    with open(dataset_info_path, "r") as f:
        dataset_info = json.load(f)

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

    with open(dataset_info_path, "w") as f:
        json.dump(dataset_info, f, indent=2)

    print(f"Registered dataset '{dataset_name}' in {dataset_info_path}")

    # Generate YAML config
    if args.generate_yaml and not args.no_yaml:
        yaml_filename = f"llama-factory_{dataset_name}.yaml"
        yaml_path = os.path.join(args.llama_factory_dir, yaml_filename)
        generate_yaml_config(args, dataset_name, yaml_path)

    print("\nDone! To start training, run:")
    print(f"  cd {args.llama_factory_dir}")
    print(f"  llamafactory-cli train llama-factory_{dataset_name}.yaml")


if __name__ == "__main__":
    main()
