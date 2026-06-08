import argparse
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--input_path", type=str, required=True)
parser.add_argument("--output_path", type=str, required=True)
parser.add_argument("--min_pass_rate", type=float, default=0.25)
parser.add_argument("--max_pass_rate", type=float, default=0.625)
parser.add_argument("--sample_size", type=int, default=1000)
args = parser.parse_args()

df = pd.read_json(args.input_path, lines=True)
df["pass_rate_72b_tir"] = pd.to_numeric(df["pass_rate_72b_tir"], errors="coerce")
filtered_df = df[(df["pass_rate_72b_tir"] >= args.min_pass_rate) & (df["pass_rate_72b_tir"] <= args.max_pass_rate)]
sampled_df = filtered_df.sample(n=args.sample_size, random_state=42)
sampled_df.to_json(args.output_path, orient="records", lines=True)
print(f"Saved {len(sampled_df)} examples to {args.output_path}")
