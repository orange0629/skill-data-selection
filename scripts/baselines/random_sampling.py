import argparse
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--input_path", type=str, required=True)
parser.add_argument("--output_path", type=str, required=True)
parser.add_argument("--sample_size", type=int, default=1000)
args = parser.parse_args()

df = pd.read_json(args.input_path, lines=True)
sampled_df = df.sample(n=args.sample_size, random_state=42)
sampled_df.to_json(args.output_path, orient="records", lines=True)
print(f"Saved {len(sampled_df)} examples to {args.output_path}")
