#!/bin/bash
# The interpreter used to execute the script

#“#SBATCH” directives that convey submission options:
#SBATCH --job-name=server
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128g
#SBATCH --gpus=8
#SBATCH --time=48:00:00
#SBATCH -p spgpu2

# The application(s) to execute along with its input arguments and options:

module load cuda/12.6.3
module load gcc/11.2.0
source ~/.bashrc
conda activate prompting

export HF_HOME=""  # set your HuggingFace model cache
export VLLM_SKIP_P2P_CHECK=1

CUDA_VISIBLE_DEVICES=0,1 nohup vllm serve Qwen/Qwen2.5-32B-Instruct --port 2341 --tensor_parallel_size 2 --dtype=half --disable-custom-all-reduce > server7.log &
sleep 300
CUDA_VISIBLE_DEVICES=2,3 nohup vllm serve Qwen/Qwen2.5-32B-Instruct --port 2342 --tensor_parallel_size 2 --dtype=half --disable-custom-all-reduce > server7.log &
CUDA_VISIBLE_DEVICES=4,5 nohup vllm serve Qwen/Qwen2.5-32B-Instruct --port 2343 --tensor_parallel_size 2 --dtype=half --disable-custom-all-reduce > server7.log &
CUDA_VISIBLE_DEVICES=6,7 nohup vllm serve Qwen/Qwen2.5-32B-Instruct --port 2344 --tensor_parallel_size 2 --dtype=half --disable-custom-all-reduce > server7.log &
sleep 604800