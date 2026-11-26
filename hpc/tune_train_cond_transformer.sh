#!/bin/sh
#BSUB -q gpuv100
#BSUB -J conditional-transformer-test
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 0:30
#BSUB -R "rusage[mem=8GB]"
#BSUB -o hpc/conditional-transformer-test.out
#BSUB -e hpc/conditional-transformer-test.err

# Submit from the repository root with:
#   bsub < hpc/train_transformer_baseline.sh

set -eu
cd "$(dirname "$0")/.."
source hpc/env.sh
for num_epochs in 15; do
#     for learning_rate in 3e-4 1e-3; do
#         for dropout in 0.1 0.2; do
    echo "Starting run: epochs=${num_epochs}, learning_rate=${learning_rate}, dropout=${dropout}"
    uv run python src/train/train_transformer.py \
    --num-epochs "${num_epochs}" \
    # --learning-rate "${learning_rate}" \
    # --dropout "${dropout}"
#         done
#     done
done
