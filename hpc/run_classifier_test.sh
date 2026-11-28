#!/usr/bin/env bash
#BSUB -q gpuv100
#BSUB -J transformer-test
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 0:30
#BSUB -R "rusage[mem=8GB]"
#BSUB -o hpc/logs/classifier_test_%J.out
#BSUB -e hpc/logs/classifier_test_%J.err

# Submit from the repository root with: bsub < hpc/run_classifier_test.sh

set -euo pipefail
cd /work3/s203530/02456-autoregressive-project
source hpc/env.sh

uv run python eval/test_classifier.py
