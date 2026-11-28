#!/usr/bin/env bash
#BSUB -q gpuv100
#BSUB -J generate-classifier
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 3:00
#BSUB -R "rusage[mem=8GB]"
#BSUB -o hpc/logs/generate_classifier%J.out
#BSUB -e hpc/logs/generate_classifier%J.err

# Submit from the repository root with: bsub < hpc/run_generate_classifier.sh

set -euo pipefail
cd /work3/s203530/02456-autoregressive-project
source hpc/env.sh

# target_family: 0 = AcyP, 1 = HPr, 2 = HU
for target_family in 0 1 2; do
    for lam in 0 0.5 1 2 5 10; do
        # Skip what is already generated, so the job can be resubmitted to finish the rest.
        # Python writes the lambda as a float into the file name (0 -> 0.0, 10 -> 10.0).
        lam_name=$(python3 -c "print(float('${lam}'))")
        if [ -f "results/sequences/classifier_sequences/family${target_family}_lam${lam_name}.csv" ]; then
            echo "Already done: target_family=${target_family}, lam=${lam}, skipping"
            continue
        fi

        echo "Generating sequences for target_family=${target_family} with lam=${lam} ..."
        uv run python src/generate/generate_classifier.py \
            --num_sequences 20 \
            --target_family "${target_family}" \
            --lam "${lam}"
    done
done