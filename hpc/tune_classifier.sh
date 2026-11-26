#!/bin/sh
#BSUB -q gpuv100
#BSUB -J "tune-classifier[1-18]%8"
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 4:00
#BSUB -R "rusage[mem=8GB]"
#BSUB -o hpc/logs/tune_classifier_%J_%I.out
#BSUB -e hpc/logs/tune_classifier_%J_%I.err

set -eu
cd /work3/s203530/02456-autoregressive-project
source /work3/s203530/02456-autoregressive-project/hpc/env.sh

num_epochs=60
seed=0
save_dir=trained_models/trained_classifier_models
run_id=0

for class_head_features in 32 64 128; do
    for dropout in 0.1 0.2 0.3; do
        for learning_rate in 3e-4 1e-3; do
            run_id=$(( run_id + 1 ))
            # Each array element trains one unique classifier-head configuration.
            [ "${run_id}" -eq "${LSB_JOBINDEX}" ] || continue

            lr_name=$(python3 -c "print(float('${learning_rate}'))")
            model_name="d128_h4_l3_cf${class_head_features}_lr${lr_name}_drop${dropout}_e${num_epochs}_s${seed}"
            if [ -f "${save_dir}/${model_name}_losses.csv" ]; then
                echo "Already done: ${model_name}, skipping"
                exit 0
            fi

            echo "Starting run ${run_id}/18: ${model_name}"
            uv run python src/train/train_classifier.py \
                --class-head-features "${class_head_features}" \
                --learning-rate "${learning_rate}" \
                --dropout "${dropout}" \
                --num-epochs "${num_epochs}" \
                --seed "${seed}" \
                --save-dir "${save_dir}"
        done
    done
done
