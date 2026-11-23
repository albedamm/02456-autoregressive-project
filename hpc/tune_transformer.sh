#!/bin/sh
#BSUB -q gpuv100
#BSUB -J "tune-stage1[1-96]%8"
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 4:00
#BSUB -R "rusage[mem=8GB]"
#BSUB -o hpc/logs/tune_stage1_%J_%I.out
#BSUB -e hpc/logs/tune_stage1_%J_%I.err

set -eu
cd /work3/s203530/02456-autoregressive-project
source /work3/s203530/02456-autoregressive-project/hpc/env.sh

num_epochs=60
seed=0
save_dir=src/train/trained_transformer_models/stage1

run_id=0
for d_model in 128 256 384 512; do
    for num_layers in 3 4 6 8; do
        for dropout in 0.1 0.2 0.3; do
            for learning_rate in 3e-4 1e-3; do
                run_id=$(( run_id + 1 ))
                # Each array element (1-96) trains only its own configuration
                [ "${run_id}" -eq "${LSB_JOBINDEX}" ] || continue

                num_heads=$(( d_model / 32 ))
                # File name as written by train_transformer.py (3e-4 -> 0.0003, 1e-3 -> 0.001)
                lr_name=$(python3 -c "print(float('${learning_rate}'))")
                name="d${d_model}_h${num_heads}_l${num_layers}_lr${lr_name}_drop${dropout}_e${num_epochs}_s${seed}"
                if [ -f "${save_dir}/${name}_losses.csv" ]; then
                    echo "Already done: ${name}, skipping"
                    exit 0
                fi

                echo "Starting run ${run_id}/96: d${d_model}_h${num_heads}_l${num_layers}_lr${lr_name}_drop${dropout}_e${num_epochs}_s${seed}"
                uv run python src/train/train_transformer.py \
                    --d-model "${d_model}" \
                    --num-heads "${num_heads}" \
                    --num-layers "${num_layers}" \
                    --num-epochs "${num_epochs}" \
                    --learning-rate "${learning_rate}" \
                    --dropout "${dropout}" \
                    --seed "${seed}" \
                    --save-dir "${save_dir}"
            done
        done
    done
done
