#!/bin/sh
#BSUB -q gpuv100
#BSUB -J "tune-conditional-stage1[1-72]%8"
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 4:00
#BSUB -R "rusage[mem=8GB]"
#BSUB -o hpc/logs/tune-conditional-stage1_%J_%I.out
#BSUB -e hpc/logs/tune-conditional-stage1_%J_%I.err

# Submit from the repository root with:
#   bsub < hpc/train_transformer_baseline.sh

set -eu
cd /dtu/blackhole/10/168784/02456-autoregressive-project
source /dtu/blackhole/10/168784/02456-autoregressive-project/hpc/env.sh

num_epochs=60
seed=0
save_dir=trained_models/trained_cond_transformer_models

run_id=0
for d_model in 128 256 384; do
    for num_layers in 3 4; do
        for dropout in 0.1 0.2; do
            for label_dropout in 0.1 0.15 0.2; do
                for learning_rate in 3e-4 1e-3; do
                    run_id=$(( run_id + 1 ))
                    # Each array element (1-108) trains only its own configuration
                    [ "${run_id}" -eq "${LSB_JOBINDEX}" ] || continue

                    num_heads=$(( d_model / 32 ))
                    # File name as written by train_transformer.py (3e-4 -> 0.0003, 1e-3 -> 0.001)
                    lr_name=$(python3 -c "print(float('${learning_rate}'))")
                    name="d${d_model}_h${num_heads}_l${num_layers}_lr${learning_rate}_drop${dropout}_labeldrop${label_dropout}_e${num_epochs}_s${seed}"
                    if [ -f "${save_dir}/${name}_losses.csv" ]; then
                        echo "Already done: ${name}, skipping"
                        exit 0
                    fi

                    echo "Starting run ${run_id}/72: d${d_model}_h${num_heads}_l${num_layers}_lr${lr_name}_drop${dropout}_labeldrop${label_dropout}_e${num_epochs}_s${seed}"
                    uv run python src/train/train_conditional_transformer.py \
                        --d-model "${d_model}" \
                        --num-heads "${num_heads}" \
                        --num-layers "${num_layers}" \
                        --num-epochs "${num_epochs}" \
                        --learning-rate "${learning_rate}" \
                        --dropout "${dropout}" \
                        --label-dropout "${label_dropout}" \
                        --seed "${seed}" \
                        --save-dir "${save_dir}"
                done
            done
        done
    done
done
