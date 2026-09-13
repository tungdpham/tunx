#!/usr/bin/env bash
set -e

# Format: "model engine batch_size"
CONFIGS=(
    "resnet50 cuda 32"
    "gpt2 cudnn 2"
)
SEEDS=(1 2 3)

mkdir -p experiments

for config in "${CONFIGS[@]}"; do
    read -r model engine batch_size <<< "$config"
    for seed in "${SEEDS[@]}"; do
        echo "Running seed $seed for model $model with engine $engine and batch size $batch_size..."
        PT_DIR="dump_pt_${model}_${seed}"
        TUNX_DIR="dump_tunx_${model}_${seed}"
        
        # 1. Dump PyTorch
        uv run python torch_benchmark/dump_utils.py --model "$model" --batch-size "$batch_size" --dump-dir "$PT_DIR" --seed "$seed"
        
        # 2. Run TunX
        ./build/bin/test_equivalence --model "$model" --pt-dir "$PT_DIR" --tunx-dir "$TUNX_DIR" --batch-size "$batch_size" --executor-mode optimized --engine "$engine"
        
        # 3. Compare Equivalence
        mkdir -p "experiments/${model}/seed_${seed}"
        uv run python torch_benchmark/compare_equivalence.py --pt_dir "$PT_DIR" --tunx_dir "$TUNX_DIR" --summary-json "experiments/${model}/seed_${seed}/summary.json" > "experiments/${model}/seed_${seed}/compare.log"
    done
done

echo "Generating summary..."
uv run python torch_benchmark/summarize_multiseed.py --results-dir experiments
