#!/usr/bin/env bash
set -euo pipefail

# Reproducible multi-seed FP32/BF16 equivalence sweep.
# Override SEEDS, RESNET_BATCH_SIZE, or BLOCK_BATCH_SIZE in the environment.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SEEDS="${SEEDS:-42 43 44 45 46}"
RESNET_BATCH_SIZE="${RESNET_BATCH_SIZE:-32}"
BLOCK_BATCH_SIZE="${BLOCK_BATCH_SIZE:-2}"
RESULTS_DIR="${RESULTS_DIR:-${ROOT_DIR}/multiseed_results}"
BUILD_DIR="${ROOT_DIR}/build"

mkdir -p "${RESULTS_DIR}/logs" "${RESULTS_DIR}/resnet" "${RESULTS_DIR}/blocks"

cat > "${RESULTS_DIR}/RUN_INSTRUCTIONS.txt" <<EOF
Multi-seed equivalence sweep
============================
Date: $(date -u +%Y-%m-%dT%H:%M:%SZ)
Seeds: ${SEEDS}
ResNet batch size: ${RESNET_BATCH_SIZE}
Isolated-block batch size: ${BLOCK_BATCH_SIZE}

Re-run from the repository root with:
  SEEDS="${SEEDS}" RESNET_BATCH_SIZE=${RESNET_BATCH_SIZE} BLOCK_BATCH_SIZE=${BLOCK_BATCH_SIZE} ./run_multiseed_equivalence.sh

Individual commands used per seed:
  uv run python torch_benchmark/dump_utils.py --model resnet50 --dump-dir <pt-dir> --batch-size ${RESNET_BATCH_SIZE} --seed <seed>
  ./build/bin/test_equivalence --model resnet50 --pt-dir <pt-dir> --tunx-dir <tunx-dir> --batch-size ${RESNET_BATCH_SIZE}
  uv run python torch_benchmark/compare_equivalence.py --pt_dir <pt-dir> --tunx_dir <tunx-dir> --summary-json <summary.json>

For each isolated block (residual, inception, attention, gpt2):
  uv run python torch_benchmark/dump_block_utils.py --block <block> --dump-dir <pt-dir> --batch-size ${BLOCK_BATCH_SIZE} --seed <seed>
  ./build/bin/test_block_equivalence --block <block> --pt-dir <pt-dir> --tunx-dir <tunx-dir> --batch-size ${BLOCK_BATCH_SIZE}
  uv run python torch_benchmark/compare_equivalence.py --pt_dir <pt-dir> --tunx_dir <tunx-dir> --summary-json <summary.json>

The raw PyTorch/TunX dumps are retained under this results directory. Full command
output is in logs/, and aggregate per-trial metrics are in trial_metrics.csv and
summary_metrics.csv.
EOF

# The C++ binary must reflect the current CMake configuration before trials begin.
cmake -S "${ROOT_DIR}" -B "${BUILD_DIR}" > "${RESULTS_DIR}/logs/cmake.log" 2>&1
cmake --build "${BUILD_DIR}" --target test_equivalence test_block_equivalence -j8 \
  > "${RESULTS_DIR}/logs/build.log" 2>&1

for seed in ${SEEDS}; do
  pt_dir="${RESULTS_DIR}/resnet/seed_${seed}/pt"
  tunx_dir="${RESULTS_DIR}/resnet/seed_${seed}/tunx"
  summary_json="${RESULTS_DIR}/resnet/seed_${seed}/summary.json"
  mkdir -p "${pt_dir}" "${tunx_dir}"

  uv run python "${ROOT_DIR}/torch_benchmark/dump_utils.py" \
    --model resnet50 --dump-dir "${pt_dir}" --batch-size "${RESNET_BATCH_SIZE}" --seed "${seed}" \
    > "${RESULTS_DIR}/logs/resnet_seed_${seed}_pytorch.log" 2>&1
  "${BUILD_DIR}/bin/test_equivalence" --model resnet50 --pt-dir "${pt_dir}" \
    --tunx-dir "${tunx_dir}" --batch-size "${RESNET_BATCH_SIZE}" \
    > "${RESULTS_DIR}/logs/resnet_seed_${seed}_tunx.log" 2>&1
  uv run python "${ROOT_DIR}/torch_benchmark/compare_equivalence.py" \
    --pt_dir "${pt_dir}" --tunx_dir "${tunx_dir}" --summary-json "${summary_json}" \
    > "${RESULTS_DIR}/logs/resnet_seed_${seed}_compare.log" 2>&1

done

for block in residual inception attention gpt2; do
  for seed in ${SEEDS}; do
    pt_dir="${RESULTS_DIR}/blocks/${block}/seed_${seed}/pt"
    tunx_dir="${RESULTS_DIR}/blocks/${block}/seed_${seed}/tunx"
    summary_json="${RESULTS_DIR}/blocks/${block}/seed_${seed}/summary.json"
    mkdir -p "${pt_dir}" "${tunx_dir}"

    uv run python "${ROOT_DIR}/torch_benchmark/dump_block_utils.py" \
      --block "${block}" --dump-dir "${pt_dir}" --batch-size "${BLOCK_BATCH_SIZE}" --seed "${seed}" \
      > "${RESULTS_DIR}/logs/${block}_seed_${seed}_pytorch.log" 2>&1
    "${BUILD_DIR}/bin/test_block_equivalence" --block "${block}" --pt-dir "${pt_dir}" \
      --tunx-dir "${tunx_dir}" --batch-size "${BLOCK_BATCH_SIZE}" \
      > "${RESULTS_DIR}/logs/${block}_seed_${seed}_tunx.log" 2>&1
    uv run python "${ROOT_DIR}/torch_benchmark/compare_equivalence.py" \
      --pt_dir "${pt_dir}" --tunx_dir "${tunx_dir}" --summary-json "${summary_json}" \
      > "${RESULTS_DIR}/logs/${block}_seed_${seed}_compare.log" 2>&1
  done
done

uv run python "${ROOT_DIR}/torch_benchmark/summarize_multiseed.py" \
  --results-dir "${RESULTS_DIR}" \
  > "${RESULTS_DIR}/logs/summary.log" 2>&1

printf 'Multi-seed sweep complete. Results: %s\n' "${RESULTS_DIR}"
