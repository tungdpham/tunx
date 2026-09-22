#!/usr/bin/env bash
# Run on both machines: rank 0 on 10.10.0.2, rank 1 on 10.10.0.1.
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: run_distributed_benchmarks.sh NODE_RANK [--dry-run]

  0 = machine 1 (10.10.0.2); 1 = machine 2 (10.10.0.1)
  Run on BOTH machines with the same benchmark settings.

Environment overrides:
  MASTER_ADDR       Rendezvous host on both nodes (default: 10.10.0.2)
  MASTER_PORT       First of eight consecutive ports (default: 29500)
  GPUS_PER_NODE     GPU processes per machine (default: 1)
  RUN_ID            Output subdirectory (default: distributed_all)
  OUTPUT_ROOT       Results directory (default: REPO/benchmark_results)
  TORCHRUN          Launcher (default: REPO/.venv/bin/torchrun)
  PRECISION         fp32 or bf16 (default: fp32)
  MICRO_BATCH_SIZE  Per-GPU microbatch (default: 8)
  WARMUP_STEPS      Warmup updates (default: 50)
  STEPS             Measured updates (default: 2000)
  ZERO_STAGE        DeepSpeed ZeRO stage 1, 2, or 3 (default: 3)
  SEED              Random seed (default: 42)
  SYNTHETIC         1 for random inputs, 0 for ImageNet100 (default: 0)
  DATA_ROOT         Optional local ImageNet100 path (may differ by node)

Results: OUTPUT_ROOT/RUN_ID/{deepspeed,fsdp}_v{1,2,3,4}.json on rank 0.
Logs:    OUTPUT_ROOT/RUN_ID/logs/node{0,1}/BACKEND_vN.log on each node.
Existing files for the same RUN_ID are overwritten. Stops on first failure.
EOF
}

if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then usage; exit 0; fi
if [[ $# -lt 1 || $# -gt 2 || ! "$1" =~ ^[01]$ ]]; then usage >&2; exit 2; fi
node_rank="$1"
dry_run=0
if [[ $# == 2 ]]; then
  [[ "$2" == --dry-run ]] || { usage >&2; exit 2; }
  dry_run=1
fi
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/.." && pwd)"
cd "$repo_root"
master_addr="${MASTER_ADDR:-10.10.0.2}"
first_port="${MASTER_PORT:-29500}"
gpus="${GPUS_PER_NODE:-1}"
run_id="${RUN_ID:-distributed_all}"
output_root="${OUTPUT_ROOT:-$repo_root/benchmark_results}"
launcher="${TORCHRUN:-$repo_root/.venv/bin/torchrun}"
precision="${PRECISION:-fp32}"
micro_batch="${MICRO_BATCH_SIZE:-8}"
warmup="${WARMUP_STEPS:-50}"
steps="${STEPS:-2000}"
zero_stage="${ZERO_STAGE:-3}"
seed="${SEED:-42}"
synthetic="${SYNTHETIC:-0}"
fail() { echo "Error: $*" >&2; exit 2; }
for value in "$first_port" "$gpus" "$micro_batch" "$steps"; do
  [[ "$value" =~ ^[1-9][0-9]*$ ]] || fail 'Ports, GPU count, batch size and steps must be positive integers.'
done
for value in "$warmup" "$seed"; do
  [[ "$value" =~ ^(0|[1-9][0-9]*)$ ]] || fail 'Warmup and seed must be nonnegative integers.'
done
(( first_port <= 65528 )) || fail 'MASTER_PORT must leave room for eight ports (maximum 65528).'
[[ "$precision" == fp32 || "$precision" == bf16 ]] || fail 'PRECISION must be fp32 or bf16.'
[[ "$zero_stage" =~ ^[123]$ ]] || fail 'ZERO_STAGE must be 1, 2 or 3.'
[[ "$synthetic" =~ ^[01]$ ]] || fail 'SYNTHETIC must be 0 or 1.'
[[ "$run_id" =~ ^[a-zA-Z0-9_-]+$ ]] || fail 'RUN_ID may contain only letters, numbers, underscores and hyphens.'
# V4 has the smallest global batch (64); V1--V3 use 128.
(( 64 % (2 * gpus * micro_batch) == 0 )) || fail '2 * GPUS_PER_NODE * MICRO_BATCH_SIZE must divide 64.'
if (( ! dry_run )); then
  command -v "$launcher" >/dev/null || fail "Launcher not found: $launcher (run uv sync or set TORCHRUN)."
fi
output_dir="$output_root/$run_id"
log_dir="$output_dir/logs/node$node_rank"
if (( ! dry_run )); then mkdir -p "$log_dir"; fi
printf 'Node %s: rendezvous=%s, GPUs/node=%s, precision=%s, output=%s\n' \
  "$node_rank" "$master_addr" "$gpus" "$precision" "$output_dir"
index=0
for backend in deepspeed fsdp; do
  for variant in 1 2 3 4; do
    name="${backend}_v${variant}"
    # A distinct rendezvous port prevents adjacent jobs from sharing a store.
    port=$((first_port + index))
    cmd=("$launcher" --nnodes=2 "--nproc-per-node=$gpus" "--node-rank=$node_rank"
         "--master-addr=$master_addr" "--master-port=$port" --max-restarts=0
         "$script_dir/train_${backend}.py"
         --config "$repo_root/sample_configs/distributed_v${variant}.json"
         --micro-batch-size "$micro_batch" --precision "$precision"
         --warmup-steps "$warmup" --steps "$steps" --seed "$seed"
         --output "$output_dir/$name.json")
    if [[ "$backend" == deepspeed ]]; then cmd+=(--zero-stage "$zero_stage"); fi
    if [[ "$synthetic" == 1 ]]; then cmd+=(--synthetic); fi
    if [[ -n "${DATA_ROOT:-}" ]]; then cmd+=(--data-root "$DATA_ROOT"); fi
    printf '\n[%d/8] %s\n' "$((index + 1))" "$name"
    printf '%q ' "${cmd[@]}"
    printf '\n'
    if (( ! dry_run )); then
      if "${cmd[@]}" 2>&1 | tee "$log_dir/$name.log"; then
        :
      else
        echo "Failed: $name. See $log_dir/$name.log. Stop the peer launcher before restarting both nodes." >&2
        exit 1
      fi
    fi
    index=$((index + 1))
  done
done
if (( dry_run )); then
  echo 'Dry run complete; no processes launched or output files written.'
else
  echo "All eight benchmarks completed. JSON results are on machine 1 in $output_dir."
fi
