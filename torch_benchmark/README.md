# Convergence trajectories with bounded disk use

Run both frameworks from the same initial parameters and sampled inputs, without
debug activation, gradient, or per-step parameter dumps:

```bash
uv run python torch_benchmark/run_convergence_tests.py --model resnet50 --steps 1000 --batch-size 8 --output-root dump/convergence_1000
uv run python torch_benchmark/run_convergence_tests.py --model gpt2 --steps 1000 --batch-size 1 --output-root dump/convergence_1000
uv run python torch_benchmark/plot_convergence.py --model resnet50 --steps 1000 --seeds 42 --input-root dump/convergence_1000 --output-dir dump/convergence_1000/plots
uv run python torch_benchmark/plot_convergence.py --model gpt2 --steps 1000 --seeds 42 --input-root dump/convergence_1000 --output-dir dump/convergence_1000/plots
```

The runner currently uses seed 42. These are training-batch metrics, not held-out
validation metrics. A single seed does not establish statistical equivalence or
prove that numerical drift is inevitable.

## Generate all plots on a remote server

To **run training and generate every plot** in one command:

```bash
uv run python torch_benchmark/run_and_plot_convergence.py \
  --output-root dump/convergence_remote \
  --steps 1000 --rolling-window 100
```

This sequentially runs PyTorch and TunX for each model, using seed 42, FP32,
ResNet50 batch size 8 and GPT-2 batch size 1, with debug dumps disabled. Each
model is plotted immediately after its training comparison finishes. The output
directory must be new; omit `--output-root` to create a timestamped directory.
Training output is saved to `OUTPUT_ROOT/resnet50.log` and `gpt2.log`; use
`tail -f` to monitor them. All figures go to `OUTPUT_ROOT/plots`.

The remote server needs the project Python environment (`uv sync`), `uv` on
PATH, a working CUDA setup, and the built `build/bin/test_convergence` executable
(`cmake --build build --target test_convergence` for an already configured build).
Dataset locations default to `data/imagenet-100` and `data/open-web-text/train.bin`;
override them with `IMAGENET100_ROOT` and `OPENWEBTEXT_PATH` environment variables.
Optional flags: `--resnet-batch-size`, `--gpt2-batch-size`, `--engine`,
`--models resnet50` (or `gpt2`), and `--preload-images` (ResNet50 only).

From the repository root, using completed trajectory CSVs:

```bash
uv run python torch_benchmark/plot_all_convergence.py \
  --input-root dump/convergence_1000_20260921 \
  --steps 1000 --seeds 42 --rolling-window 100
```

This generates raw and rolling loss plots for both models, ResNet50 accuracy,
and GPT-2 perplexity. Raw plots are PNG; rolling plots are PNG and PDF. Rolling
means use full trailing windows, so a 100-step window begins at step 100.
Output defaults to `INPUT_ROOT/plots`; override with `--output-dir PATH`.
Use `--models resnet50` or `--models gpt2` to plot only one model.
Missing CSVs or incomplete step sequences cause a nonzero exit.

No GPU, datasets, checkpoints, or training reruns are needed. Only the four CSVs
are needed for one seed; preserve their directory structure under `--input-root`:

```text
trajectory_resnet50_42/pt_trajectory_seed_42.csv
tunx_trajectory_resnet50_42/tunx_trajectory_seed_42.csv
trajectory_gpt2_42/pt_trajectory_seed_42.csv
tunx_trajectory_gpt2_42/tunx_trajectory_seed_42.csv
```

For a lightweight plotting-only environment instead of the full project environment:

```bash
python3 -m venv .venv-plots
.venv-plots/bin/python -m pip install numpy pandas matplotlib
.venv-plots/bin/python torch_benchmark/plot_all_convergence.py \
  --input-root /path/to/convergence_results --rolling-window 100
```

The script selects the noninteractive Matplotlib backend automatically. Copy both
`plot_all_convergence.py` and `plot_convergence.py` together if using them outside
this repository.

## ImageNet100 loading and shuffling

The full sample catalog is indexed before sampling. ResNet50 samples a seeded
permutation without replacement and reshuffles after each complete dataset pass.
At batch size 8, 1,000 steps visit 8,000 images, not the entire training set.

The default FP32, no-debug path lazily decodes only visited images, storing each
224x224 RGB crop once as uint8 with its int32 label. Both frameworks reconstruct
identical normalized float32 values using a shared lookup table. This avoids
cross-library JPEG/resize differences and costs 150,532 bytes per unique image:
about 1.12 GiB for 8,000 images, versus 4.49 GiB for one float32 input stream.
Initial parameters and small index/metric files are additional. No per-step TunX
input copy is created. Cache files are reusable within the trajectory, including
after an epoch reshuffle; use a fresh output root when changing the dataset or
preprocessing to avoid stale caches.

To load all source images before shuffling, add:

```bash
--preload-images --preload-image-budget-gib 20
```

This opt-in mode reads the original compressed files into RAM before generating
the permutation, then decodes and transforms selected samples on demand. It adds
no disk copy and preserves the sample indices and transformed tensors. The budget
covers encoded bytes only; leave RAM for Python, batch decoding, and training.
Preloading fails before reading image contents if their total exceeds the budget.
The default lazy mode is preferable for short runs that sample only a fraction
of the dataset. Loading all preprocessed crops instead would require roughly
18.2 GiB as uint8 or 72.9 GiB as float32 for 130,000 images, before overhead.

For longer campaigns, a possible next step is a shared packed crop store with an
index and dataset/preprocessing fingerprint. It would remove per-file overhead
and duplication between seeds; lossless compression could reduce bytes further
at the cost of decoding. Neither requires retaining activations or checkpoints.

GPT-2 keeps OpenWebText memory-mapped and shares sampled int64 token offsets with
TunX; it does not preload or duplicate the corpus.

## DeepSpeed ZeRO and PyTorch FSDP benchmarks (V1–V4)

`train_deepspeed.py` and `train_fsdp.py` reuse the models in `torch_trainer.py`
through `distributed_benchmark.py`. Both perform forward, cross-entropy backward,
and Adam/AdamW updates on a cached, GPU-resident global batch. Defaults are 50
warmup updates and 2,000 measured updates. These are throughput benchmarks, not
convergence training: the cached batch repeats and learning rate stays constant.

Use the project environment (`uv sync`); it includes PyTorch, torchvision and
DeepSpeed. Launch from the repository root with one process per CUDA GPU:

```bash
.venv/bin/torchrun --standalone --nproc-per-node=2 torch_benchmark/train_deepspeed.py \
  --config sample_configs/distributed_v1.json --micro-batch-size 8 \
  --precision bf16 --zero-stage 3 --output benchmark_results/deepspeed_v1.json

.venv/bin/torchrun --standalone --nproc-per-node=2 torch_benchmark/train_fsdp.py \
  --config sample_configs/distributed_v1.json --micro-batch-size 8 \
  --precision bf16 --output benchmark_results/fsdp_v1.json
```

Change `distributed_v1.json` to v2, v3 or v4 and give each run a distinct output
path (existing results at that path are overwritten). ImageNet100 is read from
`--data-root`, then `IMAGENET100_ROOT`, then the config's `dataset_path` resolved
relative to its directory. Each rank caches a disjoint portion of the same seeded
sample permutation. Add `--synthetic` for a dataset-free benchmark with random
224×224 images and 100-class labels; report synthetic results separately.

For two machines with one GPU each, run the following on **both** machines,
setting `NODE_RANK=0` on the first and `NODE_RANK=1` on the second. Set
`MASTER_ADDR` to the reachable address of the first machine:

```bash
export MASTER_ADDR=10.10.0.2
export NODE_RANK=0  # use 1 on the second machine
.venv/bin/torchrun --nnodes=2 --nproc-per-node=1 --node-rank="$NODE_RANK" \
  --master-addr="$MASTER_ADDR" --master-port=29500 \
  torch_benchmark/train_deepspeed.py --config sample_configs/distributed_v1.json \
  --micro-batch-size 8 --precision bf16 --output benchmark_results/deepspeed_v1.json
```

Replace the script with `train_fsdp.py` for FSDP. Install the same environment and
code on each machine; real data must be available on each. NCCL handles GPU
communication; if necessary set `NCCL_SOCKET_IFNAME` to the connected interface.
The TunX JSON worker endpoints and partition policies are not used by torchrun.

Global batch defaults to the JSON `batch_size` (128 for V1–V3; 64 for V4).
`global_batch = world_size × micro_batch_size × accumulation_steps`; accumulation
is derived and non-divisible values are rejected. `--global-batch-size` overrides
it. TunX's `num_microbatches` describes pipeline scheduling and is not interpreted
as data-parallel accumulation. To match a TunX microbatch of 32 at global batch
128 on two GPUs, use `--micro-batch-size 32` (two accumulation steps per rank).
Reduce the microbatch if activation memory exceeds GPU capacity.

Benchmark details and limits:

- DeepSpeed defaults to ZeRO stage 3; `--zero-stage 1` or `2` is also supported.
  FSDP uses FULL_SHARD with size-based wrapping at 100,000 parameters. FSDP
  synchronizes gradients on every microbatch to avoid retaining full gradients.
- Both use the config's optimizer hyperparameters, no clipping, no scheduler,
  no activation checkpointing, no offload and no compilation. BatchNorm uses
  local microbatch statistics, so differing microbatch sizes affect training.
- FP32 is the default, with TF32 disabled. BF16 uses each framework's mixed
  precision implementation, FP32 gradient communication, and cross-entropy on
  FP32 logits. These do not reproduce TunX's BF16 storage/FP32 compute contract;
  optimizer-state and BatchNorm precision can differ across backends.
- Timing excludes input loading, initialization, warmup and result collection.
  CUDA is synchronized before/after the measured loop; throughput uses the
  slowest rank's elapsed time and counts global samples once per optimizer update.
- Rank 0 writes JSON with throughput, time per update, versions, configuration,
  final losses and each rank's peak allocated/reserved CUDA bytes. Allocator
  peaks exclude some NCCL/driver memory and are not process VRAM measurements.
- These are sharded data-parallel baselines for TunX's pipeline benchmark, not
  identical parallelization strategies. Each rank initially constructs the full
  model, and activation memory remains local. Heterogeneous GPUs wait for the
  slowest rank. Keep batch, precision, input mode and hardware fixed for comparisons.

A short CUDA smoke test (repeat with either script):

```bash
.venv/bin/torchrun --standalone --nproc-per-node=1 torch_benchmark/train_fsdp.py \
  --synthetic --global-batch-size 2 --micro-batch-size 1 \
  --warmup-steps 1 --steps 2 --output /tmp/fsdp_smoke.json
```

API references: [PyTorch FSDP](https://docs.pytorch.org/docs/stable/fsdp.html)
and [DeepSpeed initialization](https://deepspeed.readthedocs.io/en/stable/initialize.html).

### Run all eight distributed benchmarks

Run `run_distributed_benchmarks.sh` on **both machines**. It runs DeepSpeed
V1–V4, then FSDP V1–V4, sequentially, with one GPU per machine by default.
Both nodes use **10.10.0.2** as `MASTER_ADDR`; **10.10.0.1** is the worker node,
not a second rendezvous master.

On machine 1 (`10.10.0.2`):

```bash
RUN_ID=all_bf16 PRECISION=bf16 bash torch_benchmark/run_distributed_benchmarks.sh 0
```

On machine 2 (`10.10.0.1`):

```bash
RUN_ID=all_bf16 PRECISION=bf16 bash torch_benchmark/run_distributed_benchmarks.sh 1
```

Use identical settings on both machines, except node rank and local data paths.
The launcher defaults to FP32 when `PRECISION` is unset. Each job uses a distinct
rendezvous port, 29500–29507 by default; these must be reachable on machine 1.
NCCL must also be able to communicate between machines. Set
`NCCL_SOCKET_IFNAME` on each node if interface selection is needed.

Results appear on machine 1 under `benchmark_results/all_bf16/`, with one JSON
per backend/model. Each machine saves its own logs under `logs/node0/` or
`logs/node1/`. Use a fresh `RUN_ID` to retain previous results. Failures stop the
local suite; stop the peer launcher before restarting both machines.

For a short dataset-free run, execute this on both machines, replacing `0` with
`1` on machine 2:

```bash
RUN_ID=smoke SYNTHETIC=1 WARMUP_STEPS=1 STEPS=2 \
  bash torch_benchmark/run_distributed_benchmarks.sh 0
```

Use `--dry-run` after the rank to inspect all eight commands without launching.
`--help` lists overrides including `DATA_ROOT`, `GPUS_PER_NODE`,
`MICRO_BATCH_SIZE`, `ZERO_STAGE`, and `TORCHRUN`.
