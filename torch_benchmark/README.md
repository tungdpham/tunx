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
