## Configuring Exposed Parameters

If you plan to use the trainers with environment file, to change the parameters, create a .env in root directory and change the params there.

If you plan to use the trainers or inferencer with json config, create one following examples in the configs directory. More detailed parameter documentation will be available in the docs in the future.

See .env.example in root directory for available tunable parameters. For more in-depth tuning, you need to change the code.

## Running Coordinator and Network Workers

Running the coordinator is similar, but first run the workers, which will receive commands from the coordinator.

```bash
# Run worker over TCP/IPv4 (default)
./bin/tcp_worker 8001

# Run worker with CUDA
./bin/tcp_worker 8001 --gpu 

# Run network worker with custom number of IO threads (default is 4)
./bin/tcp_worker 8001 --io-threads 8

# Run network worker with custom number of worker threads (default is 8)
./bin/tcp_worker 8001 --num-threads 4

# Use multiple options
./bin/tcp_worker 8001 --gpu --io-threads 8 --num-threads 16

# Run worker with RDMA (default)
./bin/roce_worker --port 8001 --device {device_id}
```

Then, run coordinator after all worker:

```bash
# Run coordinator over TCP/IPv4 (default)
./bin/tcp_coordinator

# Run coordinator with RDMA (default)
./bin/roce_coordinator --device {device_id}
```


## Running Equivalence Tests

> [!IMPORTANT]
> **Numerical Precision (TF32 vs FP32)**
> When running TunX on Ampere-class GPUs (e.g., RTX 30/40 series) with `cuBLAS`/`cuDNN`, NVIDIA drivers enable **TF32** by default. TF32 reduces the mantissa to 10 bits, whereas PyTorch benchmark scripts typically force strict 23-bit FP32 for reproducibility. This precision difference can cause a cascading divergence (up to 40% relative gradient difference after one step, especially in BatchNorm layers). 
> 
> To guarantee mathematically identical equivalence with PyTorch, you must disable TF32 in your environment before running TunX:
> `export NVIDIA_TF32_OVERRIDE=0`

```bash
uv run python torch_benchmark/dump_utils.py --model resnet50 --dump-dir dump_pt --batch-size 32

NVIDIA_TF32_OVERRIDE=0 ./build/bin/test_equivalence --model resnet50 --pt-dir dump_pt --tunx-dir dump_tunx --batch-size 32

uv run python torch_benchmark/compare_equivalence.py --pt_dir dump_pt --tunx_dir dump_tunx > comparison_output.txt 
```

## Running Test Block Equivalence

The block harness compares an isolated TunX block against its PyTorch reference,
including the forward output, parameter gradients, and updated parameters. Supported
blocks are `residual`, `inception`, `attention`, and `gpt2`.

Build the block-equivalence executable first:

```bash
cmake -S . -B build
cd build && make -j8 test_block_equivalence
```

Run one block from the repository root. The PyTorch and TunX dump directories must be
different:

```bash
uv run python torch_benchmark/dump_block_utils.py \
	--block residual \
	--dump-dir dump_block_pt \
	--batch-size 2 \
	--seed 42

NVIDIA_TF32_OVERRIDE=0 ./build/bin/test_block_equivalence \
	--block residual \
	--pt-dir dump_block_pt \
	--tunx-dir dump_block_tunx \
	--batch-size 2

uv run python torch_benchmark/compare_equivalence.py \
	--pt_dir dump_block_pt \
	--tunx_dir dump_block_tunx \
	> block_comparison_output.txt
```

To run all supported blocks, use separate directories for each block:

```bash
for block in residual inception attention gpt2; do
	uv run python torch_benchmark/dump_block_utils.py \
		--block "$block" \
		--dump-dir "dump_blocks/$block/pt" \
		--batch-size 2 \
		--seed 42

	NVIDIA_TF32_OVERRIDE=0 ./build/bin/test_block_equivalence \
		--block "$block" \
		--pt-dir "dump_blocks/$block/pt" \
		--tunx-dir "dump_blocks/$block/tunx" \
		--batch-size 2

	uv run python torch_benchmark/compare_equivalence.py \
		--pt_dir "dump_blocks/$block/pt" \
		--tunx_dir "dump_blocks/$block/tunx" \
		> "dump_blocks/$block/comparison_output.txt"
done
```

## Running Convergence Tests

The convergence test suite runs a multi-step training loop to compare the trajectory of the model in PyTorch and TunX. To run the tests:

```bash
NVIDIA_TF32_OVERRIDE=0 uv run python torch_benchmark/run_convergence_tests.py --model resnet50 --steps 1000
uv run python torch_benchmark/plot_convergence.py --model resnet50 --steps 1000
```

Image models and batches use PyTorch's channels-last memory format to match TunX's
NHWC storage. To isolate the model backward pass from the loss backward calculation:

```bash
cmake --build build --target test_convergence -j 2
NVIDIA_TF32_OVERRIDE=0 uv run python torch_benchmark/run_convergence_tests.py \
    --model resnet50 --steps 1 --debug-compare --shared-loss-gradient
```

This dumps PyTorch's `dLoss/dOutputs`, loads it into TunX before backward, and
verifies the supplied gradient bytes match. TunX also dumps its native loss
gradient as `native_loss_gradient_step_1.bin`. The frameworks still use their own
forward activations and ReLU masks, so shared output gradients do not guarantee
identical parameter gradients. Use one step to compare identical initial
parameters; later steps can already have different parameter states.

> [!WARNING]
> **Parameter Drift in Convergence**
> When running convergence tests (which run multiple forward/backward passes and optimizer steps), you will likely observe immediate parameter drift (divergence) between PyTorch and TunX by Step 2 or 3, even if the inputs and labels match perfectly in Step 1.
> 
> Our investigation has isolated the following root causes for this drift:
> 1. **TF32 Precision**: As noted above, CuDNN enables TF32 by default on Ampere+ GPUs, which causes a significant loss of precision (10-bit mantissa instead of 23-bit). This causes up to a 40% relative error in gradients propagated to the early layers of ResNet50 in just the first step. **This can be mitigated by prefixing your command with `NVIDIA_TF32_OVERRIDE=0`**.
> 2. **Forward Rounding and ReLU Masks**: Deterministic execution does not guarantee identical arithmetic across layouts or execution plans. Small forward differences can change the sign of activations near zero, causing ReLU to pass a gradient in one framework and block it in the other. Channels-last makes the tested stem convolution match exactly, but small BatchNorm differences remain.
> 3. **Error Accumulation**: Different gradients produce different optimizer updates and hence different forward states on subsequent steps. In the tested seed-42, batch-8 ResNet50 run, losses first exceeded both `1e-3` absolute and relative tolerances at step 3. Changing only PyTorch's memory layout also produced rapidly diverging parameter gradients, so trajectory drift alone does not establish a backward implementation bug.
