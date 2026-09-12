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

```bash
uv run python torch_benchmark/dump_utils.py --model resnet50 --dump-dir dump_pt --batch-size 32

./build/bin/test_equivalence --model resnet50 --pt-dir dump_pt --tunx-dir dump_tunx --batch-size 32

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

./build/bin/test_block_equivalence \
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

	./build/bin/test_block_equivalence \
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

For a reproducible multi-seed run across ResNet and all isolated blocks, use
`./run_multiseed_equivalence.sh`. Set `SEEDS`, `BLOCK_BATCH_SIZE`, and `RESULTS_DIR`
to customize the sweep; generated dump directories are ignored by Git and can be
deleted after the run to reclaim disk space.