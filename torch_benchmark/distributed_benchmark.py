"""Shared device-resident training benchmark for the TunX V1--V4 models."""
import argparse
import json
import os
from pathlib import Path
import random
import socket
import time


def parse_args(backend):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('sample_configs/distributed_v1.json'))
    parser.add_argument('--global-batch-size', type=int)
    parser.add_argument('--micro-batch-size', type=int, default=8, help='Samples per GPU per forward/backward')
    parser.add_argument('--warmup-steps', type=int, default=50)
    parser.add_argument('--steps', type=int, default=2000, help='Measured optimizer updates')
    parser.add_argument('--precision', choices=['fp32', 'bf16'], default='fp32')
    parser.add_argument('--synthetic', action='store_true', help='Use random images instead of ImageNet100')
    parser.add_argument('--data-root', type=Path, help='Overrides IMAGENET100_ROOT and config dataset_path')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', type=Path, default=Path(f'benchmark_results/{backend}.json'))
    parser.add_argument('--local-rank', '--local_rank', type=int, default=0)
    if backend == 'deepspeed':
        parser.add_argument('--zero-stage', type=int, choices=[1, 2, 3], default=3)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if config.get('model_name') not in {f'tunx_v{i}' for i in range(1, 5)}:
        parser.error('config model_name must be tunx_v1, tunx_v2, tunx_v3, or tunx_v4')
    world = int(os.environ.get('WORLD_SIZE', '1'))
    batch = args.global_batch_size if args.global_batch_size is not None else config['batch_size']
    if batch <= 0 or args.micro_batch_size <= 0 or args.steps <= 0 or args.warmup_steps < 0:
        parser.error('batch sizes and steps must be positive; warmup must be nonnegative')
    if batch % (world * args.micro_batch_size):
        parser.error('global batch must be divisible by WORLD_SIZE * micro batch')
    args.global_batch_size = batch
    args.accumulation_steps = batch // (world * args.micro_batch_size)
    return args, config


def main(backend):
    args, config = parse_args(backend)
    import numpy as np
    import torch
    import torch.distributed as dist
    import torch.nn.functional as F
    from torch_trainer import get_model_config

    if not torch.cuda.is_available():
        raise RuntimeError('This benchmark requires CUDA; launch with torchrun.')
    local_rank = int(os.environ.get('LOCAL_RANK', args.local_rank))
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda', local_rank)
    if args.precision == 'bf16' and not torch.cuda.is_bf16_supported():
        raise RuntimeError('This GPU does not support BF16.')
    dist.init_process_group('nccl')
    try:
        rank, world = dist.get_rank(), dist.get_world_size()
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        if args.data_root:
            os.environ['IMAGENET100_ROOT'] = str(args.data_root.resolve())
        elif 'IMAGENET100_ROOT' not in os.environ:
            os.environ['IMAGENET100_ROOT'] = str((args.config.parent / config['dataset_path']).resolve())
        cfg = get_model_config(config['model_name'])
        model = cfg['model_cls']()
        parameter_count = sum(p.numel() for p in model.parameters())
        opt = config['optimizer']
        if opt['type'].lower() not in ('adam', 'adamw'):
            raise ValueError('Only Adam and AdamW configs are supported')
        optimizer_cls = torch.optim.AdamW if opt['type'].lower() == 'adamw' or opt.get('decouple_weight_decay', False) else torch.optim.Adam
        def make_optimizer(parameters):
            return optimizer_cls(parameters, lr=opt['learning_rate'],
                                 betas=(opt.get('beta1', .9), opt.get('beta2', .999)),
                                 eps=opt.get('epsilon', 1e-8), weight_decay=opt.get('weight_decay', 0), foreach=False)

        dtype = torch.bfloat16 if args.precision == 'bf16' else torch.float32
        ds_config = None
        ds_version = None
        if backend == 'fsdp':
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, MixedPrecision
            from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
            from functools import partial
            model = FSDP(model, device_id=device, sync_module_states=True,
                         auto_wrap_policy=partial(size_based_auto_wrap_policy, min_num_params=100_000),
                         mixed_precision=MixedPrecision(param_dtype=dtype, reduce_dtype=torch.float32,
                                                        buffer_dtype=dtype), use_orig_params=True)
            optimizer = make_optimizer(model.parameters())
        else:
            import deepspeed
            ds_version = deepspeed.__version__
            # DeepSpeed versions may preserve FP32 buffers while casting weights.
            # BatchNorm backward requires its running statistics to match BF16 weights.
            if args.precision == 'bf16':
                for module in model.modules():
                    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                        module.to(dtype=dtype)
            ds_config = {
                'train_batch_size': args.global_batch_size,
                'train_micro_batch_size_per_gpu': args.micro_batch_size,
                'gradient_accumulation_steps': args.accumulation_steps,
                'zero_optimization': {'stage': args.zero_stage},
                'bf16': {'enabled': args.precision == 'bf16'},
                'communication_data_type': 'fp32',
                'gradient_clipping': 0.0,
                'steps_per_print': args.steps + args.warmup_steps + 1,
                'wall_clock_breakdown': False,
            }
            model, optimizer, _, _ = deepspeed.initialize(
                model=model, optimizer=make_optimizer(model.parameters()), config=ds_config,
                dist_init_required=False)
        model.train()

        # Cache one disjoint local share of a global batch. Input preparation is untimed.
        local_batch = args.global_batch_size // world
        random.seed(args.seed + rank)
        np.random.seed(args.seed + rank)
        torch.manual_seed(args.seed + rank)
        if args.synthetic:
            inputs = torch.randn(local_batch, 3, 224, 224)
            targets = torch.randint(100, (local_batch,))
        else:
            dataset = cfg['train_dataset']()
            if len(dataset) < args.global_batch_size:
                raise ValueError('Dataset must contain at least one complete global batch')
            generator = torch.Generator().manual_seed(args.seed)
            indices = torch.randperm(len(dataset), generator=generator)[:args.global_batch_size]
            samples = [dataset[int(i)] for i in indices[rank * local_batch:(rank + 1) * local_batch]]
            inputs = torch.stack([x for x, _ in samples])
            targets = torch.tensor([y for _, y in samples], dtype=torch.long)
        inputs = inputs.to(device=device, dtype=dtype)
        targets = targets.to(device)
        batches = list(zip(inputs.split(args.micro_batch_size), targets.split(args.micro_batch_size)))

        def update():
            if backend == 'fsdp':
                optimizer.zero_grad(set_to_none=True)
            losses = torch.zeros((), device=device)
            for x, y in batches:
                logits = model(x)
                loss = F.cross_entropy(logits.float(), y)
                losses += loss.detach() / args.accumulation_steps
                if backend == 'deepspeed':
                    model.backward(loss)  # DeepSpeed scales loss for accumulation.
                    model.step()
                else:
                    (loss / args.accumulation_steps).backward()
            if backend == 'fsdp':
                optimizer.step()
            return losses

        if rank == 0:
            print(f'{backend}: {config["model_name"]}, world={world}, global batch={args.global_batch_size}, '
                  f'micro batch={args.micro_batch_size}, accumulation={args.accumulation_steps}', flush=True)
        for _ in range(args.warmup_steps):
            update()
        torch.cuda.synchronize()
        dist.barrier()
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        for _ in range(args.steps):
            loss = update()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        allocated = torch.cuda.max_memory_allocated()
        reserved = torch.cuda.max_memory_reserved()
        local = {'rank': rank, 'hostname': socket.gethostname(), 'gpu': torch.cuda.get_device_name(),
                 'elapsed_seconds': elapsed, 'peak_allocated_bytes': allocated,
                 'peak_reserved_bytes': reserved, 'final_loss': loss.item()}
        results = [None] * world
        dist.all_gather_object(results, local)
        if not all(np.isfinite(r['final_loss']) for r in results):
            raise RuntimeError('Nonfinite final loss; benchmark is invalid')
        if rank == 0:
            seconds = max(r['elapsed_seconds'] for r in results)
            report = {
                'backend': backend, 'model': config['model_name'], 'world_size': world,
                'global_batch_size': args.global_batch_size, 'micro_batch_size': args.micro_batch_size,
                'gradient_accumulation_steps': args.accumulation_steps,
                'warmup_steps': args.warmup_steps, 'measured_steps': args.steps,
                'precision': args.precision, 'data': 'synthetic' if args.synthetic else 'cached_imagenet100',
                'data_root': None if args.synthetic else os.environ['IMAGENET100_ROOT'],
                'parallelism': 'sharded_data_parallel',
                'fsdp_config': ({'sharding_strategy': 'FULL_SHARD' if world > 1 else 'NO_SHARD',
                                 'min_num_params': 100_000, 'use_orig_params': True}
                                if backend == 'fsdp' else None),
                'seed': args.seed, 'parameters': parameter_count, 'optimizer': opt,
                'scheduler': 'constant_lr', 'torch_version': torch.__version__,
                'cuda_version': torch.version.cuda, 'deepspeed_version': ds_version,
                'deepspeed_config': ds_config, 'source_config': config,
                'elapsed_seconds': seconds, 'step_ms': seconds * 1000 / args.steps,
                'samples_per_second': args.steps * args.global_batch_size / seconds,
                'ranks': results,
            }
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2) + '\n')
            print(f'{report["samples_per_second"]:.2f} samples/s; {report["step_ms"]:.2f} ms/update; {args.output}')
    finally:
        dist.destroy_process_group()
