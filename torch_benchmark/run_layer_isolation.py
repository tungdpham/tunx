#!/usr/bin/env python3
"""Run isolated FP32 layer forward/backward comparisons with shared upstream gradients."""
import argparse
import json
import os
from pathlib import Path
import subprocess

import numpy as np
import torch
from torch import nn


def save(path, tensor):
    tensor = tensor.detach()
    if tensor.ndim == 4:
        tensor = tensor.permute(0, 2, 3, 1)
    tensor.contiguous().cpu().numpy().tofile(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', default='dump/layer_isolation')
    parser.add_argument('--engines', nargs='+', default=['default', 'cuda', 'cudnn'])
    parser.add_argument('--seeds', nargs='+', type=int, default=[42])
    parser.add_argument('--cases', nargs='+', help='Run only these named layer cases')
    parser.add_argument('--max-relative-l2', type=float, help='Fail if any tensor exceeds this relative L2 error')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    cases = {
        'conv_stem': ((8, 3, 224, 224), lambda: nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False)),
        'conv3': ((8, 64, 56, 56), lambda: nn.Conv2d(64, 64, 3, padding=1, bias=False)),
        'conv1': ((8, 64, 56, 56), lambda: nn.Conv2d(64, 256, 1, bias=False)),
        'batchnorm': ((8, 64, 56, 56), lambda: nn.BatchNorm2d(64)),
        'batchnorm_relu': ((8, 64, 56, 56), lambda: nn.Sequential(nn.BatchNorm2d(64), nn.ReLU())),
        'batchnorm_relu_odd': ((8, 63, 56, 56), lambda: nn.Sequential(nn.BatchNorm2d(63), nn.ReLU())),
        'maxpool': ((8, 64, 112, 112), lambda: nn.MaxPool2d(3, stride=2, padding=1)),
        'maxpool_relu': ((8, 64, 112, 112), lambda: nn.MaxPool2d(3, stride=2, padding=1)),
        'relu': ((8, 64, 56, 56), nn.ReLU),
        'dense': ((8, 2048), lambda: nn.Linear(2048, 100)),
    }
    if args.cases:
        unknown = set(args.cases) - cases.keys()
        if unknown:
            parser.error(f'Unknown cases: {sorted(unknown)}')
        cases = {name: cases[name] for name in args.cases}
    results = []
    env = dict(os.environ, NVIDIA_TF32_OVERRIDE='0')
    for seed in args.seeds:
        for case, (shape, factory) in cases.items():
            torch.manual_seed(seed)
            folder = output / f'{case}_{seed}'
            pt = folder / 'pytorch'
            pt.mkdir(parents=True, exist_ok=True)
            layer = factory().cuda().train()
            x = torch.randn(shape, device='cuda')
            if case == 'maxpool_relu':
                x = x.relu()
            if case == 'relu':
                x.flatten()[::17] = 0  # Exercise the derivative at exactly zero.
            if x.ndim == 4:
                x = x.contiguous(memory_format=torch.channels_last)
                layer = layer.to(memory_format=torch.channels_last)
            x.requires_grad_()
            params = list(layer.parameters())
            for name, param in zip(['weight', 'bias'], params):
                if case.startswith('batchnorm'):
                    with torch.no_grad():
                        param.normal_()
                        param[0] = 0  # Exactly-zero BN channel checks ReLU boundary semantics.
                save(pt / f'{name}.bin', param)
            save(pt / 'inputs.bin', x)
            y = layer(x)
            dy = torch.randn_like(y)
            save(pt / 'outputs.bin', y)
            save(pt / 'grad_output.bin', dy)
            y.backward(dy)
            save(pt / 'input_grad.bin', x.grad)
            for name, param in zip(['weight', 'bias'], params):
                save(pt / f'{name}_grad.bin', param.grad)
            for engine in args.engines:
                dest = folder / engine
                dest.mkdir(exist_ok=True)
                with (dest / 'runner.log').open('w') as log:
                    run = subprocess.run([str(root / 'build/bin/test_layer_isolation'), case, engine, str(pt), str(dest)], env=env, stdout=log, stderr=subprocess.STDOUT)
                if run.returncode:
                    results.append(dict(case=case, seed=seed, engine=engine, error=f'runner exited {run.returncode}'))
                    print(f'{case} {engine}: runner failed; see {dest / "runner.log"}', flush=True)
                    continue
                for filename in ['outputs.bin', 'input_grad.bin'] + [f'{name}_grad.bin' for name in ['weight', 'bias'][:len(params)]]:
                    a = np.fromfile(pt / filename, dtype=np.float32).astype(np.float64)
                    b = np.fromfile(dest / filename, dtype=np.float32).astype(np.float64)
                    if a.shape != b.shape:
                        raise RuntimeError(f'{case}/{engine}/{filename}: {a.shape} != {b.shape}')
                    diff = np.abs(a-b)
                    row = dict(case=case, seed=seed, engine=engine, tensor=filename, max_abs=float(diff.max()), relative_l2=float(np.linalg.norm(diff)/max(np.linalg.norm(a),1e-30)), passed_pct=float(100*np.isclose(a,b,atol=1e-4,rtol=1e-4).mean()))
                    results.append(row)
                    print(f'{case:16} {engine:7} {filename:16} max={row["max_abs"]:.3e} L2={row["relative_l2"]:.3e} pass={row["passed_pct"]:.3f}%', flush=True)
            (output / 'summary.json').write_text(json.dumps(results, indent=2))
    print(f'Results: {output / "summary.json"}')
    failures = [r for r in results if 'error' in r or not np.isfinite(r['relative_l2'])
                or (args.max_relative_l2 is not None and r['relative_l2'] > args.max_relative_l2)]
    if failures:
        raise SystemExit(f'{len(failures)} failed comparisons; see summary.json')


if __name__ == '__main__':
    main()
