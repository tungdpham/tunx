#!/usr/bin/env python3
"""Run PyTorch/TunX training comparisons and generate raw and rolling plots."""
import argparse
from datetime import datetime
import math
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path,
                        help='New output directory (default: dump/convergence_TIMESTAMP)')
    parser.add_argument('--models', nargs='+', choices=['resnet50', 'gpt2'],
                        default=['resnet50', 'gpt2'])
    parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument('--rolling-window', type=int, default=100)
    parser.add_argument('--resnet-batch-size', type=int, default=8)
    parser.add_argument('--gpt2-batch-size', type=int, default=1)
    parser.add_argument('--engine', choices=['default', 'cuda', 'cudnn'], default='default')
    parser.add_argument('--preload-images', action='store_true')
    parser.add_argument('--preload-image-budget-gib', type=float, default=20)
    args = parser.parse_args()
    if not 1 <= args.rolling_window <= args.steps:
        parser.error('--rolling-window must be between 1 and --steps')
    if min(args.resnet_batch_size, args.gpt2_batch_size) <= 0:
        parser.error('Batch sizes must be positive')
    if not math.isfinite(args.preload_image_budget_gib) or args.preload_image_budget_gib <= 0:
        parser.error('Preload memory budget must be finite and positive')

    scripts = Path(__file__).resolve().parent
    project = scripts.parent
    binary = project / 'build/bin/test_convergence'
    if not os.access(binary, os.X_OK):
        parser.error(f'Build the test_convergence target first; executable missing: {binary}')
    output = (args.output_root or project / 'dump' /
              f'convergence_{datetime.now():%Y%m%d_%H%M%S_%f}').resolve()
    # A fresh directory avoids stale image caches or overwriting earlier runs.
    if output.exists():
        parser.error(f'Output directory already exists; choose a new path: {output}')
    output.mkdir(parents=True)
    env = os.environ.copy()
    env['PYTHONUNBUFFERED'] = '1'
    env['MPLBACKEND'] = 'Agg'
    env.pop('TUNX_COMPARE_FP64', None)

    print(f'Results: {output}\nSeed: 42; FP32; debug dumps disabled.', flush=True)
    for model in dict.fromkeys(args.models):
        batch_size = args.resnet_batch_size if model == 'resnet50' else args.gpt2_batch_size
        command = [sys.executable, str(scripts / 'run_convergence_tests.py'),
                   '--model', model, '--steps', str(args.steps),
                   '--batch-size', str(batch_size), '--engine', args.engine,
                   '--output-root', str(output)]
        if model == 'resnet50' and args.preload_images:
            command.extend(['--preload-images', '--preload-image-budget-gib',
                            str(args.preload_image_budget_gib)])
        log_path = output / f'{model}.log'
        print(f'Running {model}, batch size {batch_size}. Progress log: {log_path}', flush=True)
        with log_path.open('w') as log:
            result = subprocess.run(command, cwd=project, env=env, stdout=log,
                                    stderr=subprocess.STDOUT)
        if result.returncode:
            print(f'Training failed; see {log_path}', file=sys.stderr)
            return 1
        subprocess.run([
            sys.executable, str(scripts / 'plot_all_convergence.py'),
            '--models', model, '--input-root', str(output), '--steps', str(args.steps),
            '--seeds', '42', '--rolling-window', str(args.rolling_window),
        ], cwd=project, env=env, check=True)
    print(f'Completed training and plotting. Plots: {output / "plots"}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
