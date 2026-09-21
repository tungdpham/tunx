#!/usr/bin/env python3
"""Generate raw and rolling convergence plots from existing trajectory CSVs."""
import argparse
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-root', type=Path, required=True,
                        help='Directory containing trajectory_* and tunx_trajectory_* folders')
    parser.add_argument('--output-dir', type=Path,
                        help='Defaults to INPUT_ROOT/plots')
    parser.add_argument('--models', nargs='+', default=['resnet50', 'gpt2'],
                        choices=['resnet50', 'gpt2'])
    parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument('--seeds', nargs='+', type=int, default=[42])
    parser.add_argument('--rolling-window', type=int, default=100)
    args = parser.parse_args()
    if not 1 <= args.rolling_window <= args.steps:
        parser.error('--rolling-window must be between 1 and --steps')

    input_root = args.input_root.resolve()
    output_dir = (args.output_dir or input_root / 'plots').resolve()
    missing = []
    for model in args.models:
        for seed in args.seeds:
            for folder, prefix in [('trajectory', 'pt'), ('tunx_trajectory', 'tunx')]:
                path = input_root / f'{folder}_{model}_{seed}' / f'{prefix}_trajectory_seed_{seed}.csv'
                if not path.is_file():
                    missing.append(str(path))
    if missing:
        parser.error('Missing required trajectory CSVs:\n' + '\n'.join(missing))

    env = os.environ.copy()
    env['MPLBACKEND'] = 'Agg'  # Works on remote servers without a display.
    plotter = Path(__file__).resolve().with_name('plot_convergence.py')
    for model in args.models:
        print(f'Generating raw and {args.rolling_window}-step rolling plots for {model}...', flush=True)
        subprocess.run([
            sys.executable, str(plotter), '--model', model,
            '--steps', str(args.steps), '--seeds', *map(str, args.seeds),
            '--rolling-window', str(args.rolling_window),
            '--input-root', str(input_root), '--output-dir', str(output_dir),
        ], check=True, env=env)
    print(f'All plots saved to {output_dir}')


if __name__ == '__main__':
    main()
