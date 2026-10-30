#!/usr/bin/env python3
"""Launch the 60-run TunX/ZeRO/FSDP campaign from machine 1; control machine 2 over SSH."""
import argparse
import csv
import json
import math
import os
from pathlib import Path
import re
import select
import shlex
import signal
import socket
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
BACKENDS = ('tunx', 'deepspeed', 'fsdp')


def stop(process):
    if process is not None and process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()


def remote_job(argv):
    """Internal SSH helper. EOF on SSH stdin cancels only this helper's child group."""
    p = argparse.ArgumentParser()
    p.add_argument('--log', required=True)
    p.add_argument('--ready-port', type=int)
    p.add_argument('command', nargs=argparse.REMAINDER)
    a = p.parse_args(argv)
    command = a.command[1:] if a.command[:1] == ['--'] else a.command
    Path(a.log).parent.mkdir(parents=True, exist_ok=True)
    child = None
    try:
        if a.ready_port is not None:
            with socket.socket() as probe:
                probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                probe.bind(('0.0.0.0', a.ready_port))
        with open(a.log, 'w') as log:
            child = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log,
                                     stderr=subprocess.STDOUT, start_new_session=True)
            ready = a.ready_port is None
            deadline = time.monotonic() + 120
            if ready:
                print('READY', flush=True)
            while child.poll() is None:
                if not ready:
                    # Inspect local LISTEN sockets without opening an unframed TunX connection.
                    port_hex = f'{a.ready_port:04X}'
                    for table in ('/proc/net/tcp', '/proc/net/tcp6'):
                        if Path(table).exists():
                            rows = Path(table).read_text().splitlines()[1:]
                            if any(r.split()[1].endswith(':' + port_hex) and r.split()[3] == '0A' for r in rows):
                                ready = True
                    if ready:
                        print('READY', flush=True)
                    elif time.monotonic() > deadline:
                        raise TimeoutError('TunX worker did not open its listening port')
                if select.select([sys.stdin], [], [], .2)[0] and not os.read(sys.stdin.fileno(), 1):
                    return 0
            return child.returncode
    finally:
        stop(child)


def summarize(directory):
    manifest = json.loads((directory / 'manifest.json').read_text())
    rows = []
    for model in manifest['models']:
        for backend in BACKENDS:
            values = []
            for seed in manifest['seeds']:
                path = directory / f'{backend}_v{model}_seed{seed}.json'
                if not path.exists():
                    raise ValueError(f'Incomplete campaign: missing {path.name}; no partial averages reported')
                result = json.loads(path.read_text())
                if (result['seed'], result['backend'], result['model']) != (seed, backend, f'tunx_v{model}'):
                    raise ValueError(f'Mismatched result metadata: {path}')
                if not all(math.isfinite(result[k]) and result[k] > 0 for k in ('samples_per_second', 'step_ms')):
                    raise ValueError(f'Invalid measurements: {path}')
                values.append(result)
            row = {'model': f'v{model}', 'backend': backend, 'n': len(values)}
            for metric in ('samples_per_second', 'step_ms'):
                series = [r[metric] for r in values]
                row[metric + '_mean'] = statistics.mean(series)
                row[metric + '_std'] = statistics.stdev(series)
            rows.append(row)
    with (directory / 'summary.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    (directory / 'summary.json').write_text(json.dumps(rows, indent=2) + '\n')
    lines = ['Model | Backend | n | Samples/s (mean ± SD) | ms/update (mean ± SD)',
             '--- | --- | ---: | ---: | ---:']
    for r in rows:
        lines.append(f"{r['model']} | {r['backend']} | {r['n']} | "
                     f"{r['samples_per_second_mean']:.2f} ± {r['samples_per_second_std']:.2f} | "
                     f"{r['step_ms_mean']:.2f} ± {r['step_ms_std']:.2f}")
    lines += ['', 'SD is the sample standard deviation (ddof=1), not the standard error.',
              'TunX streams batches and subtracts data-loading time; ZeRO/FSDP reuse a GPU-resident batch.',
              'Seeds control initialization and sample order; frameworks do not share identical weights or input tensors.',
              'Augmentation is disabled. BF16 storage/compute and optimizer-state policies differ by backend.',
              'TunX uses pipeline parallelism; ZeRO/FSDP use sharded data parallelism.']
    report = '\n'.join(lines) + '\n'
    (directory / 'summary.md').write_text(report)
    print(report)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--remote', default='10.10.0.1', help='SSH destination, optionally user@host')
    p.add_argument('--remote-repo', default=str(ROOT), help='Absolute repository path on machine 2')
    p.add_argument('--master-addr', default='10.10.0.2')
    p.add_argument('--worker-addr', default='10.10.0.1', help='TunX worker IP, independent of SSH alias')
    p.add_argument('--output', type=Path, default=ROOT / 'benchmark_results/distributed_5seeds')
    p.add_argument('--remote-output', help='Absolute log directory on machine 2')
    p.add_argument('--data-root', type=Path, default=ROOT / 'data/imagenet-100')
    p.add_argument('--remote-data-root', help='Absolute ImageNet100 path on machine 2')
    p.add_argument('--seeds', type=int, nargs=5, default=[1, 2, 3, 4, 5])
    p.add_argument('--models', type=int, nargs='+', choices=[1, 2, 3, 4], default=[1, 2, 3, 4])
    p.add_argument('--precision', choices=['fp32', 'bf16'], default='fp32')
    p.add_argument('--micro-batch-size', type=int, default=8)
    p.add_argument('--zero-stage', type=int, choices=[1, 2, 3], default=3)
    p.add_argument('--master-port', type=int, default=29500, help='First of 40 torchrun ports')
    p.add_argument('--timeout', type=float, default=7200, help='Maximum seconds per trial')
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--summarize-only', action='store_true')
    a = p.parse_args()
    a.output = a.output.resolve()
    if a.summarize_only:
        summarize(a.output)
        return
    if len(set(a.seeds)) != 5 or any(not 1 <= s <= 4294967295 for s in a.seeds):
        p.error('Provide five distinct seeds in [1, 4294967295]')
    if len(set(a.models)) != len(a.models):
        p.error('Models must be unique')
    if a.micro_batch_size <= 0 or 64 % (2 * a.micro_batch_size):
        p.error('2 * micro-batch-size must divide 64')
    if not 1 <= a.master_port <= 65496 or not math.isfinite(a.timeout) or a.timeout <= 0:
        p.error('Invalid port range or timeout')
    remote_root = Path(a.remote_repo)
    if not remote_root.is_absolute():
        p.error('--remote-repo must be absolute')
    remote_output = Path(a.remote_output or str(remote_root / 'benchmark_results' / a.output.name))
    remote_data = a.remote_data_root or str(remote_root / 'data/imagenet-100')
    if not remote_output.is_absolute() or not Path(remote_data).is_absolute():
        p.error('Remote paths must be absolute')
    if not a.dry_run:
        if a.output.exists():
            p.error('Output directory already exists; choose a fresh --output (or --summarize-only)')
        help_text = subprocess.check_output([ROOT / 'build/bin/tcp_coordinator', '--help'], text=True)
        if '--seed' not in help_text:
            p.error('Rebuild tcp_coordinator: this binary lacks --seed support')
        # SSH must be noninteractive. Remote helper must be the same revision.
        check = f'test -x {shlex.quote(str(remote_root / ".venv/bin/torchrun"))} && test -x {shlex.quote(str(remote_root / "build/bin/tcp_worker"))} && test -f {shlex.quote(str(remote_root / "torch_benchmark/run_distributed_multiseed.py"))}'
        subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', a.remote, check], check=True)
        a.output.mkdir(parents=True)
        (a.output / 'manifest.json').write_text(json.dumps(vars(a), default=str, indent=2) + '\n')
    index = 0
    for seed in a.seeds:
        for model in a.models:
            original = json.loads((ROOT / f'sample_configs/distributed_v{model}.json').read_text())
            for backend in BACKENDS:
                name = f'{backend}_v{model}_seed{seed}'
                output = a.output / f'{name}.json'
                config = dict(original)
                config.update(host=a.master_addr, local_worker_position=0, dataset_path=str(a.data_root.resolve()),
                              augmentation=False, benchmark_mode=True, gradient_accumulation_steps=1)
                config['workers'][0]['endpoint']['host'] = a.master_addr
                config['workers'][1]['endpoint']['host'] = a.worker_addr
                if a.precision == 'fp32':
                    config.update(io_dtype='FP32', param_dtype='FP32', compute_dtype='FP32')
                cfg_path = a.output / f'{name}_config.json'
                if not a.dry_run:
                    cfg_path.write_text(json.dumps(config, indent=2) + '\n')
                ready_port = None
                if backend == 'tunx':
                    local_cmd = [str(ROOT / 'build/bin/tcp_coordinator'), '--config', str(cfg_path), '--seed', str(seed)]
                    ready_port = config['workers'][1]['endpoint']['port']
                    remote_cmd = [str(remote_root / 'build/bin/tcp_worker'), '--gpu', '0', '--bootstrap-offload',
                                  '--io-threads', str(config['num_io_threads']), '--num-threads', str(config['num_threads']), str(ready_port)]
                else:
                    port = a.master_port + index
                    index += 1
                    common = ['--nnodes=2', '--nproc-per-node=1', f'--master-addr={a.master_addr}',
                              f'--master-port={port}', '--max-restarts=0']
                    training = ['--micro-batch-size', str(a.micro_batch_size), '--precision', a.precision,
                                '--seed', str(seed), '--warmup-steps', '50', '--steps', '2000', '--no-augmentation']
                    if backend == 'deepspeed':
                        training += ['--zero-stage', str(a.zero_stage)]
                    local_cmd = [str(ROOT / '.venv/bin/torchrun'), *common, '--node-rank=0',
                                 str(ROOT / f'torch_benchmark/train_{backend}.py'), '--config', str(cfg_path),
                                 '--data-root', str(a.data_root.resolve()), '--output', str(output), *training]
                    remote_cmd = [str(remote_root / '.venv/bin/torchrun'), *common, '--node-rank=1',
                                  str(remote_root / f'torch_benchmark/train_{backend}.py'),
                                  '--config', str(remote_root / f'sample_configs/distributed_v{model}.json'),
                                  '--data-root', remote_data, '--output', str(remote_output / f'{name}.json'), *training]
                helper = ['python3', str(remote_root / 'torch_benchmark/run_distributed_multiseed.py'),
                          '--remote-job', '--log', str(remote_output / f'{name}.log')]
                if ready_port is not None:
                    helper += ['--ready-port', str(ready_port)]
                ssh = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', '-o', 'ServerAliveInterval=15',
                       '-o', 'ServerAliveCountMax=3', a.remote,
                       'cd ' + shlex.quote(str(remote_root)) + ' && exec ' + shlex.join(helper + ['--', *remote_cmd])]
                print(f'\n{name}\n  local: {shlex.join(local_cmd)}\n  remote: {shlex.join(ssh)}', flush=True)
                if a.dry_run:
                    continue
                local = remote = None
                try:
                    with (a.output / f'{name}.log').open('w') as log, (a.output / f'{name}_ssh.log').open('w') as ssh_log:
                        remote = subprocess.Popen(ssh, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                                  stderr=ssh_log, start_new_session=True)
                        if not select.select([remote.stdout], [], [], 150)[0] or remote.stdout.readline().strip() != b'READY':
                            raise RuntimeError(f'{name}: remote startup failed; see SSH and remote logs')
                        local = subprocess.Popen(local_cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                                 start_new_session=True)
                        deadline = time.monotonic() + a.timeout
                        while local.poll() is None:
                            if remote.poll() not in (None, 0):
                                raise RuntimeError(f'{name}: remote failed ({remote.returncode})')
                            if time.monotonic() > deadline:
                                raise TimeoutError(f'{name}: trial timed out')
                            time.sleep(.5)
                        if local.returncode:
                            raise RuntimeError(f'{name}: local failed ({local.returncode}); see {log.name}')
                        if remote.poll() not in (None, 0):
                            raise RuntimeError(f'{name}: remote failed ({remote.returncode})')
                        if backend != 'tunx':
                            remote.wait(timeout=60)
                            if remote.returncode:
                                raise RuntimeError(f'{name}: remote failed ({remote.returncode})')
                    if backend == 'tunx':
                        text = (a.output / f'{name}.log').read_text()
                        throughput = re.findall(r'Throughput:\s*([\d.]+) samples/s', text)
                        elapsed = re.findall(r'Elapsed time for 2000 steps:\s*([\d.]+) s', text)
                        if len(throughput) != 1 or len(elapsed) != 1 or f'Benchmark seed: {seed}' not in text:
                            raise ValueError(f'{name}: missing or ambiguous benchmark/seed output')
                        output.write_text(json.dumps({'backend': backend, 'model': f'tunx_v{model}', 'seed': seed,
                            'samples_per_second': float(throughput[0]), 'step_ms': float(elapsed[0]) / 2,
                            'measured_steps': 2000, 'warmup_steps': 50, 'source_config': config}, indent=2) + '\n')
                finally:
                    stop(local)
                    if remote is not None:
                        try:
                            remote.stdin.close()  # EOF asks remote helper to clean up its own worker.
                        except BrokenPipeError:
                            pass
                        try:
                            remote.wait(timeout=15)
                        except subprocess.TimeoutExpired:
                            stop(remote)
                        remote.stdout.close()
    if not a.dry_run:
        summarize(a.output)


if __name__ == '__main__':
    if sys.argv[1:2] == ['--remote-job']:
        sys.exit(remote_job(sys.argv[2:]))
    try:
        main()
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError, KeyboardInterrupt) as error:
        print(f'Campaign stopped: {error}', file=sys.stderr)
        sys.exit(1)
