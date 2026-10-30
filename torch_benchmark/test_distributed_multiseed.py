"""CPU-only campaign tests: statistics, completeness, remote cleanup and job matrix."""
import contextlib
import io
import json
import os
from pathlib import Path
import select
import statistics
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import run_distributed_multiseed as suite


class CampaignTests(unittest.TestCase):
    def test_campaign_with_fake_executables(self):
        # Exercise real subprocesses and SSH-helper lifecycle without GPUs or networking across hosts.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            import shutil
            for folder in ('build/bin', '.venv/bin', 'torch_benchmark', 'sample_configs', 'tools'):
                (root / folder).mkdir(parents=True)
            shutil.copy(suite.__file__, root / 'torch_benchmark/run_distributed_multiseed.py')
            config = json.loads((suite.ROOT / 'sample_configs/distributed_v1.json').read_text())
            import socket
            with socket.socket() as sock:
                sock.bind(('127.0.0.1', 0))
                config['workers'][1]['endpoint']['port'] = sock.getsockname()[1]
            (root / 'sample_configs/distributed_v1.json').write_text(json.dumps(config))
            scripts = {
                'tools/ssh': 'import os,sys\nos.execl("/bin/sh", "sh", "-c", sys.argv[-1])\n',
                'build/bin/tcp_coordinator': '''import sys
if '--help' in sys.argv:
    print('--seed')
else:
    seed = sys.argv[sys.argv.index('--seed') + 1]
    print('Benchmark seed: ' + seed)
    print('Throughput: 100.00 samples/s')
    print('Elapsed time for 2000 steps: 2560.000 s')
''',
                'build/bin/tcp_worker': '''import socket,sys,time
s = socket.socket(); s.bind(('0.0.0.0', int(sys.argv[-1]))); s.listen()
time.sleep(120)
''',
                '.venv/bin/torchrun': '''import sys,json,pathlib
if '--node-rank=0' in sys.argv:
    backend = 'deepspeed' if any('train_deepspeed.py' in x for x in sys.argv) else 'fsdp'
    seed = int(sys.argv[sys.argv.index('--seed') + 1])
    output = pathlib.Path(sys.argv[sys.argv.index('--output') + 1])
    output.write_text(json.dumps(dict(backend=backend, seed=seed, model='tunx_v1', samples_per_second=seed*10, step_ms=seed*2)))
''',
            }
            for path, code in scripts.items():
                target = root / path
                target.write_text('#!' + sys.executable + '\n' + code)
                target.chmod(0o755)
            output = root / 'results'
            argv = [suite.__file__, '--models', '1', '--remote-repo', str(root), '--output', str(output)]
            with patch.object(suite, 'ROOT', root), patch.object(sys, 'argv', argv), \
                    patch.dict(os.environ, {'PATH': str(root / 'tools') + ':' + os.environ['PATH']}), \
                    contextlib.redirect_stdout(io.StringIO()):
                suite.main()
            rows = json.loads((output / 'summary.json').read_text())
            self.assertEqual([row['n'] for row in rows], [5, 5, 5])
            self.assertEqual(rows[0]['samples_per_second_std'], 0)
            self.assertEqual(rows[1]['samples_per_second_mean'], 30)

    def test_statistics_and_incomplete_campaign(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'manifest.json').write_text(json.dumps({'models': [1], 'seeds': [1, 2, 3, 4, 5]}))
            for backend in suite.BACKENDS:
                for seed in range(1, 6):
                    (root / f'{backend}_v1_seed{seed}.json').write_text(json.dumps({
                        'model': 'tunx_v1', 'backend': backend, 'seed': seed,
                        'samples_per_second': seed * 10, 'step_ms': seed * 2}))
            with contextlib.redirect_stdout(io.StringIO()):
                suite.summarize(root)
            rows = json.loads((root / 'summary.json').read_text())
            self.assertEqual(len(rows), 3)
            for row in rows:
                self.assertEqual(row['n'], 5)
                self.assertEqual(row['samples_per_second_mean'], 30)
                self.assertAlmostEqual(row['samples_per_second_std'], statistics.stdev([10, 20, 30, 40, 50]))
            (root / 'tunx_v1_seed5.json').unlink()
            with self.assertRaisesRegex(ValueError, 'Incomplete campaign'):
                suite.summarize(root)

    def test_dry_run_matrix(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / 'not_created'
            result = subprocess.run([sys.executable, suite.__file__, '--dry-run', '--output', str(output)],
                                    check=True, capture_output=True, text=True)
            self.assertEqual(result.stdout.count('  local:'), 60)
            self.assertEqual(result.stdout.count('  remote:'), 60)
            self.assertEqual(result.stdout.count('--node-rank=0'), 40)
            self.assertEqual(result.stdout.count('--node-rank=1'), 40)
            self.assertIn('--seed 5', result.stdout)
            self.assertIn('--master-addr=10.10.0.2', result.stdout)
            self.assertIn('10.10.0.1', result.stdout)
            self.assertFalse(output.exists())

    def test_remote_eof_terminates_child(self):
        with tempfile.TemporaryDirectory() as tmp:
            pidfile = Path(tmp) / 'pid'
            code = f'import os,time; open({str(pidfile)!r}, "w").write(str(os.getpid())); time.sleep(120)'
            proc = subprocess.Popen([sys.executable, suite.__file__, '--remote-job', '--log', tmp + '/remote.log',
                                     '--', sys.executable, '-c', code], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
            try:
                self.assertTrue(select.select([proc.stdout], [], [], 5)[0])
                self.assertEqual(proc.stdout.readline().strip(), b'READY')
                # READY means spawned, not necessarily that Python has reached its first statement.
                import time
                deadline = time.monotonic() + 5
                while not pidfile.exists() and time.monotonic() < deadline:
                    time.sleep(.01)
                self.assertTrue(pidfile.exists())
                pid = int(pidfile.read_text())
                proc.stdin.close()
                self.assertEqual(proc.wait(timeout=15), 0)
                with self.assertRaises(ProcessLookupError):
                    os.kill(pid, 0)
            finally:
                if proc.poll() is None:
                    proc.stdin.close()
                    proc.wait(timeout=15)
                proc.stdout.close()


if __name__ == '__main__':
    unittest.main()
