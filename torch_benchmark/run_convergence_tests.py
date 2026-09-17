#!/usr/bin/env python3
import os
import subprocess
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="resnet50")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--steps", type=int, default=1000)
    args = parser.parse_args()

    seeds = [42, 420, 4200]
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(base_dir, ".."))
    
    for seed in seeds:
        print(f"==================================================")
        print(f"Running convergence test for {args.model} with seed {seed}")
        print(f"==================================================")
        
        dump_dir = os.path.join(project_root, "dump", f"trajectory_{args.model}_{seed}")
        tunx_dir = os.path.join(project_root, "dump", f"tunx_trajectory_{args.model}_{seed}")
        
        # 1. Run PyTorch generator
        print(">>> Running PyTorch generator...")
        cmd_pt = [
            "python", os.path.join(base_dir, "dump_trajectory.py"),
            "--model", args.model,
            "--batch-size", str(args.batch_size),
            "--steps", str(args.steps),
            "--seed", str(seed),
            "--dump-dir", dump_dir,
            "--no-aug"
        ]
        subprocess.run(cmd_pt, check=True)
        
        # 2. Run TunX convergence test
        print(">>> Running TunX convergence runner...")
        cmd_tunx = [
            os.path.join(project_root, "build", "bin", "test_convergence"),
            "--model", args.model,
            "--batch-size", str(args.batch_size),
            "--steps", str(args.steps),
            "--seed", str(seed),
            "--pt-dir", dump_dir,
            "--tunx-dir", tunx_dir,
            "--no-aug"
        ]
        subprocess.run(cmd_tunx, check=True)

    print("All convergence tests completed.")

if __name__ == "__main__":
    main()
