#!/usr/bin/env python3
import os
import subprocess
import argparse
import csv
import numpy as np


def compare_step_tensor(pt_path, tunx_path, step, tensor_name):
    dtype = np.float32 if tensor_name == "inputs" else np.int32
    pt_values = np.fromfile(pt_path, dtype=dtype)
    tunx_values = np.fromfile(tunx_path, dtype=dtype)
    if pt_values.shape != tunx_values.shape:
        raise RuntimeError(
            f"Step {step}: {tensor_name} size mismatch: "
            f"PyTorch={pt_values.size}, TunX={tunx_values.size}")

    if tensor_name == "labels":
        mismatches = np.count_nonzero(pt_values != tunx_values)
        print(f"Step {step}: labels mismatched={mismatches}/{pt_values.size}")
        if mismatches:
            raise RuntimeError(f"Step {step}: labels differ")
        return

    differences = np.abs(pt_values - tunx_values)
    print(
        f"Step {step}: inputs max_abs={differences.max():.3e}, "
        f"mean_abs={differences.mean():.3e}, "
        f"l2_rel={np.linalg.norm(differences) / max(np.linalg.norm(pt_values), 1e-12):.3e}")


def compare_step_inputs(pt_dir, tunx_dir, steps):
    print("--- Per-step input comparison ---")
    for step in range(1, steps + 1):
        compare_step_tensor(
            os.path.join(pt_dir, f"inputs_step_{step}.bin"),
            os.path.join(tunx_dir, f"inputs_step_{step}.bin"), step, "inputs")
        compare_step_tensor(
            os.path.join(pt_dir, f"labels_step_{step}.bin"),
            os.path.join(tunx_dir, f"labels_step_{step}.bin"), step, "labels")


def compare_step_parameters(pt_dir, tunx_dir, steps):
    print("--- Per-step parameter comparison ---")
    for step in range(1, steps + 1):
        pt_step_dir = os.path.join(pt_dir, f"params_step_{step}")
        tunx_step_dir = os.path.join(tunx_dir, f"params_step_{step}")
        results = []
        for filename in sorted(os.listdir(pt_step_dir)):
            pt_path = os.path.join(pt_step_dir, filename)
            tunx_path = os.path.join(tunx_step_dir, filename)
            if not os.path.exists(tunx_path):
                raise RuntimeError(f"Step {step}: missing TunX parameter {filename}")

            pt_values = np.fromfile(pt_path, dtype=np.float32)
            tunx_values = np.fromfile(tunx_path, dtype=np.float32)
            if pt_values.shape != tunx_values.shape:
                raise RuntimeError(
                    f"Step {step}: parameter size mismatch for {filename}: "
                    f"PyTorch={pt_values.size}, TunX={tunx_values.size}")

            difference = tunx_values - pt_values
            pt_norm = np.linalg.norm(pt_values)
            tunx_norm = np.linalg.norm(tunx_values)
            relative_l2 = np.linalg.norm(difference) / max(pt_norm, 1e-12)
            cosine = np.dot(pt_values, tunx_values) / max(pt_norm * tunx_norm, 1e-12)
            jointly_nonzero = (pt_values != 0) & (tunx_values != 0)
            opposite_sign_rate = np.count_nonzero(
                (pt_values[jointly_nonzero] * tunx_values[jointly_nonzero]) < 0
            ) / max(np.count_nonzero(jointly_nonzero), 1)
            kind = "gradients" if filename.endswith(".grad.bin") else "updated parameters"
            results.append((kind, relative_l2, filename, np.abs(difference).max(), cosine,
                            opposite_sign_rate))

        if not results:
            raise RuntimeError(f"Step {step}: no PyTorch parameter snapshots found")
        for kind in ("gradients", "updated parameters"):
            print(f"Step {step}: largest {kind}")
            kind_results = [result for result in results if result[0] == kind]
            for _, relative_l2, filename, max_abs, cosine, opposite_sign_rate in sorted(
                    kind_results, key=lambda result: result[1], reverse=True)[:10]:
                print(
                    f"  {filename} max_abs={max_abs:.3e}, l2_rel={relative_l2:.3e}, "
                    f"cosine={cosine:.8f}, opposite_sign={opposite_sign_rate:.2%}")


def compare_step_metrics(pt_csv, tunx_csv, seed, abs_tolerance, rel_tolerance):
    with open(pt_csv, newline="") as pt_file, open(tunx_csv, newline="") as tunx_file:
        pt_rows = list(csv.DictReader(pt_file))
        tunx_rows = list(csv.DictReader(tunx_file))

    if len(pt_rows) != len(tunx_rows):
        raise RuntimeError(
            f"Seed {seed}: row-count mismatch: PyTorch={len(pt_rows)}, TunX={len(tunx_rows)}")

    print(f"--- Per-step loss comparison (seed {seed}) ---")
    for pt_row, tunx_row in zip(pt_rows, tunx_rows):
        if pt_row["step"] != tunx_row["step"]:
            raise RuntimeError(
                f"Seed {seed}: step mismatch: PyTorch={pt_row['step']}, TunX={tunx_row['step']}")

        pt_loss = float(pt_row["loss"])
        tunx_loss = float(tunx_row["loss"])
        abs_error = abs(pt_loss - tunx_loss)
        rel_error = abs_error / max(abs(pt_loss), 1e-12)
        print(
            f"Step {pt_row['step']}: pt={pt_loss:.8f}, tunx={tunx_loss:.8f}, "
            f"abs={abs_error:.3e}, rel={rel_error:.3e}")
        if abs_error > abs_tolerance and rel_error > rel_tolerance:
            raise RuntimeError(
                f"Seed {seed}: loss mismatch at step {pt_row['step']}: "
                f"abs={abs_error:.3e} > {abs_tolerance:.3e}, "
                f"rel={rel_error:.3e} > {rel_tolerance:.3e}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="resnet50")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--debug-compare", action="store_true",
                        help="Compare PyTorch and TunX loss CSVs after each seed")
    parser.add_argument("--loss-abs-tolerance", type=float, default=1e-3)
    parser.add_argument("--loss-rel-tolerance", type=float, default=1e-3)
    args = parser.parse_args()

    seeds = [42, 420, 4200]
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(base_dir, ".."))
    
    # Ensure dataset paths are absolute and passed to subprocesses
    env = os.environ.copy()
    if "IMAGENET100_ROOT" not in env:
        env["IMAGENET100_ROOT"] = os.path.join(project_root, "data", "imagenet-100")
    if "OPENWEBTEXT_PATH" not in env:
        env["OPENWEBTEXT_PATH"] = os.path.join(project_root, "data", "open-web-text", "train.bin")
    
    for seed in seeds:
        print(f"==================================================")
        print(f"Running convergence test for {args.model} with seed {seed}")
        print(f"==================================================")
        
        dump_dir = os.path.join(project_root, "dump", f"trajectory_{args.model}_{seed}")
        tunx_dir = os.path.join(project_root, "dump", f"tunx_trajectory_{args.model}_{seed}")
        
        # 1. Run PyTorch generator
        print(">>> Running PyTorch generator...")
        cmd_pt = [
            "uv", "run", "python", os.path.join(base_dir, "dump_trajectory.py"),
            "--model", args.model,
            "--batch-size", str(args.batch_size),
            "--steps", str(args.steps),
            "--seed", str(seed),
            "--dump-dir", dump_dir,
            "--no-aug",
        ]
        if args.debug_compare:
            cmd_pt.extend(["--dump-inputs", "--dump-params"])
        subprocess.run(cmd_pt, check=True, env=env)
        
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
            "--no-aug",
        ]
        if args.debug_compare:
            cmd_tunx.extend(["--dump-inputs", "--load-inputs", "--dump-params"])
        subprocess.run(cmd_tunx, check=True, env=env)

        if args.debug_compare:
            compare_step_inputs(dump_dir, tunx_dir, args.steps)
            compare_step_parameters(dump_dir, tunx_dir, args.steps)
            compare_step_metrics(
                os.path.join(dump_dir, f"pt_trajectory_seed_{seed}.csv"),
                os.path.join(tunx_dir, f"tunx_trajectory_seed_{seed}.csv"),
                seed,
                args.loss_abs_tolerance,
                args.loss_rel_tolerance,
            )

    print("All convergence tests completed.")

if __name__ == "__main__":
    main()
