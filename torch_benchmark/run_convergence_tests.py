#!/usr/bin/env python3
import os
import subprocess
import argparse
import csv
import numpy as np

from compare_equivalence import compare_tensors, print_table, load_tensor_bin, load_topology_order

def compare_step_tensor(pt_path, tunx_path, step, tensor_name):
    if tensor_name == "inputs":
        dtype = np.float64 if os.environ.get("TUNX_COMPARE_FP64") == "1" else np.float32
    else:
        dtype = np.int32
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


def compare_step_inputs(pt_dir, tunx_dir, steps, *, start_step=1):
    print("--- Per-step input comparison ---")
    for step in range(start_step, steps + 1):
        compare_step_tensor(
            os.path.join(pt_dir, f"inputs_step_{step}.bin"),
            os.path.join(tunx_dir, f"inputs_step_{step}.bin"), step, "inputs")
        compare_step_tensor(
            os.path.join(pt_dir, f"labels_step_{step}.bin"),
            os.path.join(tunx_dir, f"labels_step_{step}.bin"), step, "labels")


def compare_step_parameters(pt_dir, tunx_dir, steps, *, start_step=1):
    print("--- Per-step parameter comparison ---")
    for step in range(start_step, steps + 1):
        pt_step_dir = os.path.join(pt_dir, f"params_step_{step}")
        tunx_step_dir = os.path.join(tunx_dir, f"params_step_{step}")
        if not os.path.exists(pt_step_dir):
            continue
        
        results_grad = []
        results_upd = []
        
        files_to_compare = sorted(os.listdir(pt_step_dir))
        for filename in files_to_compare:
            pt_path = os.path.join(pt_step_dir, filename)
            tunx_path = os.path.join(tunx_step_dir, filename)
            if not os.path.exists(tunx_path):
                raise RuntimeError(f"Step {step}: missing TunX parameter {filename}")

            pt_values = load_tensor_bin(pt_path)
            tunx_values = load_tensor_bin(tunx_path)
            
            res = compare_tensors(pt_values, tunx_values, filename)
            if res:
                if filename.endswith(".grad.bin"):
                    results_grad.append(res)
                else:
                    results_upd.append(res)
                    
        print(f"\n=================== Step {step} Parameters ===================")
        if results_grad:
            print_table(f"Gradients (Step {step})", results_grad)
        if results_upd:
            print_table(f"Updated Parameters (Step {step})", results_upd)


def compare_shared_loss_gradients(pt_dir, tunx_dir, steps, *, start_step=1):
    print("--- Shared loss-gradient verification ---")
    for step in range(start_step, steps + 1):
        filename = f"loss_gradient_step_{step}.bin"
        pt = np.fromfile(os.path.join(pt_dir, filename), dtype=np.uint8)
        tunx = np.fromfile(os.path.join(tunx_dir, filename), dtype=np.uint8)
        if not np.array_equal(pt, tunx):
            raise RuntimeError(f"Step {step}: shared loss-gradient bytes differ")
        print(f"Step {step}: dLoss/dOutputs is bit-for-bit identical")


def compare_step_debug_layer(pt_dir, tunx_dir, steps, debug_layer, *, start_step=1):
    if not debug_layer:
        return
    print(f"--- Per-step debug layer '{debug_layer}' comparison ---")
    for step in range(start_step, steps + 1):
        results_act = []
        results_grad = []
        for suffix, is_grad in [
            (f".output_step_{step}.bin", False), 
            (f".grad_input_step_{step}.bin", True),
            (f".grad_output_step_{step}.bin", True)
        ]:
            filename = f"{debug_layer}{suffix}"
            pt_path = os.path.join(pt_dir, filename)
            tunx_path = os.path.join(tunx_dir, filename)
            if os.path.exists(pt_path) and os.path.exists(tunx_path):
                res = compare_tensors(load_tensor_bin(pt_path), load_tensor_bin(tunx_path), filename)
                if res:
                    if is_grad:
                        results_grad.append(res)
                    else:
                        results_act.append(res)
        
        if results_act or results_grad:
            print(f"\n=================== Step {step} Debug Layer '{debug_layer}' ===================")
            if results_act: print_table("Intermediate Activations", results_act)
            if results_grad: print_table("Intermediate Gradients", results_grad)


def compare_step_activations(pt_dir, tunx_dir, steps, *, start_step=1):
    """Compare all-layer intermediate activations and gradients dumped with --dump-activations."""
    import glob
    print("--- Per-step intermediate activations/gradients comparison ---")
    for step in range(start_step, steps + 1):
        results_act = []
        results_grad = []
        pt_outputs = glob.glob(os.path.join(pt_dir, f"*.output_step_{step}.bin"))
        layer_names = sorted(
            set(os.path.basename(p).replace(f".output_step_{step}.bin", "") for p in pt_outputs)
        )
        for layer_name in layer_names:
            for suffix, is_grad in [
                (f".output_step_{step}.bin", False),
                (f".grad_input_step_{step}.bin", True),
                (f".grad_output_step_{step}.bin", True),
            ]:
                filename = f"{layer_name}{suffix}"
                pt_path = os.path.join(pt_dir, filename)
                tunx_path = os.path.join(tunx_dir, filename)
                if os.path.exists(pt_path) and os.path.exists(tunx_path):
                    res = compare_tensors(
                        load_tensor_bin(pt_path), load_tensor_bin(tunx_path), filename
                    )
                    if res:
                        if is_grad:
                            results_grad.append(res)
                        else:
                            results_act.append(res)
        if results_act or results_grad:
            print(f"\n=================== Step {step} Intermediate Tensors ===================")
            if results_act:
                print_table(f"Intermediate Activations (Step {step})", results_act)
            if results_grad:
                print_table(f"Intermediate Gradients (Step {step})", results_grad)


def compare_step_metrics(pt_csv, tunx_csv, seed, abs_tolerance, rel_tolerance, *, step=None):

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

        if step is not None and int(pt_row["step"]) != step:
            continue

        pt_loss = float(pt_row["loss"])
        tunx_loss = float(tunx_row["loss"])
        abs_error = abs(pt_loss - tunx_loss)
        rel_error = abs_error / max(abs(pt_loss), 1e-12)
        print(
            f"Step {pt_row['step']}: pt={pt_loss:.8f}, tunx={tunx_loss:.8f}, "
            f"abs={abs_error:.3e}, rel={rel_error:.3e}")
        if abs_error > abs_tolerance and rel_error > rel_tolerance:
            print(
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
    parser.add_argument("--shared-loss-gradient", action="store_true",
                        help="Feed PyTorch's dLoss/dOutputs into TunX backward; use --steps 1 "
                             "to isolate backward before parameter trajectories diverge")
    parser.add_argument("--loss-abs-tolerance", type=float, default=1e-3)
    parser.add_argument("--loss-rel-tolerance", type=float, default=1e-3)
    parser.add_argument("--debug-layer", type=str, default=None,
                        help="Dump backward input/output gradients for one mapped layer")
    parser.add_argument("--engine", type=str, default="default",
                        help="Engine to run the tunx convergence test (e.g., cuda, cudnn)")
    parser.add_argument("--fp64", action="store_true",
                        help="Run model and tensors in double precision (FP64)")
    args = parser.parse_args()

    seeds = [42]
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(base_dir, ".."))
    
    # Ensure dataset paths are absolute and passed to subprocesses
    env = os.environ.copy()
    if "IMAGENET100_ROOT" not in env:
        env["IMAGENET100_ROOT"] = os.path.join(project_root, "data", "imagenet-100")
    if "OPENWEBTEXT_PATH" not in env:
        env["OPENWEBTEXT_PATH"] = os.path.join(project_root, "data", "open-web-text", "train.bin")
    env["NVIDIA_TF32_OVERRIDE"] = "0"
    if args.fp64:
        env["TUNX_COMPARE_FP64"] = "1"
        os.environ["TUNX_COMPARE_FP64"] = "1"
    
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
            "--dump-inputs",
        ]
        if args.debug_compare:
            cmd_pt.extend(["--dump-params"])
        if args.shared_loss_gradient:
            cmd_pt.append("--dump-loss-gradients")
        if args.debug_layer:
            cmd_pt.extend(["--debug-layer", args.debug_layer])
        if args.debug_compare:
            cmd_pt.append("--dump-activations")
        if args.fp64:
            cmd_pt.append("--fp64")
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
            "--engine", args.engine,
            "--load-inputs",
        ]
        if args.debug_compare:
            cmd_tunx.extend(["--dump-inputs", "--dump-params"])
        if args.shared_loss_gradient:
            cmd_tunx.append("--load-loss-gradients")
        if args.debug_layer:
            cmd_tunx.extend(["--debug-layer", args.debug_layer])
        if args.debug_compare:
            cmd_tunx.append("--dump-activations")
        if args.engine != "default":
            cmd_tunx.extend(["--engine", args.engine])
        if args.fp64:
            cmd_tunx.append("--fp64")
        subprocess.run(cmd_tunx, check=True, env=env)

        if args.debug_compare:
            load_topology_order(dump_dir)

        if args.debug_compare or args.shared_loss_gradient:
            for step in range(1, args.steps + 1):
                print(f"\n=================== Step {step} ===================")
                if args.shared_loss_gradient:
                    compare_shared_loss_gradients(dump_dir, tunx_dir, step, start_step=step)
                if args.debug_compare:
                    compare_step_inputs(dump_dir, tunx_dir, step, start_step=step)
                    if args.debug_layer:
                        compare_step_debug_layer(
                            dump_dir, tunx_dir, step, args.debug_layer, start_step=step)
                    compare_step_activations(dump_dir, tunx_dir, step, start_step=step)
                    compare_step_parameters(dump_dir, tunx_dir, step, start_step=step)
                    compare_step_metrics(
                        os.path.join(dump_dir, f"pt_trajectory_seed_{seed}.csv"),
                        os.path.join(tunx_dir, f"tunx_trajectory_seed_{seed}.csv"),
                        seed,
                        args.loss_abs_tolerance,
                        args.loss_rel_tolerance,
                        step=step,
                    )

    print("All convergence tests completed.")

if __name__ == "__main__":
    main()
