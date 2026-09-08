#!/usr/bin/env python3
import os
import argparse
import numpy as np
from pathlib import Path
from tabulate import tabulate

def load_tensor_bin(path, dtype=np.float32):
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        data = np.frombuffer(f.read(), dtype=dtype)
    return data

def compare_tensors(t1, t2, name, rtol=1e-4, atol=1e-4):
    if t1 is None or t2 is None:
        return None
        
    if t1.shape != t2.shape:
        print(f"Shape mismatch for {name}: {t1.shape} vs {t2.shape}")
        # Try to flatten
        t1 = t1.flatten()
        t2 = t2.flatten()
        
        if t1.shape != t2.shape:
            return None
            
    abs_diff = np.abs(t1 - t2)
    max_abs = np.max(abs_diff)
    
    # Avoid division by zero
    t2_safe = np.where(np.abs(t2) < 1e-12, 1e-12, t2)
    rel_diff = np.abs(t1 - t2) / np.abs(t2_safe)
    max_rel = np.max(rel_diff)
    
    # Cosine similarity
    norm1 = np.linalg.norm(t1)
    norm2 = np.linalg.norm(t2)
    if norm1 < 1e-12 or norm2 < 1e-12:
        cos_sim = 1.0 if max_abs < 1e-6 else 0.0
    else:
        cos_sim = np.dot(t1.flatten(), t2.flatten()) / (norm1 * norm2)
        
    is_allclose = np.allclose(t1, t2, rtol=rtol, atol=atol)
    
    return {
        'name': name,
        'max_abs': max_abs,
        'max_rel': max_rel,
        'cos_sim': cos_sim,
        'allclose': is_allclose
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pt_dir", type=str, required=True)
    parser.add_argument("--tunx_dir", type=str, required=True)
    args = parser.parse_args()

    results = []
    
    # 1. Compare outputs
    pt_out = load_tensor_bin(os.path.join(args.pt_dir, "outputs.bin"))
    tunx_out = load_tensor_bin(os.path.join(args.tunx_dir, "outputs.bin"))
    
    res = compare_tensors(pt_out, tunx_out, "Forward Outputs")
    if res:
        results.append(res)
    else:
        print("Missing outputs.bin in one of the directories.")
        
    # 2. Compare gradients and updated parameters
    # Find all parameter names from the dump_pt directory
    pt_files = os.listdir(args.pt_dir)
    param_names = set()
    for f in pt_files:
        if f.endswith(".weight.bin") or f.endswith(".bias.bin"):
            # strip .bin
            param_names.add(f[:-4])
            
    for name in sorted(param_names):
        # Compare gradients
        pt_grad = load_tensor_bin(os.path.join(args.pt_dir, f"{name}.grad.bin"))
        tunx_grad = load_tensor_bin(os.path.join(args.tunx_dir, f"{name}.grad.bin"))
        if pt_grad is not None and tunx_grad is not None:
            res = compare_tensors(pt_grad, tunx_grad, f"{name}.grad")
            if res: results.append(res)
            
        # Compare updated params
        pt_upd = load_tensor_bin(os.path.join(args.pt_dir, f"{name}.updated.bin"))
        tunx_upd = load_tensor_bin(os.path.join(args.tunx_dir, f"{name}.updated.bin"))
        if pt_upd is not None and tunx_upd is not None:
            res = compare_tensors(pt_upd, tunx_upd, f"{name}.updated")
            if res: results.append(res)

    # Summarize overall metrics (like in the paper)
    if not results:
        print("No comparison results found.")
        return

    # Table 4: Numerical Equivalence
    # Max Absolute Error, Max Relative Error, Min Cosine Similarity, allclose Passes
    
    # Calculate aggregates across all tensors
    overall_max_abs = max(r['max_abs'] for r in results)
    overall_max_rel = max(r['max_rel'] for r in results)
    overall_min_cos = min(r['cos_sim'] for r in results)
    
    total_tensors = len(results)
    allclose_passes = sum(1 for r in results if r['allclose'])
    allclose_pct = (allclose_passes / total_tensors) * 100
    
    print("\n--- Detailed Tensor Comparison ---")
    table_data = [[r['name'], f"{r['max_abs']:.2e}", f"{r['max_rel']:.2e}", f"{r['cos_sim']:.6f}", r['allclose']] for r in results]
    print(tabulate(table_data, headers=["Tensor", "Max Abs Err", "Max Rel Err", "Cosine Sim", "Allclose (1e-4)"]))
    
    print("\n--- Aggregate Metrics (RQ4) ---")
    agg_data = [
        ["Max absolute error", f"{overall_max_abs:.2e}"],
        ["Max relative error", f"{overall_max_rel:.2e}"],
        ["Min cosine similarity", f"{overall_min_cos:.6f}"],
        ["allclose passes (%)", f"{allclose_pct:.2f}%"]
    ]
    print(tabulate(agg_data, headers=["Metric", "Value"]))

if __name__ == "__main__":
    main()
