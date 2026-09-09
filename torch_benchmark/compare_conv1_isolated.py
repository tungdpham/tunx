#!/usr/bin/env python3
"""Compares the isolated conv1 outputs from test_conv1_isolated.py (PyTorch) and
test_conv1_isolated.cpp (TunX). Reports normalized L2 error and the full error
distribution, not just the max, since max-only metrics are dominated by outliers near
zero-valued elements.
"""
import argparse
import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pt_dir", type=str, required=True)
    parser.add_argument("--tunx_dir", type=str, required=True)
    parser.add_argument("--plot_path", type=str, default="conv1_isolated_error_dist.png")
    args = parser.parse_args()

    pt = np.fromfile(os.path.join(args.pt_dir, "conv1_isolated.act.bin"), dtype=np.float32)
    tunx = np.fromfile(os.path.join(args.tunx_dir, "conv1_isolated.act.bin"), dtype=np.float32)

    if pt.shape != tunx.shape:
        print(f"Shape mismatch: pt={pt.shape} tunx={tunx.shape}")
        return

    diff = tunx - pt
    abs_diff = np.abs(diff)

    pt_l2 = np.linalg.norm(pt)
    tunx_l2 = np.linalg.norm(tunx)
    diff_l2 = np.linalg.norm(diff)
    normalized_l2 = diff_l2 / pt_l2 if pt_l2 > 0 else float("nan")

    cos_sim = np.dot(pt, tunx) / (pt_l2 * tunx_l2) if pt_l2 > 0 and tunx_l2 > 0 else float("nan")

    percentiles = [50, 90, 99, 99.9, 100]
    abs_pcts = np.percentile(abs_diff, percentiles)

    print(f"pt   output L2 norm: {pt_l2:.6g}")
    print(f"tunx output L2 norm: {tunx_l2:.6g}")
    print(f"||tunx - pt||_2 / ||pt||_2 (normalized L2 error): {normalized_l2:.6g}")
    print(f"cosine similarity: {cos_sim:.8f}")
    print("abs error distribution (percentiles):")
    for p, v in zip(percentiles, abs_pcts):
        print(f"  p{p}: {v:.6g}")
    print(f"mean abs error: {abs_diff.mean():.6g}, std: {abs_diff.std():.6g}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].hist(abs_diff, bins=100, color="steelblue")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("abs error")
    axes[0].set_ylabel("count (log scale)")
    axes[0].set_title("conv1 isolated: abs error histogram")

    sorted_abs = np.sort(abs_diff)
    cdf = np.arange(1, len(sorted_abs) + 1) / len(sorted_abs)
    axes[1].plot(sorted_abs, cdf)
    axes[1].set_xlabel("abs error")
    axes[1].set_ylabel("cumulative fraction of elements")
    axes[1].set_title("conv1 isolated: abs error CDF")
    axes[1].axvline(1e-4, color="red", linestyle="--", label="atol=1e-4")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(args.plot_path, dpi=150)
    print(f"Saved distribution plot to {args.plot_path}")


if __name__ == "__main__":
    main()
