#!/usr/bin/env python3
"""Isolated conv1-only forward pass on the PyTorch side.

Loads the exact conv1.weight.bin and inputs.bin dumped by dump_utils.py (already stored
NHWC by that script) and runs a single F.conv2d with no bias, stride 2, pad 3 - matching
examples/test_conv1_isolated.cpp bit-for-bit in problem shape. This isolates the stem conv
from every downstream op so any divergence there can only come from the conv itself
(cuDNN plan/algorithm, reduction order), not from accumulated depth.
"""
import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pt_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"cudnn version: {torch.backends.cudnn.version()}, deterministic: "
          f"{torch.backends.cudnn.deterministic}, benchmark: {torch.backends.cudnn.benchmark}, "
          f"allow_tf32: {torch.backends.cudnn.allow_tf32}")

    weight_nhwc = np.fromfile(os.path.join(args.pt_dir, "conv1.weight.bin"), dtype=np.float32)
    weight_nhwc = weight_nhwc.reshape(64, 7, 7, 3)
    weight = torch.from_numpy(weight_nhwc).permute(0, 3, 1, 2).contiguous().to(args.device)

    inputs_nhwc = np.fromfile(os.path.join(args.pt_dir, "inputs.bin"), dtype=np.float32)
    inputs_nhwc = inputs_nhwc.reshape(args.batch_size, 224, 224, 3)
    inputs = torch.from_numpy(inputs_nhwc).permute(0, 3, 1, 2).contiguous().to(args.device)

    with torch.no_grad():
        output = F.conv2d(inputs, weight, bias=None, stride=2, padding=3)

    l2_norm = torch.linalg.norm(output).item()
    print(f"conv1 (isolated) output L2 norm: {l2_norm}")

    output_nhwc = output.permute(0, 2, 3, 1).contiguous().cpu().numpy()
    output_nhwc.tofile(os.path.join(args.pt_dir, "conv1_isolated.act.bin"))
    print(f"Dump complete: {os.path.join(args.pt_dir, 'conv1_isolated.act.bin')}")


if __name__ == "__main__":
    main()
