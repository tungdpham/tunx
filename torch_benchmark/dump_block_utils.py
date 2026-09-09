#!/usr/bin/env python3
"""Dumps isolated-block PyTorch reference tensors (inputs, params, grads, updated params)
that mirror TunX's graph_builder helpers (bottleneck_residual_block, inception_block,
attention, gpt_block) exactly, so build/bin/test_block_equivalence can be run against
the same weights/upstream-gradient and compare_equivalence.py can diff the two dumps.

A random grad_output is generated and dumped alongside the inputs, and fed directly to
backward() on both sides -- this removes the need for a classifier head/loss when testing
a block in isolation.
"""
import os
import math
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from dump_utils import save_tensor_bin

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.set_float32_matmul_precision("highest")


class BottleneckBlock(nn.Module):
    """Matches TunX's bottleneck_residual_block(mid, out, stride)."""

    def __init__(self, in_channels, mid_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, mid_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels, eps=1e-5, momentum=0.1)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, 3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channels, eps=1e-5, momentum=0.1)
        self.conv3 = nn.Conv2d(mid_channels, out_channels, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels, eps=1e-5, momentum=0.1)
        self.shortcut = None
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels, eps=1e-5, momentum=0.1),
            )

    def forward(self, x):
        sc = self.shortcut(x) if self.shortcut is not None else x
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = F.relu(self.bn3(self.conv3(out)))
        return F.relu(out + sc)

    def named_tensor_params(self):
        yield "block_conv1", "weight", self.conv1.weight
        yield "block_bn0", "weight", self.bn1.weight
        yield "block_bn0", "bias", self.bn1.bias
        yield "block_conv2", "weight", self.conv2.weight
        yield "block_bn1", "weight", self.bn2.weight
        yield "block_bn1", "bias", self.bn2.bias
        yield "block_conv3", "weight", self.conv3.weight
        yield "block_bn2", "weight", self.bn3.weight
        yield "block_bn2", "bias", self.bn3.bias
        if self.shortcut is not None:
            yield "block_conv0", "weight", self.shortcut[0].weight
            yield "block_bn3", "weight", self.shortcut[1].weight
            yield "block_bn3", "bias", self.shortcut[1].bias


class InceptionBlock(nn.Module):
    """Matches TunX's inception_block(out_channels): 4 branches concatenated on channel dim."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.b1_conv = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.b1_bn = nn.BatchNorm2d(out_channels, eps=1e-5, momentum=0.1)

        self.b2_conv1 = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.b2_bn1 = nn.BatchNorm2d(out_channels, eps=1e-5, momentum=0.1)
        self.b2_conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.b2_bn2 = nn.BatchNorm2d(out_channels, eps=1e-5, momentum=0.1)

        self.b3_conv1 = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.b3_bn1 = nn.BatchNorm2d(out_channels, eps=1e-5, momentum=0.1)
        self.b3_conv2 = nn.Conv2d(out_channels, out_channels, 5, padding=2, bias=False)
        self.b3_bn2 = nn.BatchNorm2d(out_channels, eps=1e-5, momentum=0.1)

        self.b4_pool = nn.MaxPool2d(3, stride=1, padding=1)
        self.b4_conv = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.b4_bn = nn.BatchNorm2d(out_channels, eps=1e-5, momentum=0.1)

    def forward(self, x):
        b1 = F.relu(self.b1_bn(self.b1_conv(x)))
        b2 = F.relu(self.b2_bn1(self.b2_conv1(x)))
        b2 = F.relu(self.b2_bn2(self.b2_conv2(b2)))
        b3 = F.relu(self.b3_bn1(self.b3_conv1(x)))
        b3 = F.relu(self.b3_bn2(self.b3_conv2(b3)))
        b4 = F.relu(self.b4_bn(self.b4_conv(self.b4_pool(x))))
        out = torch.cat([b1, b2, b3, b4], dim=1)
        return F.relu(out)

    def named_tensor_params(self):
        yield "block_b1_conv", "weight", self.b1_conv.weight
        yield "block_b1_bn", "weight", self.b1_bn.weight
        yield "block_b1_bn", "bias", self.b1_bn.bias
        yield "block_b2_conv1", "weight", self.b2_conv1.weight
        yield "block_b2_bn1", "weight", self.b2_bn1.weight
        yield "block_b2_bn1", "bias", self.b2_bn1.bias
        yield "block_b2_conv2", "weight", self.b2_conv2.weight
        yield "block_b2_bn2", "weight", self.b2_bn2.weight
        yield "block_b2_bn2", "bias", self.b2_bn2.bias
        yield "block_b3_conv1", "weight", self.b3_conv1.weight
        yield "block_b3_bn1", "weight", self.b3_bn1.weight
        yield "block_b3_bn1", "bias", self.b3_bn1.bias
        yield "block_b3_conv2", "weight", self.b3_conv2.weight
        yield "block_b3_bn2", "weight", self.b3_bn2.weight
        yield "block_b3_bn2", "bias", self.b3_bn2.bias
        yield "block_b4_conv", "weight", self.b4_conv.weight
        yield "block_b4_bn", "weight", self.b4_bn.weight
        yield "block_b4_bn", "bias", self.b4_bn.bias


class AttentionBlock(nn.Module):
    """Matches TunX's FlashAttentionBlock: separate (not fused) q/k/v/out linear projections."""

    def __init__(self, embed_dim, num_heads, is_causal=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.is_causal = is_causal
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)

    def forward(self, x):
        b, s, e = x.shape
        q = self.q_proj(x).view(b, s, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, s, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, s, self.num_heads, self.head_dim).transpose(1, 2)
        scale = 1.0 / math.sqrt(self.head_dim)
        att = (q @ k.transpose(-2, -1)) * scale
        if self.is_causal:
            mask = torch.triu(torch.ones(s, s, device=x.device, dtype=torch.bool), diagonal=1)
            att = att.masked_fill(mask, float("-inf"))
        att = F.softmax(att, dim=-1)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(b, s, e)
        return self.out_proj(y)

    def named_tensor_params(self, prefix="block"):
        yield prefix, "q_proj.weight", self.q_proj.weight
        yield prefix, "q_proj.bias", self.q_proj.bias
        yield prefix, "k_proj.weight", self.k_proj.weight
        yield prefix, "k_proj.bias", self.k_proj.bias
        yield prefix, "v_proj.weight", self.v_proj.weight
        yield prefix, "v_proj.bias", self.v_proj.bias
        yield prefix, "out_proj.weight", self.out_proj.weight
        yield prefix, "out_proj.bias", self.out_proj.bias


class GPT2Block(nn.Module):
    """Matches TunX's gpt_block: LN->attn->residual, LN->MLP(GELU-tanh)->residual."""

    def __init__(self, embed_dim, num_heads, ffn_dim, is_causal=True):
        super().__init__()
        self.ln_1 = nn.LayerNorm(embed_dim, eps=1e-5)
        self.attn = AttentionBlock(embed_dim, num_heads, is_causal)
        self.ln_2 = nn.LayerNorm(embed_dim, eps=1e-5)
        self.mlp_fc1 = nn.Linear(embed_dim, ffn_dim, bias=True)
        self.mlp_fc2 = nn.Linear(ffn_dim, embed_dim, bias=True)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        # TunX's GELU kernel uses the tanh approximation, not the exact erf-based one.
        ffn = self.mlp_fc2(F.gelu(self.mlp_fc1(self.ln_2(x)), approximate="tanh"))
        return x + ffn

    def named_tensor_params(self):
        yield "block_ln_1", "weight", self.ln_1.weight
        yield "block_ln_1", "bias", self.ln_1.bias
        for _, suffix, t in self.attn.named_tensor_params(prefix="block_attn"):
            yield "block_attn", suffix, t
        yield "block_ln_2", "weight", self.ln_2.weight
        yield "block_ln_2", "bias", self.ln_2.bias
        yield "block_mlp_fc1", "weight", self.mlp_fc1.weight
        yield "block_mlp_fc1", "bias", self.mlp_fc1.bias
        yield "block_mlp_fc2", "weight", self.mlp_fc2.weight
        yield "block_mlp_fc2", "bias", self.mlp_fc2.bias


def get_block(name):
    if name == "residual":
        return BottleneckBlock(64, 64, 256, stride=1), (2, 56, 56, 64)
    if name == "inception":
        return InceptionBlock(64, 32), (2, 28, 28, 64)
    if name == "attention":
        return AttentionBlock(64, 4, is_causal=True), (2, 16, 64)
    if name == "gpt2":
        return GPT2Block(64, 4, 256, is_causal=True), (2, 16, 64)
    raise ValueError(f"Unknown block: {name}")


def make_input(shape, device, dtype=torch.float32):
    if len(shape) == 4:
        n, h, w, c = shape
        return torch.randn(n, c, h, w, dtype=dtype, device=device)
    return torch.randn(*shape, dtype=dtype, device=device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--block", type=str, required=True,
                        choices=["residual", "inception", "attention", "gpt2"])
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--dump-dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.dump_dir, exist_ok=True)
    torch.manual_seed(args.seed)

    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    model, shape = get_block(args.block)
    shape = (args.batch_size,) + shape[1:]
    model = model.to(args.device)
    model.train()

    # cuDNN's SDPA (used by TunX's attention/gpt2 blocks) only supports half/bf16, so run
    # these two blocks' reference computation in bfloat16 too for a fair comparison.
    needs_bf16 = args.block in ("attention", "gpt2")
    param_dtype = torch.bfloat16 if needs_bf16 else torch.float32
    if needs_bf16:
        model = model.to(param_dtype)

    print(f"Dumping initial parameters for block '{args.block}'...")
    for layer_name, suffix, tensor in model.named_tensor_params():
        save_tensor_bin(tensor, os.path.join(args.dump_dir, f"{layer_name}.{suffix}.bin"))

    inputs = make_input(shape, args.device, dtype=param_dtype)
    print("Dumping inputs...")
    save_tensor_bin(inputs, os.path.join(args.dump_dir, "inputs.bin"))

    optimizer = optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=3e-4)
    optimizer.zero_grad()

    print("Running forward pass...")
    inputs.requires_grad_(True)
    output = model(inputs)
    save_tensor_bin(output, os.path.join(args.dump_dir, "outputs.bin"))

    # Random upstream gradient shared by both frameworks, so backward doesn't need a
    # classifier head/loss to produce a gradient signal.
    grad_output = torch.randn_like(output)
    save_tensor_bin(grad_output, os.path.join(args.dump_dir, "grad_output.bin"))

    print("Running backward pass...")
    output.backward(grad_output)

    print("Dumping gradients...")
    for layer_name, suffix, tensor in model.named_tensor_params():
        save_tensor_bin(tensor.grad, os.path.join(args.dump_dir, f"{layer_name}.{suffix}.grad.bin"))

    print("Running optimizer step...")
    optimizer.step()

    print("Dumping updated parameters...")
    for layer_name, suffix, tensor in model.named_tensor_params():
        save_tensor_bin(tensor, os.path.join(args.dump_dir, f"{layer_name}.{suffix}.updated.bin"))

    print(f"Dump complete for block '{args.block}' in {args.dump_dir}")


if __name__ == "__main__":
    main()
