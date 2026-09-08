#!/usr/bin/env python3
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import json
from pathlib import Path

# Import the models from the existing torch_trainer
from torch_trainer import TunxV1, TunxV2, TunxV3, TunxV4, ResNet50ImageNet100, GPT2Small

def get_model(model_name):
    if model_name == "tunx_v1":
        return TunxV1(num_classes=100)
    elif model_name == "tunx_v2":
        return TunxV2(num_classes=100)
    elif model_name == "tunx_v3":
        return TunxV3(num_classes=100)
    elif model_name == "tunx_v4":
        return TunxV4(num_classes=100)
    elif model_name == "resnet50":
        return ResNet50ImageNet100(num_classes=100)
    elif model_name == "gpt2_small":
        return GPT2Small(vocab_size=50257, seq_len=1024)
    else:
        raise ValueError(f"Unknown model: {model_name}")

def get_input_shape(model_name, batch_size):
    if "tunx" in model_name or "resnet50" in model_name:
        return (batch_size, 3, 224, 224)
    elif "gpt2" in model_name:
        return (batch_size, 1024)
    else:
        raise ValueError(f"Unknown model: {model_name}")

def save_tensor_bin(tensor, path, name=""):
    """Saves a PyTorch tensor to a raw binary file."""
    # TunX uses NHWC format for 4D tensors, PyTorch uses NCHW
    if len(tensor.shape) == 4:
        # For weights: (out, in, H, W) -> (out, H, W, in)
        # For activations: (N, C, H, W) -> (N, H, W, C)
        tensor = tensor.permute(0, 2, 3, 1)
        
    # Transpose fc.weight for flattened NHWC inputs
    if name.endswith("fc.weight") and len(tensor.shape) == 2:
        out_features, in_features = tensor.shape
        # tunx_v1 uses 64*56*56
        if in_features == 64 * 56 * 56:
            tensor = tensor.view(out_features, 64, 56, 56).permute(0, 2, 3, 1).reshape(out_features, in_features)

    # Ensure it's on CPU and correct dtype before numpy conversion
    if tensor.dtype == torch.float16 or tensor.dtype == torch.bfloat16:
        tensor = tensor.to(torch.float32)
    elif tensor.dtype == torch.int64:
        tensor = tensor.to(torch.int32)
        
    data = tensor.detach().cpu().contiguous().numpy()
        
    with open(path, "wb") as f:
        f.write(data.tobytes())

def map_param_name(name):
    """
    Maps PyTorch parameter names (e.g. 'group1_block1.conv1.weight') 
    to TunX style ('group1_block1_conv1.weight')
    """
    parts = name.split(".")
    # The last part is 'weight' or 'bias'
    param_type = parts[-1]
    # Everything before is the module path, replace dots with underscores
    module_path = "_".join(parts[:-1])
    mapped_name = f"{module_path}.{param_type}"
    if mapped_name.startswith("fc."):
        mapped_name = "output." + param_type
    return mapped_name

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="tunx_v1")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--dump_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.dump_dir, exist_ok=True)
    
    # Set deterministic seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f"Instantiating model {args.model} on {args.device}...")
    model = get_model(args.model).to(args.device)
    model.train() # Make sure model is in train mode
    
    # Dump initial parameters
    print("Dumping initial parameters...")
    for name, param in model.named_parameters():
        mapped_name = map_param_name(name)
        save_tensor_bin(param, os.path.join(args.dump_dir, f"{mapped_name}.bin"), name=name)
        
    # Generate deterministic inputs
    input_shape = get_input_shape(args.model, args.batch_size)
    is_lm = "gpt2" in args.model
    
    if is_lm:
        inputs = torch.randint(0, 50257, input_shape, dtype=torch.long, device=args.device)
        labels = torch.randint(0, 50257, input_shape, dtype=torch.long, device=args.device)
    else:
        inputs = torch.randn(input_shape, dtype=torch.float32, device=args.device)
        labels = torch.randint(0, 100, (args.batch_size,), dtype=torch.long, device=args.device)
        
    print("Dumping inputs and labels...")
    save_tensor_bin(inputs, os.path.join(args.dump_dir, "inputs.bin"))
    
    # Save as INT32 for TunX (CrossEntropyLoss expects class indices)
    save_tensor_bin(labels.to(torch.int32), os.path.join(args.dump_dir, "labels.bin"))

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=3e-4)
    optimizer.zero_grad()

    print("Running forward pass...")
    outputs = model(inputs)
    save_tensor_bin(outputs, os.path.join(args.dump_dir, "outputs.bin"))

    print("Running backward pass...")
    # Standard loss computation
    if is_lm:
        loss = torch.nn.functional.cross_entropy(outputs.view(-1, 50257), labels.view(-1))
    else:
        loss = torch.nn.functional.cross_entropy(outputs, labels)
        
    loss.backward()
    
    print("Dumping gradients...")
    for name, param in model.named_parameters():
        if param.grad is not None:
            mapped_name = map_param_name(name)
            save_tensor_bin(param.grad, os.path.join(args.dump_dir, f"{mapped_name}.grad.bin"), name=name)
            
    print("Running optimizer step...")
    optimizer.step()
    
    print("Dumping updated parameters...")
    for name, param in model.named_parameters():
        mapped_name = map_param_name(name)
        save_tensor_bin(param, os.path.join(args.dump_dir, f"{mapped_name}.updated.bin"), name=name)
        
    print(f"Dump complete for {args.model} in {args.dump_dir}")

if __name__ == "__main__":
    main()
