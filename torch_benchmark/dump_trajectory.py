#!/usr/bin/env python3
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import csv
import math

from dump_utils import get_model, get_input_shape, save_tensor_bin, save_mapped_param
from torch_trainer import get_model_config
from torch.utils.data import DataLoader

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="resnet50")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--dump-dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-aug", action="store_true", help="Disable augmentation for dataset")
    args = parser.parse_args()

    os.makedirs(args.dump_dir, exist_ok=True)
    
    # Set deterministic seeds and algorithms
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Enforce deterministic FP32 protocol
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Disable TF32 to ensure pure FP32 math equivalence
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")

    print(f"Instantiating model {args.model} on {args.device}...")
    model = get_model(args.model).to(args.device)
    model.train()
    
    # Dump initial parameters
    print("Dumping initial parameters...")
    for name, param in model.named_parameters():
        save_mapped_param(param, name, args.dump_dir, suffix=".bin")
        
    # Generate deterministic inputs for all steps from dataset
    model_name_mapped = args.model
    if "resnet50" in model_name_mapped:
        internal_model_name = "resnet50_imagenet100"
    elif "gpt2" in model_name_mapped:
        internal_model_name = "gpt2_small"
    else:
        internal_model_name = model_name_mapped

    cfg = get_model_config(internal_model_name)
    
    if args.no_aug and "resnet50" in internal_model_name:
        import torchvision.transforms as T
        from torch_trainer import ImageNet100Dataset
        IMAGENET_MEAN = [0.485, 0.456, 0.406]
        IMAGENET_STD = [0.229, 0.224, 0.225]
        train_set = ImageNet100Dataset(
            root=os.getenv("IMAGENET100_ROOT", "data/imagenet-100"),
            train=True,
            transform=T.Compose([
                T.Resize(256),
                T.CenterCrop(224),
                T.ToTensor(),
                T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ])
        )
    else:
        train_set = cfg["train_dataset"]()
    
    g = torch.Generator()
    g.manual_seed(args.seed)

    print(f"Generating indices trajectory for {args.steps} steps...")
    dataset_len = len(train_set)
    total_samples = args.steps * args.batch_size
    
    indices = torch.randint(0, dataset_len, (total_samples,), generator=g, dtype=torch.int32)
    
    indices_np = indices.numpy().astype(np.int32)
    indices_path = os.path.join(args.dump_dir, "indices_trajectory.bin")
    indices_np.tofile(indices_path)
    
    is_lm = "gpt2" in args.model
        
    # No need to dump massive inputs/labels binaries anymore

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=3e-4)

    # Prepare CSV logging
    csv_path = os.path.join(args.dump_dir, f"pt_trajectory_seed_{args.seed}.csv")
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    if is_lm:
        csv_writer.writerow(["step", "loss", "perplexity"])
    else:
        csv_writer.writerow(["step", "loss", "accuracy"])

    print(f"Running training loop for {args.steps} steps...")
    
    indices_reshaped = indices.view(args.steps, args.batch_size)
    
    for step in range(args.steps):
        batch_indices = indices_reshaped[step]
        
        batch_inputs = []
        batch_labels = []
        for idx in batch_indices:
            inp, lbl = train_set[idx.item()]
            batch_inputs.append(inp)
            batch_labels.append(lbl)
            
        inputs = torch.stack(batch_inputs).to(args.device)
        if isinstance(batch_labels[0], torch.Tensor):
            labels = torch.stack(batch_labels).to(args.device)
        else:
            labels = torch.tensor(batch_labels, dtype=torch.long, device=args.device)
        
        optimizer.zero_grad()
        
        outputs = model(inputs)
        
        # Standard loss computation
        if is_lm:
            loss = torch.nn.functional.cross_entropy(outputs.view(-1, 50257), labels.view(-1))
            ppl = math.exp(min(loss.item(), 20))
            csv_writer.writerow([step + 1, loss.item(), ppl])
        else:
            loss = torch.nn.functional.cross_entropy(outputs, labels)
            _, predicted = outputs.max(1)
            correct = predicted.eq(labels).sum().item()
            acc = 100.0 * correct / labels.size(0)
            csv_writer.writerow([step + 1, loss.item(), acc])
            
        loss.backward()
        optimizer.step()
        
        if (step + 1) % 100 == 0:
            print(f"Step {step + 1}/{args.steps} | Loss: {loss.item():.4f}")

    csv_file.close()
    print(f"Dump trajectory complete for {args.model} in {args.dump_dir}")

if __name__ == "__main__":
    main()
