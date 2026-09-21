#!/usr/bin/env python3
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import csv
import math

from dump_utils import get_model, get_input_shape, map_param_name, save_tensor_bin, save_mapped_param
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
    parser.add_argument("--dump-inputs", action="store_true",
                        help="Dump each pre-forward batch for cross-framework debugging")
    parser.add_argument("--dump-params", action="store_true",
                        help="Dump trainable parameters after each optimizer step")
    parser.add_argument("--dump-loss-gradients", action="store_true",
                        help="Dump dLoss/dOutputs for identical-upstream-gradient diagnostics")
    parser.add_argument("--debug-layer", type=str, default=None,
                        help="Dump backward input/output gradients for one mapped layer")
    parser.add_argument("--dump-activations", action="store_true",
                        help="Dump intermediate activations and gradients for all layers per step")
    parser.add_argument("--fp64", action="store_true",
                        help="Run model and tensors in double precision (FP64)")
    parser.add_argument("--compact-inputs", action="store_true", help="Cache sampled ResNet50 images once as uint8")
    parser.add_argument("--preload-images", action="store_true",
                        help="Load all ResNet50 source images into RAM before shuffling; no extra disk copy")
    parser.add_argument("--preload-image-budget-gib", type=float, default=20,
                        help="Maximum compressed image bytes to preload in GiB (default: 20)")
    args = parser.parse_args()
    if args.preload_images and (args.model != "resnet50" or args.preload_image_budget_gib <= 0):
        parser.error("--preload-images requires resnet50 and a positive memory budget")
    if args.compact_inputs and (args.model != "resnet50" or not args.no_aug or args.fp64):
        parser.error("--compact-inputs requires FP32 resnet50 with --no-aug")

    os.makedirs(args.dump_dir, exist_ok=True)
    
    if args.fp64:
        torch.set_default_dtype(torch.float64)
    
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
    if args.fp64:
        model = model.double()
    # Match TunX's NHWC storage for image models while keeping NCHW logical shapes.
    is_lm = "gpt2" in args.model
    if not is_lm:
        model = model.to(memory_format=torch.channels_last)
        print("Using channels-last memory format for model and image inputs")
    model.train()

    debug_gradients = {}
    debug_activations = {}
    if args.debug_layer:
        for name, module in model.named_modules():
            mapped_name = map_param_name(name + ".weight").removesuffix(".weight")
            if mapped_name == args.debug_layer:
                def capture_backward_gradients(module, grad_input, grad_output):
                    if grad_input and grad_input[0] is not None:
                        debug_gradients["input"] = grad_input[0].detach()
                    if grad_output and grad_output[0] is not None:
                        debug_gradients["output"] = grad_output[0].detach()

                def capture_forward_output(module, inputs, output):
                    debug_activations["output"] = output.detach()

                module.register_forward_hook(capture_forward_output)
                module.register_full_backward_hook(capture_backward_gradients)
                break
        else:
            raise ValueError(f"Unknown mapped debug layer: {args.debug_layer}")

    # All-layer activation/gradient hooks for --dump-activations
    execution_order = []
    all_step_activations = {}  # mapped_name -> tensor
    all_step_grad_inputs = {}  # mapped_name -> tensor
    all_step_grad_outputs = {}  # mapped_name -> tensor
    if args.dump_activations:
        def make_act_hook(mapped_name):
            def hook(module, inputs, output):
                if isinstance(output, torch.Tensor):
                    if not any(module.children()) and mapped_name not in execution_order:
                        execution_order.append(mapped_name)
                    all_step_activations[mapped_name] = output.detach().clone()
            return hook
        def make_grad_hook(mapped_name):
            def hook(module, grad_input, grad_output):
                if grad_input and grad_input[0] is not None:
                    all_step_grad_inputs[mapped_name] = grad_input[0].detach().clone()
                if grad_output and grad_output[0] is not None:
                    all_step_grad_outputs[mapped_name] = grad_output[0].detach().clone()
            return hook
        for name, module in model.named_modules():
            if not name:
                continue
            mapped_name = map_param_name(name + ".weight").removesuffix(".weight")
            if mapped_name:
                module.register_forward_hook(make_act_hook(mapped_name))
                module.register_full_backward_hook(make_grad_hook(mapped_name))
    
    # Dump initial parameters
    print("Dumping initial parameters...")
    topology_order = []
    for name, param in model.named_parameters():
        saved_names = save_mapped_param(param, name, args.dump_dir, suffix=".bin")
        if saved_names:
            topology_order.extend(saved_names)
            
    with open(os.path.join(args.dump_dir, "topology_order.txt"), "w") as f:
        for item in topology_order:
            f.write(f"{item}\n")
        
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
    
    if args.preload_images:
        train_set.preload_images(args.preload_image_budget_gib * 2**30)

    g = torch.Generator()
    g.manual_seed(args.seed)

    print(f"Generating indices trajectory for {args.steps} steps...")
    dataset_len = len(train_set)
    total_samples = args.steps * args.batch_size
    
    if args.model == "resnet50":
        # Shuffle without replacement, then reshuffle after each complete dataset pass.
        indices = torch.cat([torch.randperm(dataset_len, generator=g, dtype=torch.int32)
                             for _ in range((total_samples + dataset_len - 1) // dataset_len)])[:total_samples]
    else:
        indices = torch.randint(0, dataset_len, (total_samples,), generator=g, dtype=torch.int32)
    
    indices_np = indices.numpy().astype(np.int32)
    indices_path = os.path.join(args.dump_dir, "indices_trajectory.bin")
    indices_np.tofile(indices_path)
    if is_lm:
        # Python indexes sequences; TunX indexes raw token offsets. Use int64 to
        # support corpora with more than 2**31 tokens without wrapping offsets.
        (indices_np.astype(np.int64) * train_set.seq_len).tofile(
            os.path.join(args.dump_dir, "indices_trajectory_i64.bin"))
    if args.compact_inputs:
        from compact_image_cache import CompactImageCache
        train_set = CompactImageCache(train_set, os.path.join(args.dump_dir, "image_cache"))
    
    # No need to dump massive inputs/labels binaries anymore

    # Optimizer
    if is_lm:
        optimizer = optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=3e-4)
    else:
        optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=1e-4)

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
        debug_activations.clear()
        debug_gradients.clear()
        batch_indices = indices_reshaped[step]
        
        batch_inputs = []
        batch_labels = []
        for idx in batch_indices:
            inp, lbl = train_set[idx.item()]
            batch_inputs.append(inp)
            batch_labels.append(lbl)
            
        inputs = torch.stack(batch_inputs).to(args.device)
        if not is_lm:
            inputs = inputs.contiguous(memory_format=torch.channels_last)
        if isinstance(batch_labels[0], torch.Tensor):
            labels = torch.stack(batch_labels).to(args.device)
        else:
            labels = torch.tensor(batch_labels, dtype=torch.long, device=args.device)

        if args.dump_inputs:
            save_tensor_bin(inputs, os.path.join(args.dump_dir, f"inputs_step_{step + 1}.bin"))
            save_tensor_bin(labels, os.path.join(args.dump_dir, f"labels_step_{step + 1}.bin"))
        
        optimizer.zero_grad()
        all_step_activations.clear()
        all_step_grad_inputs.clear()
        all_step_grad_outputs.clear()
        
        outputs = model(inputs)
        if args.dump_loss_gradients:
            outputs.retain_grad()
        if args.debug_layer:
            save_tensor_bin(
                debug_activations["output"],
                os.path.join(args.dump_dir, f"{args.debug_layer}.output_step_{step + 1}.bin"),
                name=args.debug_layer,
            )
        
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
        if args.dump_loss_gradients:
            save_tensor_bin(outputs.grad, os.path.join(
                args.dump_dir, f"loss_gradient_step_{step + 1}.bin"))
        if args.dump_activations:
            step_str = step + 1
            for layer_name, tensor in all_step_activations.items():
                save_tensor_bin(
                    tensor,
                    os.path.join(args.dump_dir, f"{layer_name}.output_step_{step_str}.bin"),
                    name=layer_name,
                )
            for layer_name, tensor in all_step_grad_inputs.items():
                save_tensor_bin(
                    tensor,
                    os.path.join(args.dump_dir, f"{layer_name}.grad_input_step_{step_str}.bin"),
                    name=layer_name,
                )
            for layer_name, tensor in all_step_grad_outputs.items():
                save_tensor_bin(
                    tensor,
                    os.path.join(args.dump_dir, f"{layer_name}.grad_output_step_{step_str}.bin"),
                    name=layer_name,
                )
        if args.debug_layer:
            for direction, gradient in debug_gradients.items():
                save_tensor_bin(
                    gradient,
                    os.path.join(args.dump_dir,
                                 f"{args.debug_layer}.grad_{direction}_step_{step + 1}.bin"),
                    name=args.debug_layer,
                )
        if args.dump_params:
            step_dir = os.path.join(args.dump_dir, f"params_step_{step + 1}")
            os.makedirs(step_dir, exist_ok=True)
            for name, param in model.named_parameters():
                if param.grad is not None:
                    save_mapped_param(param.grad, name, step_dir, suffix=".grad.bin")

        optimizer.step()

        if args.dump_params:
            for name, param in model.named_parameters():
                save_mapped_param(param, name, step_dir, suffix=".updated.bin")
        
        csv_file.flush()
        if (step + 1) % 10 == 0:
            print(f"Step {step + 1}/{args.steps} | Loss: {loss.item():.4f}", flush=True)

    if args.dump_activations:
        with open(os.path.join(args.dump_dir, "topology_order.txt"), "w") as f:
            f.write("".join(f"{name}\n" for name in execution_order))

    csv_file.close()
    print(f"Dump trajectory complete for {args.model} in {args.dump_dir}")

if __name__ == "__main__":
    main()
