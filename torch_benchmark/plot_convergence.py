#!/usr/bin/env python3
import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="resnet50")
    parser.add_argument("--steps", type=int, default=1000)
    args = parser.parse_args()

    seeds = [42, 420, 4200]
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(base_dir, ".."))
    
    pt_dfs = []
    tunx_dfs = []
    
    metric_col = 'perplexity' if 'gpt2' in args.model else 'accuracy'
    
    for seed in seeds:
        dump_dir = os.path.join(project_root, "dump", f"trajectory_{args.model}_{seed}")
        tunx_dir = os.path.join(project_root, "dump", f"tunx_trajectory_{args.model}_{seed}")
        
        pt_csv = os.path.join(dump_dir, f"pt_trajectory_seed_{seed}.csv")
        tunx_csv = os.path.join(tunx_dir, f"tunx_trajectory_seed_{seed}.csv")
        
        if os.path.exists(pt_csv) and os.path.exists(tunx_csv):
            pt_dfs.append(pd.read_csv(pt_csv))
            tunx_dfs.append(pd.read_csv(tunx_csv))
        else:
            print(f"Warning: Missing CSVs for seed {seed}")

    if not pt_dfs or not tunx_dfs:
        print("Error: No data found.")
        return

    # Aggregate PyTorch data
    pt_steps = pt_dfs[0]['step'].values
    pt_loss = np.array([df['loss'].values for df in pt_dfs])
    pt_metric = np.array([df[metric_col].values for df in pt_dfs])
    
    pt_loss_mean = np.mean(pt_loss, axis=0)
    pt_loss_std = np.std(pt_loss, axis=0)
    pt_metric_mean = np.mean(pt_metric, axis=0)
    pt_metric_std = np.std(pt_metric, axis=0)

    # Aggregate TunX data
    tunx_steps = tunx_dfs[0]['step'].values
    tunx_loss = np.array([df['loss'].values for df in tunx_dfs])
    tunx_metric = np.array([df[metric_col].values for df in tunx_dfs])
    
    tunx_loss_mean = np.mean(tunx_loss, axis=0)
    tunx_loss_std = np.std(tunx_loss, axis=0)
    tunx_metric_mean = np.mean(tunx_metric, axis=0)
    tunx_metric_std = np.std(tunx_metric, axis=0)

    # Output mean +- std of final metric to terminal
    print(f"--- Convergence Summary ({args.model}) ---")
    print(f"PyTorch Final {metric_col.capitalize()}: {pt_metric_mean[-1]:.4f} +- {pt_metric_std[-1]:.4f}")
    print(f"TunX Final {metric_col.capitalize()}   : {tunx_metric_mean[-1]:.4f} +- {tunx_metric_std[-1]:.4f}")

    for i, seed in enumerate(seeds):
        if i >= len(pt_dfs) or i >= len(tunx_dfs):
            continue
            
        # Plot Loss for this seed
        plt.figure(figsize=(10, 6))
        plt.plot(pt_steps, pt_dfs[i]['loss'].values, label=f'PyTorch (Seed {seed})', color='orange', linestyle='--')
        plt.plot(tunx_steps, tunx_dfs[i]['loss'].values, label=f'TunX (Seed {seed})', color='blue', linestyle='-')
        
        plt.xlabel('Steps')
        plt.ylabel('Loss')
        plt.title(f'{args.model} Convergence (Loss) over {args.steps} Steps (Seed {seed})')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(project_root, f"{args.model}_convergence_loss_seed_{seed}.png"))
        plt.close()
        
        # Plot Metric for this seed
        plt.figure(figsize=(10, 6))
        plt.plot(pt_steps, pt_dfs[i][metric_col].values, label=f'PyTorch (Seed {seed})', color='orange', linestyle='--')
        plt.plot(tunx_steps, tunx_dfs[i][metric_col].values, label=f'TunX (Seed {seed})', color='blue', linestyle='-')
        
        plt.xlabel('Steps')
        plt.ylabel(metric_col.capitalize())
        plt.title(f'{args.model} Convergence ({metric_col.capitalize()}) over {args.steps} Steps (Seed {seed})')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(project_root, f"{args.model}_convergence_{metric_col}_seed_{seed}.png"))
        plt.close()

    print(f"Plots saved to {project_root}")

if __name__ == "__main__":
    main()
