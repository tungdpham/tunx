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
    parser.add_argument("--input-root", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 420, 4200])
    parser.add_argument("--rolling-window", type=int, default=None,
                        help="Also plot trailing means over this many steps (full windows only)")
    args = parser.parse_args()
    if args.rolling_window is not None and not 1 <= args.rolling_window <= args.steps:
        parser.error("--rolling-window must be between 1 and --steps")

    seeds = args.seeds
    loaded_seeds = []
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(base_dir, ".."))
    
    output_dir = args.output_dir or project_root
    os.makedirs(output_dir, exist_ok=True)
    pt_dfs = []
    tunx_dfs = []
    
    metric_col = 'perplexity' if 'gpt2' in args.model else 'accuracy'
    
    for seed in seeds:
        dump_dir = os.path.join(args.input_root or os.path.join(project_root, "dump"), f"trajectory_{args.model}_{seed}")
        tunx_dir = os.path.join(args.input_root or os.path.join(project_root, "dump"), f"tunx_trajectory_{args.model}_{seed}")
        
        pt_csv = os.path.join(dump_dir, f"pt_trajectory_seed_{seed}.csv")
        tunx_csv = os.path.join(tunx_dir, f"tunx_trajectory_seed_{seed}.csv")
        
        if os.path.exists(pt_csv) and os.path.exists(tunx_csv):
            pt = pd.read_csv(pt_csv).iloc[:args.steps]
            tunx = pd.read_csv(tunx_csv).iloc[:args.steps]
            expected = np.arange(1, args.steps + 1)
            if not np.array_equal(pt['step'].values, expected) or not np.array_equal(tunx['step'].values, expected):
                raise ValueError(f"Seed {seed}: expected exactly {args.steps} aligned steps")
            pt_dfs.append(pt)
            tunx_dfs.append(tunx)
            loaded_seeds.append(seed)
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

    for i, seed in enumerate(loaded_seeds):
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
        plt.savefig(os.path.join(output_dir, f"{args.model}_convergence_loss_seed_{seed}.png"))
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
        plt.savefig(os.path.join(output_dir, f"{args.model}_convergence_{metric_col}_seed_{seed}.png"))
        plt.close()

        if args.rolling_window is not None:
            window = args.rolling_window
            for column in ['loss', metric_col]:
                fig, ax = plt.subplots(figsize=(10, 6))
                for frame, label, color, style in [
                    (pt_dfs[i], 'PyTorch', 'darkorange', '--'),
                    (tunx_dfs[i], 'TunX', 'royalblue', '-'),
                ]:
                    # Align each full trailing window to its final training step.
                    values = frame[column].rolling(window, min_periods=window).mean()
                    ax.plot(frame['step'], values, label=label, color=color,
                            linestyle=style, linewidth=2)
                ylabel = {'loss': 'Mean training loss',
                          'accuracy': 'Mean training accuracy (%)',
                          'perplexity': 'Mean batch perplexity'}[column]
                ax.set(xlabel='Training step (window end)', ylabel=ylabel,
                       title=f'{args.model}: {window}-step rolling {column} (seed {seed})')
                ax.legend()
                ax.grid(True, linestyle='--', alpha=0.4)
                fig.tight_layout()
                stem = f'{args.model}_rolling_{column}_window_{window}_seed_{seed}'
                fig.savefig(os.path.join(output_dir, stem + '.png'), dpi=160)
                fig.savefig(os.path.join(output_dir, stem + '.pdf'))
                plt.close(fig)

    print(f"Plots saved to {output_dir}")

if __name__ == "__main__":
    main()
