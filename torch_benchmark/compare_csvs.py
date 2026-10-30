import os
import argparse
import pandas as pd

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="resnet50")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(base_dir, ".."))
    
    pt_csv = os.path.join(project_root, "dump", f"trajectory_{args.model}_{args.seed}", f"pt_trajectory_seed_{args.seed}.csv")
    tunx_csv = os.path.join(project_root, "dump", f"tunx_trajectory_{args.model}_{args.seed}", f"tunx_trajectory_seed_{args.seed}.csv")
    
    if not os.path.exists(pt_csv) or not os.path.exists(tunx_csv):
        print(f"CSVs not found for seed {args.seed}")
        return

    pt_df = pd.read_csv(pt_csv)
    tunx_df = pd.read_csv(tunx_csv)

    print(f"Step | PyTorch Loss | TunX Loss | Difference")
    print("-" * 45)
    for i in range(len(pt_df)):
        step = pt_df.iloc[i]['step']
        pt_loss = pt_df.iloc[i]['loss']
        tunx_loss = tunx_df.iloc[i]['loss']
        diff = abs(pt_loss - tunx_loss)
        if i % 1 == 0 or i == len(pt_df) - 1:
            print(f"{int(step):4d} | {pt_loss:12.6f} | {tunx_loss:9.6f} | {diff:10.6f}")

    # Plot the trajectories
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 6))
        plt.plot(pt_df['step'], pt_df['loss'], label='PyTorch', alpha=0.7)
        plt.plot(tunx_df['step'], tunx_df['loss'], label='TunX', alpha=0.7, linestyle='--')
        plt.xlabel('Step')
        plt.ylabel('Loss')
        plt.title(f'Convergence Trajectory: {args.model} (Seed {args.seed})')
        plt.legend()
        plt.grid(True)
        
        plot_path = os.path.join(project_root, "dump", f"convergence_plot_{args.model}_{args.seed}.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"\nPlot saved to {plot_path}")
    except ImportError:
        print("\nmatplotlib not installed, skipping plot generation. Install with: uv add matplotlib")

if __name__ == "__main__":
    main()
