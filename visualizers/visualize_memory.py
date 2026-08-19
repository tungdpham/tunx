import argparse

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import matplotlib.cm as cm


def plot_memory(csv_file, offset_column='bfd_offset', output_file='memory_placement.png'):
    df = pd.read_csv(csv_file)

    fig, ax = plt.subplots(figsize=(16, 9))

    max_step = df['free_step'].max()
    max_memory = (df[offset_column] + df['size']).max()

    # Generate colors based on unique edge_uids to easily distinguish them
    unique_edges = df['edge_uid'].unique()

    # Use plt.get_cmap which is more robust across matplotlib versions
    cmap = plt.get_cmap('tab20')
    edge_to_color = {edge: cmap(i % 20) for i, edge in enumerate(unique_edges)}

    for idx, row in df.iterrows():
        # Rectangle(xy, width, height)
        # xy is bottom-left corner
        # x: alloc_step, y: offset
        # width: free_step - alloc_step, height: size

        width = row['free_step'] - row['alloc_step']
        # If allocated and freed in the same step, give it a width of 1 so it's visible
        if width == 0:
            width = 1

        color = edge_to_color[row['edge_uid']]

        rect = patches.Rectangle(
            (row['alloc_step'], row[offset_column]),
            width,
            row['size'],
            linewidth=0.5,
            edgecolor='black',
            facecolor=color,
            alpha=0.8
        )
        ax.add_patch(rect)

    ax.set_xlim(-1, max_step + 2)
    ax.set_ylim(0, max_memory * 1.05)

    ax.set_xlabel('Step', fontsize=12)
    # Format y-axis to show MBs for readability if it's large
    if max_memory > 1024 * 1024:
        ax.set_ylabel('Memory Offset (MB)', fontsize=12)
        ticks = ax.get_yticks()
        ax.set_yticks(ticks)
        ax.set_yticklabels([f'{int(tick / (1024 * 1024))}' for tick in ticks])
    else:
        ax.set_ylabel('Memory Offset (Bytes)', fontsize=12)

    ax.set_title(f'Memory Placement ({offset_column})', fontsize=14)

    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    print(f'Saved visualization to {output_file}')
    # plt.show() # Uncomment if running interactively


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Visualize memory placement from CSV.')
    parser.add_argument('csv_file', nargs='?', type=str, default='build/memory_placement.csv',
                        help='Path to the CSV file (default: memory_placement.csv)')
    parser.add_argument('--offset', type=str, default='bfd_offset', choices=[
                        'naive_offset', 'bfd_offset'], help='Offset column to use (default: bfd_offset)')
    parser.add_argument('--out', type=str, default='memory_placement.png',
                        help='Output image file (default: memory_placement.png)')

    args = parser.parse_args()
    plot_memory(args.csv_file, args.offset, args.out)
