#!/usr/bin/env python3
"""Plot convergence comparison between PyTorch and TunX for ResNet50 FP64 1000-step run."""

import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path

DATA_FILE = "convergence_1000_fp64.txt"
OUT_FILE = "convergence_plot.png"

steps, pt_losses, tunx_losses = [], [], []
with open(DATA_FILE) as f:
    for line in f:
        m = re.match(r'\s*(\d+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|', line)
        if m:
            steps.append(int(m.group(1)))
            pt_losses.append(float(m.group(2)))
            tunx_losses.append(float(m.group(3)))

steps = np.array(steps)
pt_losses = np.array(pt_losses)
tunx_losses = np.array(tunx_losses)
abs_diff = np.abs(pt_losses - tunx_losses)

# Smoothed losses (moving average)
def moving_avg(x, w=30):
    return np.convolve(x, np.ones(w)/w, mode='valid')

w = 30
s_smooth = steps[w-1:]
pt_smooth = moving_avg(pt_losses, w)
tunx_smooth = moving_avg(tunx_losses, w)
diff_smooth = moving_avg(abs_diff, w)

# --- Dark-mode figure ---
BG     = "#0d1117"
PANEL  = "#161b22"
PT_C   = "#58a6ff"   # blue
TUNX_C = "#3fb950"   # green
DIFF_C = "#f78166"   # orange-red
GRID   = "#30363d"
TEXT   = "#e6edf3"
MUTED  = "#8b949e"

fig = plt.figure(figsize=(14, 9), facecolor=BG)
fig.suptitle(
    "ResNet50 Convergence  ·  PyTorch vs TunX  ·  FP64  ·  LR=1e-3 (After Weight Fix)",
    fontsize=16, fontweight='bold', color=TEXT, y=0.97
)

gs = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.08)

# ---- Top panel: raw + smoothed losses ----
ax1 = fig.add_subplot(gs[0])
ax1.set_facecolor(PANEL)

ax1.plot(steps, pt_losses,   color=PT_C,   alpha=0.15, linewidth=0.6)
ax1.plot(steps, tunx_losses, color=TUNX_C, alpha=0.15, linewidth=0.6)
ax1.plot(s_smooth, pt_smooth,   color=PT_C,   linewidth=2.2, label="PyTorch (FP64)")
ax1.plot(s_smooth, tunx_smooth, color=TUNX_C, linewidth=2.2, label="TunX   (FP64)")

ax1.set_ylabel("Cross-Entropy Loss", color=TEXT, fontsize=12)
ax1.tick_params(colors=TEXT, which='both')
ax1.spines[:].set_color(GRID)
ax1.tick_params(labelbottom=False)
ax1.set_xlim(1, 1000)
ax1.grid(True, color=GRID, linewidth=0.5, linestyle='--', alpha=0.6)

# Stats annotations
final_pt   = pt_losses[-1]
final_tunx = tunx_losses[-1]
final_diff = abs(final_pt - final_tunx)
ax1.annotate(
    f"Step 1000  PyTorch={final_pt:.4f}  TunX={final_tunx:.4f}  Δ={final_diff:.4f}",
    xy=(0.5, 0.04), xycoords='axes fraction',
    ha='center', fontsize=10, color=MUTED,
    bbox=dict(boxstyle='round,pad=0.3', facecolor=BG, edgecolor=GRID, alpha=0.8)
)

leg = ax1.legend(loc='upper right', fontsize=11,
                 facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT)
ax1.set_title("Loss Trajectory  (faint = per-step, solid = 30-step moving avg)",
              color=MUTED, fontsize=10, pad=6)

# ---- Bottom panel: absolute difference ----
ax2 = fig.add_subplot(gs[1], sharex=ax1)
ax2.set_facecolor(PANEL)

ax2.fill_between(steps, abs_diff, color=DIFF_C, alpha=0.15)
ax2.plot(steps, abs_diff, color=DIFF_C, alpha=0.3, linewidth=0.6)
ax2.plot(s_smooth, diff_smooth, color=DIFF_C, linewidth=2.0, label="|PyTorch − TunX|")

# Avg lines per quartile to visualise trend
q_size = len(steps) // 4
for q in range(4):
    sl = slice(q*q_size, (q+1)*q_size)
    avg = abs_diff[sl].mean()
    ax2.hlines(avg, steps[sl][0], steps[sl][-1],
               colors=MUTED, linestyles='dashed', linewidth=1.0, alpha=0.6)

ax2.set_xlabel("Training Step", color=TEXT, fontsize=12)
ax2.set_ylabel("|Δ Loss|", color=TEXT, fontsize=11)
ax2.tick_params(colors=TEXT, which='both')
ax2.spines[:].set_color(GRID)
ax2.grid(True, color=GRID, linewidth=0.5, linestyle='--', alpha=0.6)
ax2.set_ylim(bottom=0)
ax2.legend(loc='upper right', fontsize=10,
           facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT)

# Trailing avg annotation
trail_avg = abs_diff[-100:].mean()
ax2.annotate(f"last-100 avg Δ = {trail_avg:.4f}",
             xy=(900, trail_avg), xytext=(700, trail_avg * 2.5 + 0.05),
             color=MUTED, fontsize=9,
             arrowprops=dict(arrowstyle='->', color=MUTED, lw=1),
             bbox=dict(boxstyle='round,pad=0.25', facecolor=BG, edgecolor=GRID))

for ax in [ax1, ax2]:
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{int(x)}'))
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)

plt.savefig(OUT_FILE, dpi=160, bbox_inches='tight', facecolor=BG)
print(f"Saved to {OUT_FILE}")

# Print summary stats
print(f"\n{'Step Range':<20} {'Avg |Δ Loss|':>14}")
print("-" * 36)
for i in range(0, 1000, 100):
    sl = slice(i, i+100)
    print(f"{i+1:>4}-{i+100:<12}   {abs_diff[sl].mean():>10.4f}")
print(f"\nOverall avg |Δ|: {abs_diff.mean():.4f}")
print(f"Max    |Δ|:      {abs_diff.max():.4f} (step {steps[abs_diff.argmax()]})")
