import sys
import os
import math
import matplotlib.pyplot as plt
from matplotlib.patches import Arc


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from distribution_network.supermarket_network import (coords, angle_w1, W1_IDX, W2_IDX)  # noqa: E402, E501


def create_environment_plot():
    fig, ax = plt.subplots(figsize=(7, 7))

    # Plot the Stores (Indices 0 to 22) - Light Blue
    ax.scatter(
        coords[0:23, 0], coords[0:23, 1], c='lightblue',
        edgecolors='steelblue', marker='o', s=80, label='Stores',
        zorder=2)
    for i in range(23):
        ax.annotate(
            rf"${i+1}$",  # 'r' is to set as raw string so it can use latex
            (coords[i, 0] - 0.8, coords[i, 1] + 1.5),
            fontsize=9, zorder=3)

    # Plot the Warehouses (Indices 23 and 24) - Light Red
    ax.scatter(
        coords[23:25, 0], coords[23:25, 1], c='lightcoral',
        edgecolors='darkred', marker='o', s=240, label='Warehouses',
        zorder=2)
    ax.annotate(r"$W1$", (coords[W1_IDX, 0] - 1.5, coords[W1_IDX, 1] - 4.5),
                fontsize=11, fontweight='bold', color='darkred')
    ax.annotate(r"$W2$", (coords[W2_IDX, 0] - 1.5, coords[W2_IDX, 1] - 4.5),
                fontsize=11, fontweight='bold', color='darkred')

    # --- EXAMPLE FOR THE REPORT ---
    # Example: Store 17 (Index 16) from W1 (Index 23)
    STORE_EX = 16
    w1_x, w1_y = coords[W1_IDX, 0], coords[W1_IDX, 1]
    s_x, s_y = coords[STORE_EX, 0], coords[STORE_EX, 1]

    # Draw Euclidean Distance Line
    ax.plot([w1_x, s_x], [w1_y, s_y], color='gray',
            linestyle='--', alpha=0.8, zorder=1)
    mid_x, mid_y = (w1_x + s_x) / 2, (w1_y + s_y) / 2
    line_angle_degrees = math.degrees(angle_w1[STORE_EX])
    ax.text(mid_x - 3, mid_y - 3, f'$d_{{W1, {STORE_EX + 1}}}$',
            color='black', fontsize=12, fontweight='bold',
            rotation=line_angle_degrees, rotation_mode='anchor')

    # Draw Horizontal Reference Line for Polar Angle
    ax.plot([w1_x, w1_x + 10], [w1_y, w1_y], color='darkred',
            linestyle=':', alpha=0.6, zorder=1)

    # Draw the Polar Angle Arc (theta)
    angle_degrees = math.degrees(angle_w1[STORE_EX])
    arc = Arc(
        (w1_x, w1_y), 15, 15, angle=0, theta1=0,
        theta2=angle_degrees, color='darkred', linewidth=1.5, zorder=2)
    ax.add_patch(arc)
    ax.text(w1_x + 6, w1_y + 6, r'$\theta$',
            color='darkred', fontsize=14, fontweight='bold')

    # Formatting
    ax.set_xlabel("X Coordinate (Miles)")
    ax.set_ylabel("Y Coordinate (Miles)")
    ax.legend(loc='lower right', frameon=False, labelspacing=1.2)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)

    plt.tight_layout()
    plt.savefig("distribution_network/supermarket_network.png", dpi=300)
    print("Saved: distribution_network/supermarket_network.png")
    plt.close(fig)


if __name__ == "__main__":
    create_environment_plot()
