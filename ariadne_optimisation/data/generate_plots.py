"""
Data Visualization Script

Reads the dataset and generates three plots:
1. Resolution vs. Compute Time
2. Resolution vs. Path Occlusion Rate
3. The Pareto Front (Compute Time vs. Occlusion Rate)
"""

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(SCRIPT_DIR, "data.csv")
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))

if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)


def plot_resolution_vs_time(df):
    plt.figure(figsize=(7, 4.5))

    plt.plot(
        df['Resolution (R)'], df['Average Compute Time (ms)'],
        marker='o', linestyle='-', color='cornflowerblue', markersize=4
    )
    plt.xlabel("Grid Resolution (R)")
    plt.ylabel("Average Compute Time (ms)")
    plt.grid(True, linestyle='--', alpha=0.7)

    save_path = os.path.join(SCRIPT_DIR, "plot_1_resolution_vs_time.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.05)
    plt.close()
    print(f" -> Saved: {save_path}")


def plot_resolution_vs_occlusion(df):
    plt.figure(figsize=(7, 4.5))

    plt.plot(
        df['Resolution (R)'], df['Occlusion Rate'],
        marker='o', linestyle='-', color='indianred', markersize=4
    )
    plt.xlabel("Grid Resolution (R)")
    plt.ylabel("Path Occlusion Rate")
    plt.grid(True, linestyle='--', alpha=0.7)

    save_path = os.path.join(SCRIPT_DIR, "plot_2_resolution_vs_occlusion.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.05)
    plt.close()
    print(f" -> Saved: {save_path}")


def plot_pareto_front(df):
    """
    Generates the Pareto Front (Time vs. Occlusion Rate).
    """
    plt.figure(figsize=(8, 5))

    plt.plot(
        df['Average Compute Time (ms)'], df['Occlusion Rate'],
        linestyle='-', color='gray', alpha=0.4, zorder=1
    )
    scatter = plt.scatter(
        df['Average Compute Time (ms)'], df['Occlusion Rate'],
        c=df['Resolution (R)'], cmap='coolwarm', s=50, edgecolor='k', zorder=5
    )

    cbar = plt.colorbar(scatter)
    cbar.set_label('Grid Resolution (R)')

    plt.xlabel("Average Compute Time (ms)")
    plt.ylabel("Path Occlusion Rate")
    plt.grid(True, linestyle='--', alpha=0.7)

    save_path = os.path.join(SCRIPT_DIR, "plot_3_pareto_front.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.05)
    plt.close()
    print(f" -> Saved: {save_path}")


def main():
    print("[*] Loading empirical dataset...")
    df = pd.read_csv(CSV_PATH)

    print("[*] Generating plots...")
    plot_resolution_vs_time(df)
    plot_resolution_vs_occlusion(df)
    plot_pareto_front(df)

    print("\n[+] Plot generation complete.")


if __name__ == "__main__":
    main()
