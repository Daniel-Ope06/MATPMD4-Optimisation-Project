"""
Simulated Annealing Optimisation Algorithm

This script navigates the pre-computed objective space to find the optimal
grid resolution (R).
"""

import os
import math
import random
import pandas as pd


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(SCRIPT_DIR, "data", "data.csv")
OUTPUT_PATH = os.path.join(SCRIPT_DIR, "optimal_resolution.txt")


def load_and_normalize_data(filepath):
    """
    Loads the empirical data and applies Min-Max normalization to the
    compute latency to bind it to the [0, 1] domain.
    Returns a dictionary for O(1) lookup:
    {R: {'norm_time': val, 'occlusion': val}}
    """
    df = pd.read_csv(filepath)

    time_min = df['Average Compute Time (ms)'].min()
    time_max = df['Average Compute Time (ms)'].max()

    objective_space = {}
    for _, row in df.iterrows():
        R = int(row['Resolution (R)'])
        t = row['Average Compute Time (ms)']
        occ = row['Occlusion Rate']

        # Min-Max Normalisation
        norm_t = (t - time_min) / (time_max - time_min)

        objective_space[R] = {
            'norm_time': norm_t,
            'occlusion': occ,
            'raw_time': t  # Kept for final logging
        }

    return objective_space


def evaluate_cost(R, objective_space, w_time=0.4, w_occ=0.6):
    """
    The Total Cost Function J(R).
    Note: Weighting occlusion (0.6) slightly higher than time (0.4)
    prioritizes UGV safety over micro-optimizations in speed.
    """
    norm_time = objective_space[R]['norm_time']
    occlusion = objective_space[R]['occlusion']

    return (w_time * norm_time) + (w_occ * occlusion)


def tweak(current_R):
    """
    Generates a neighboring state by stepping R up or down by 2.
    Bounds the output strictly between 10 and 100.
    """
    step = random.choice([-2, 2])
    new_R = current_R + step

    # Boundary enforcement
    if new_R < 10:
        return 12
    elif new_R > 100:
        return 98

    return new_R


def simulated_annealing(objective_space, initial_temp=1.0, cooling_rate=0.95, epochs=1000):  # noqa: E501
    """
    The core Simulated Annealing algorithm.
    """
    # Initialization
    available_resolutions = list(objective_space.keys())
    current_R = random.choice(available_resolutions)
    best_R = current_R

    current_cost = evaluate_cost(current_R, objective_space)
    best_cost = current_cost

    temp = initial_temp

    # The Cooling Loop
    for iteration in range(epochs):
        next_R = tweak(current_R)
        next_cost = evaluate_cost(next_R, objective_space)

        delta_E = next_cost - current_cost

        # Acceptance Logic
        if delta_E < 0:
            current_R = next_R
            current_cost = next_cost

            if current_cost < best_cost:
                best_R = current_R
                best_cost = current_cost

        else:
            acceptance_probability = math.exp(-delta_E / temp)
            if random.random() < acceptance_probability:
                current_R = next_R
                current_cost = next_cost

        # Decrease the temperature
        temp = temp * cooling_rate

        if temp < 0.0001:
            break

    return best_R, best_cost


def main():
    if not os.path.exists(CSV_PATH):
        print(f"[!] Error: Dataset not found at {CSV_PATH}")
        return

    print("[*] Loading and normalising objective space...")
    objective_space = load_and_normalize_data(CSV_PATH)

    print("[*] Executing Simulated Annealing (Multi-Start)...")
    best_overall_R = None
    best_overall_cost = float('inf')

    for run in range(5):
        best_R, best_cost = simulated_annealing(objective_space)
        if best_cost < best_overall_cost:
            best_overall_cost = best_cost
            best_overall_R = best_R

    final_raw_time = objective_space[best_overall_R]['raw_time']
    final_occ = objective_space[best_overall_R]['occlusion']

    # Format the final output string
    report_output = (
        "==================================================\n"
        "OPTIMAL STATE DISCOVERED via Simulated Annealing\n"
        "==================================================\n"
        f"    Grid Resolution : {best_overall_R}x{best_overall_R}\n"
        f"    Total Math Cost : {best_overall_cost:.4f}\n"
        f"    Compute Latency : {final_raw_time:.2f} ms\n"
        f"    Occlusion Rate  : {final_occ*100:.1f} %\n"
        "==================================================\n"
    )

    # Print to console
    print("\n" + report_output)

    # Save to text file
    with open(OUTPUT_PATH, "w") as text_file:
        text_file.write(report_output)

    print(f"\n -> Output successfully saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
