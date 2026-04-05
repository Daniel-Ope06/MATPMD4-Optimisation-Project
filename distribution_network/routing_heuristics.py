import os
import sys
import math

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from distribution_network.supermarket_network import (dist, angle_w1, angle_w2, W1_IDX, W2_IDX)  # noqa: E402, E501


def calculate_sweep_savings_heuristic(warehouse_idx, current_node, candidate_node):  # noqa: E501
    """
    Calculates the desirability of an ant moving to a candidate node, combining
    Clarke-Wright distance savings with an angular sweep penalty.
    """
    # Clarke-Wright Savings:
    # Distance saved by combining the two stores into one route
    savings = dist[warehouse_idx][current_node] + \
        dist[warehouse_idx][candidate_node] - \
        dist[current_node][candidate_node]

    # ACO requires positive heuristic values.
    # If savings are negative, provide a small baseline.
    base_desirability = max(0.001, savings)

    # Sweep Angle Penalty:
    # Determine the polar angle of both nodes relative to the active warehouse
    if warehouse_idx == W1_IDX:
        angle_current = angle_w1[current_node]
        angle_candidate = angle_w1[candidate_node]
    elif warehouse_idx == W2_IDX:
        angle_current = angle_w2[current_node]
        angle_candidate = angle_w2[candidate_node]

    # Calculate the absolute angular difference (0 to pi)
    angle_diff = abs(angle_current - angle_candidate)  # type: ignore
    if angle_diff > math.pi:
        angle_diff = (2 * math.pi) - angle_diff

    # Final Heuristic:
    # High savings and a small angle difference yield the highest score
    # Add 1.0 to angle_diff to mathematically prevent a DivisionByZero error
    # in the event two stores share the exact same polar angle.
    heuristic_value = base_desirability / (1.0 + angle_diff)

    return heuristic_value


def apply_two_opt_untangling(route):
    """
    Applies the 2-Opt local search algorithm to a closed route.
    It mathematically detects intersecting lines and reverses the segment to
    untangle them, guaranteeing a shorter overall distance based on the
    Triangle Inequality.

    Route Representation Example: [23, 15, 6, 8, 23]
    - Index 0 and 4 are the warehouse (W1).
    - Indices 1, 2, 3 are the visited stores.
    If edges cross, this function reverses a sub-slice
    (e.g., reversing [15, 6] to [6, 15]).
    """
    best_route = route.copy()  # Create a separate copy
    tangled = True

    while tangled:
        tangled = False  # Assume no intersecting lines exist on this pass

        # Iterate over every possible pair of non-adjacent edges in the route
        for i in range(1, len(best_route) - 2):
            for j in range(i + 1, len(best_route) - 1):
                if j - i == 1:
                    continue  # Ignore adjacent edges

                # Calculate the current distance of the two crossing edges
                current_dist = dist[best_route[i-1]][best_route[i]] + \
                    dist[best_route[j]][best_route[j+1]]

                # Calculate the distance if uncrossed
                # (connecting i-1 to j, and i to j+1)
                new_dist = dist[best_route[i-1]][best_route[j]] + \
                    dist[best_route[i]][best_route[j+1]]

                # If uncrossing them yields a shorter distance, swap them.
                # Round to 4 decimal places to prevent infinite loops
                # caused by microscopic floating-point math inaccuracies.
                if round(new_dist, 4) < round(current_dist, 4):  # type: ignore
                    # Reverse the routing segment between i and j in place
                    best_route[i:j+1] = best_route[i:j+1][::-1]
                    tangled = True  # Made a change, loop again to verify

    return best_route
