import os
import sys
import random
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from distribution_network.supermarket_network import (dist, N, W1_IDX, W2_IDX)  # noqa: E402, E501
from distribution_network.routing_heuristics import (calculate_sweep_savings_heuristic, apply_two_opt_untangling)  # noqa: E402, E501
from distribution_network.plot_network import (plot_final_routes)  # noqa: E402

# --- VEHICLE & COST CONSTANTS ---
VAN_CAPACITY = 5
LORRY_CAPACITY = 18
VAN_COST_PER_MILE = 2.0
LORRY_COST_PER_MILE = 3.0


def calculate_subroute_cost(route):
    """Calculates the cost of a single sub-route
    based on distance and vehicle type.
    """
    route_dist = sum(dist[route[k]][route[k+1]] for k in range(len(route)-1))
    stores_visited = len(route) - 2  # Subtract the start and end depot

    if stores_visited == 0:
        return 0
    elif stores_visited <= VAN_CAPACITY:
        return route_dist * VAN_COST_PER_MILE
    else:
        return route_dist * LORRY_COST_PER_MILE


def get_human_readable_name(node_idx):
    """Converts a matrix index into a readable location string
    (e.g., 'W1' or 'Store 5').
    """
    if node_idx == W1_IDX:
        return "W1"
    if node_idx == W2_IDX:
        return "W2"
    # +1 because Stores 1-23 are at indices 0-22
    return f"Store {node_idx + 1}"


def save_solution_to_file(best_routes, best_cost, filename="distribution_network/optimal_routing.txt"):  # noqa: E501
    """Writes the final routing strategy to a text file."""
    with open(filename, "w", encoding="utf-8") as file:
        file.write("--- OPTIMAL SUPERMARKET ROUTING STRATEGY ---\n")
        file.write(f"Total Daily Cost: £{best_cost:.2f}\n\n")

        for idx, route in enumerate(best_routes):
            stores_visited = len(route) - 2
            vehicle_type = "Van" if stores_visited <= VAN_CAPACITY else "Lorry"
            cost_rate = VAN_COST_PER_MILE if vehicle_type == "Van" else LORRY_COST_PER_MILE  # noqa: E501
            route_cost = calculate_subroute_cost(route)

            # Format the routes
            locations_list = [get_human_readable_name(n) for n in route]
            locations_str = " -> ".join(locations_list)

            file.write(f"ROUTE {idx + 1}:\n")
            file.write(f"- Warehouse: {get_human_readable_name(route[0])}\n")
            file.write(f"- Vehicle Assigned: {vehicle_type} (£{cost_rate:.2f}/mile)\n")  # noqa: E501
            file.write(f"- Stores Visited: {stores_visited}\n")
            file.write(f"- Route Cost: £{route_cost:.2f}\n")
            file.write(f"- Location Path: {locations_str}\n\n")

    print(f"\nOptimization complete!\nSaved: '{filename}'.")  # noqa: E501


# --- MAIN ACO ALGORITHM ---
def run_aco():
    # ACO Hyperparameters
    num_ants = 40
    generations = 100
    # Importance of pheromones
    alpha = 1.0
    # Importance of the heuristic (Set high to respect the Savings/Sweep)
    beta = 3.0
    evaporation = 0.1  # Pheromone decay rate

    # Initialize uniform pheromone matrix
    pheromone_matrix = np.ones((N, N))

    global_best_routes = []
    global_best_cost = float('inf')

    print("Initializing Ant Colony Optimization...")

    for gen in range(generations):
        for ant in range(num_ants):
            unvisited_stores = set(range(23))  # Indices 0 to 22
            ant_routes = []
            ant_total_cost = 0

            # Keep building routes until all stores are supplied
            while unvisited_stores:
                # Pick a warehouse for the new sub-route
                # based on the closest unvisited store
                sample_store = random.choice(list(unvisited_stores))
                warehouse = W1_IDX if dist[W1_IDX][sample_store] < dist[W2_IDX][sample_store] else W2_IDX  # noqa: E501

                route = [warehouse]
                curr = warehouse

                # Build the sub-route node by node
                while True:
                    probs = []
                    valid_nodes = list(unvisited_stores)

                    # Ant can choose to return to warehouse
                    # if it holds at least 1 store
                    if len(route) - 1 > 0:
                        valid_nodes.append(warehouse)

                    # HARD CONSTRAINT: If Lorry is full,
                    # the ant MUST return to warehouse
                    if len(route) - 1 == LORRY_CAPACITY:
                        valid_nodes = [warehouse]

                    # Calculate transition probabilities
                    for candidate in valid_nodes:
                        if candidate == warehouse:
                            # Heuristic to go home
                            heuristic_value = 1.0 / dist[curr][warehouse]
                        elif curr == warehouse:
                            # Heuristic leaving warehouse
                            heuristic_value = 1.0 / dist[curr][candidate]
                        else:
                            heuristic_value = calculate_sweep_savings_heuristic(warehouse, curr, candidate)  # noqa: E501

                        # Standard ACO probability formula:
                        # (Pheromone^alpha) * (Heuristic^beta)
                        p = (pheromone_matrix[curr][candidate] ** alpha) * (heuristic_value ** beta)  # noqa: E501
                        probs.append(p)

                    # Probabilistic selection for the next node
                    probs = np.array(probs) / sum(probs)
                    next_node = np.random.choice(valid_nodes, p=probs)
                    route.append(next_node)

                    if next_node == warehouse:
                        break  # Route successfully closed
                    else:
                        unvisited_stores.remove(next_node)
                        curr = next_node

                # Apply 2-Opt Untangling
                optimized_route = apply_two_opt_untangling(route)
                ant_routes.append(optimized_route)
                ant_total_cost += calculate_subroute_cost(optimized_route)

            # Track the global best ant
            if ant_total_cost < global_best_cost:
                global_best_cost = ant_total_cost
                global_best_routes = ant_routes
                print(f"Generation {gen+1}: New best cost found -> £{global_best_cost:.2f}")  # noqa: E501

        # Global Pheromone Update (Evaporation + Elitist Deposit)
        pheromone_matrix = pheromone_matrix * (1 - evaporation)
        for r in global_best_routes:
            for i in range(len(r)-1):
                pheromone_matrix[r[i]][r[i+1]] += 1000.0 / global_best_cost

    # Save final results to a text file
    save_solution_to_file(global_best_routes, global_best_cost)

    # Generate and save the visualization of the final routes
    plot_final_routes(global_best_routes)


if __name__ == "__main__":
    run_aco()
