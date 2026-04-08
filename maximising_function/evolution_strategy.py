import math
import random
import itertools


def evaluate_fitness(x, y, z):
    """The mathematical objective function to maximize."""
    term1 = -math.exp(-(x - 0.55*z)**2) * math.cos(81*(x + z + 0.12*y))
    term2 = math.sin(44*(y + 0.03*(x - z)))
    term3 = -math.cos(69 * math.sin(x*z + 0.07*y))
    term4 = -math.sin(1.32*(x - z))
    term5 = -0.32*(z + 0.22*x*z)**2 * math.exp(math.sin(63*(z - x)))
    term6 = math.cos(77*x) + math.sin(71*z)
    term7 = -math.exp(-y**2) * math.cos(75*y)
    term8 = math.sin(43*y) + math.cos(73*y)
    term9 = 0.05*(x**2 + y**2 + z**2)

    return (
        term1 + term2 + term3 +
        term4 + term5 + term6 +
        term7 + term8 + term9
    )


def clamp(value, min_bound=0.0, max_bound=5.0):
    """Enforces strict [0, 5] boundaries on coordinate values."""
    return max(min_bound, min(value, max_bound))


def get_fitness(candidate):
    """Helper function to sort candidates by their fitness score."""
    return candidate['fitness']


def run_evolution_strategy(
        mu, lambda_, generations=300, initial_mutation_step_size=0.8):
    """Executes a single (mu + lambda) ES trial."""
    population = []

    # Initialize parent population
    for _ in range(mu):
        candidate = {'x': random.uniform(0, 5), 'y': random.uniform(
            0, 5), 'z': random.uniform(0, 5)}
        candidate['fitness'] = evaluate_fitness(
            candidate['x'], candidate['y'], candidate['z'])
        population.append(candidate)

    global_best = max(population, key=get_fitness)
    mutation_step_size = initial_mutation_step_size

    for gen in range(generations):
        offspring = []

        # Generate lambda offspring via Gaussian mutation
        for _ in range(lambda_):
            parent = random.choice(population)
            child = {
                'x': clamp(parent['x'] + random.gauss(0, mutation_step_size)),
                'y': clamp(parent['y'] + random.gauss(0, mutation_step_size)),
                'z': clamp(parent['z'] + random.gauss(0, mutation_step_size))
            }
            child['fitness'] = evaluate_fitness(
                child['x'], child['y'], child['z'])
            offspring.append(child)

        # (mu + lambda) selection
        combined_population = population + offspring
        combined_population.sort(key=get_fitness, reverse=True)
        population = combined_population[:mu]

        if population[0]['fitness'] > global_best['fitness']:
            global_best = population[0]

        # Decay mutation size slightly to fine-tune at the peak
        mutation_step_size *= 0.99

    return global_best


def execute_parameter_sweep():
    """Runs the ES across varying parameters to ensure robust optimization."""
    mu_options = [10, 30, 50]
    lambda_options = [50, 150, 300]
    trials_per_combo = 3

    best_overall = {'fitness': -float('inf')}
    best_params = {}

    print("Beginning hyperparameter sweep...\n")

    for mu, lambda_ in itertools.product(mu_options, lambda_options):
        print(f"Testing mu={mu}, lambda={lambda_}...")
        for trial in range(trials_per_combo):
            result = run_evolution_strategy(mu, lambda_)

            if result['fitness'] > best_overall['fitness']:
                best_overall = result
                best_params = {'mu': mu, 'lambda': lambda_}

    # Format the final output string
    output_text = (
        "========================================\n"
        "          OPTIMIZATION COMPLETE         \n"
        "========================================\n"
        f"Best Parameters: mu={best_params['mu']}, lambda={best_params['lambda']}\n"  # noqa: E501
        f"Max Fitness:     {best_overall['fitness']:.6f}\n"
        f"Coordinates:     x={best_overall['x']:.6f}, y={best_overall['y']:.6f}, z={best_overall['z']:.6f}\n"  # noqa: E501
    )

    # Print to console
    print("\n" + output_text)

    # Save to text file
    filepath = "maximising_function/optimization_results.txt"
    with open(filepath, "w") as file:
        file.write(output_text)

    print(f"Results successfully saved to '{filepath}'")


def run_focused_evolution_strategy(
        center_x, center_y, center_z, radius=0.2,
        mu=10, lambda_=300, generations=300, initial_mutation_step_size=0.05):
    """Executes a fine-tuning (mu + lambda) ES trial around a specific peak."""
    population = []

    # Define local boundaries
    min_x, max_x = max(0.0, center_x - radius), min(5.0, center_x + radius)
    min_y, max_y = max(0.0, center_y - radius), min(5.0, center_y + radius)
    min_z, max_z = max(0.0, center_z - radius), min(5.0, center_z + radius)

    # Initialize parent population clustered tightly around the known peak
    for _ in range(mu):
        candidate = {
            'x': random.uniform(min_x, max_x),
            'y': random.uniform(min_y, max_y),
            'z': random.uniform(min_z, max_z)
        }
        candidate['fitness'] = evaluate_fitness(
            candidate['x'], candidate['y'], candidate['z'])
        population.append(candidate)

    global_best = max(population, key=get_fitness)
    mutation_step_size = initial_mutation_step_size

    for gen in range(generations):
        offspring = []

        # Generate lambda offspring via micro-mutations
        for _ in range(lambda_):
            parent = random.choice(population)
            child = {
                'x': clamp(parent['x'] + random.gauss(0, mutation_step_size), min_x, max_x),  # noqa: E501
                'y': clamp(parent['y'] + random.gauss(0, mutation_step_size), min_y, max_y),  # noqa: E501
                'z': clamp(parent['z'] + random.gauss(0, mutation_step_size), min_z, max_z)  # noqa: E501
            }
            child['fitness'] = evaluate_fitness(
                child['x'], child['y'], child['z'])
            offspring.append(child)

        # (mu + lambda) selection
        combined_population = population + offspring
        combined_population.sort(key=get_fitness, reverse=True)
        population = combined_population[:mu]

        if population[0]['fitness'] > global_best['fitness']:
            global_best = population[0]

        # Decay mutation size to fine-tune even closer to the absolute peak
        mutation_step_size *= 0.99

    return global_best


def execute_focused_search():
    """
    Runs the fine-tuning ES around the best coordinates
    found during the sweep."""
    # The best coordinates from parameter sweep
    best_x = 4.000815
    best_y = 1.204875
    best_z = 0.200288

    # Best parameters from sweep
    optimal_mu = 10
    optimal_lambda = 300

    print(f"Beginning focused search around ({best_x}, {best_y}, {best_z})...\n")  # noqa: E501

    # Run the focused search
    fine_tuned_result = run_focused_evolution_strategy(
        center_x=best_x,
        center_y=best_y,
        center_z=best_z,
        radius=0.2,  # Search strictly within +/- 0.2 of the coordinates
        mu=optimal_mu,
        lambda_=optimal_lambda,
        generations=1000
    )

    # Format the final output string
    output_text = (
        "========================================\n"
        "         FINE-TUNING COMPLETE           \n"
        "========================================\n"
        f"Original Fitness: 7.952102\n"
        f"New Max Fitness:  {fine_tuned_result['fitness']:.6f}\n"
        f"New Coordinates:  x={fine_tuned_result['x']:.6f}, y={fine_tuned_result['y']:.6f}, z={fine_tuned_result['z']:.6f}\n"  # noqa: E501
    )

    # Print to console
    print(output_text)

    # Save to text file
    # (appending so it doesn't overwrite initial sweep results)
    filepath = "maximising_function/optimization_results.txt"
    with open(filepath, "a") as file:
        file.write("\n" + output_text)

    print(f"Results successfully appended to '{filepath}'")


if __name__ == "__main__":
    # execute_parameter_sweep()
    execute_focused_search()
