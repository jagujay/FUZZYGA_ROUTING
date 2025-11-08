import random
import networkx as nx
import numpy as np

# We import the functions from our previously created modules
from environment import create_city_graph, update_environment
from fuzzy_system import create_risk_assessment_system


# --- Helper Functions for the GA ---

def generate_random_route(graph, start_node, end_node):
    """
    Generates a single, valid, random route from start to end.
    A simple way to get diverse valid paths is to use a shortest path algorithm
    on a graph with randomized edge weights.
    """
    # Create a temporary copy of the graph to modify weights
    temp_graph = graph.copy()
    for u, v in temp_graph.edges():
        temp_graph.edges[u, v]['weight'] = random.uniform(0.5, 1.5)

    try:
        # Use Dijkstra's algorithm to find a path on the randomized graph
        path = nx.shortest_path(temp_graph, source=start_node, target=end_node, weight='weight')
        return path
    except nx.NetworkXNoPath:
        return None  # No path exists


def calculate_fitness(route, graph, risk_system):
    """
    Calculates the fitness of a single route. Fitness is the inverse of total risk.
    """
    total_risk = 0.0
    for i in range(len(route) - 1):
        u = route[i]
        v = route[i + 1]

        # Get edge data
        edge_data = graph.get_edge_data(u, v)

        # Pass inputs to the fuzzy system
        risk_system.input['traffic'] = edge_data['traffic']
        risk_system.input['weather'] = edge_data['weather']
        risk_system.input['road_quality'] = edge_data['road_quality']

        # Compute the risk for this segment
        risk_system.compute()
        segment_risk = risk_system.output['segment_risk']

        # Add the segment risk multiplied by distance to get a cost
        total_risk += segment_risk * edge_data['distance']

    # Handle the case of zero risk to avoid division by zero
    if total_risk == 0:
        return float('inf')

    # Fitness is the inverse of the total risk, so a lower risk means higher fitness
    return 1.0 / total_risk


def selection(population, fitnesses, num_parents):
    """
    Selects the best individuals from the current generation as parents.
    Using Tournament Selection.
    """
    parents = []
    for _ in range(num_parents):
        # Select 3 random individuals for the tournament
        tournament_size = 3
        participants_indices = random.sample(range(len(population)), tournament_size)

        # Find the best individual among the participants
        best_participant_index = -1
        max_fitness = -1
        for index in participants_indices:
            if fitnesses[index] > max_fitness:
                max_fitness = fitnesses[index]
                best_participant_index = index

        parents.append(population[best_participant_index])
    return parents


def crossover(parent1, parent2):
    """
    Performs crossover between two parent routes.
    Finds a common node and swaps the tails of the paths.
    """
    common_nodes = list(set(parent1) & set(parent2))
    # Exclude start and end nodes from being crossover points
    common_nodes = [node for node in common_nodes if node != parent1[0] and node != parent1[-1]]

    if not common_nodes:
        # If no common nodes, return one of the parents as is
        return parent1

    crossover_point = random.choice(common_nodes)

    # Create the child by combining parts of the parents
    crossover_idx1 = parent1.index(crossover_point)
    crossover_idx2 = parent2.index(crossover_point)

    child = parent1[:crossover_idx1] + parent2[crossover_idx2:]
    return child


def mutation(route, graph):
    """
    Performs mutation on a single route by finding a new sub-path.
    """
    if len(route) < 3:
        return route  # Cannot mutate a route that's too short

    # Select two random, non-adjacent nodes in the route
    idx1, idx2 = sorted(random.sample(range(1, len(route) - 1), 2))

    start_sub_path = route[idx1]
    end_sub_path = route[idx2]

    # Find a new path between these two nodes
    new_sub_path = generate_random_route(graph, start_sub_path, end_sub_path)

    if new_sub_path:
        # Reconstruct the route with the new sub-path
        mutated_route = route[:idx1] + new_sub_path + route[idx2 + 1:]
        return mutated_route

    return route  # Return original if no new sub-path found


# --- Main Genetic Algorithm Function ---

def run_genetic_algorithm(graph, start_node, end_node, pop_size=50, generations=100, crossover_rate=0.8,
                          mutation_rate=0.2):
    """
    Executes the complete Genetic Algorithm.
    """
    # 1. Initialize the fuzzy system
    risk_system = create_risk_assessment_system()

    # 2. Initialize the population
    print("\nInitializing population...")
    population = [generate_random_route(graph, start_node, end_node) for _ in range(pop_size)]
    population = [route for route in population if route is not None]  # Filter out None values

    best_route_overall = None
    best_fitness_overall = -1

    print("Starting genetic algorithm evolution...")
    for gen in range(generations):
        # 3. Calculate fitness for the entire population
        fitnesses = [calculate_fitness(route, graph, risk_system) for route in population]

        # 4. Select parents for the next generation
        num_parents = int(pop_size * 0.5)  # 50% of population will be parents
        parents = selection(population, fitnesses, num_parents)

        # 5. Create the next generation
        offspring_size = pop_size - len(parents)
        offspring = []
        for _ in range(offspring_size):
            # Crossover
            parent1, parent2 = random.sample(parents, 2)
            if random.random() < crossover_rate:
                child = crossover(parent1, parent2)
            else:
                child = parent1  # No crossover, just copy a parent

            # Mutation
            if random.random() < mutation_rate:
                child = mutation(child, graph)

            offspring.append(child)

        # The new population consists of the best parents and the new offspring
        population = parents + offspring

        # Track the best route found so far
        best_fitness_in_gen = max(fitnesses)
        if best_fitness_in_gen > best_fitness_overall:
            best_fitness_overall = best_fitness_in_gen
            best_route_overall = population[np.argmax(fitnesses)]

        if (gen + 1) % 10 == 0:
            print(f"Generation {gen + 1}/{generations} | Best Fitness: {best_fitness_overall:.4f}")

    print("Genetic algorithm finished.")
    total_risk = 1.0 / best_fitness_overall
    return best_route_overall, total_risk


if __name__ == "__main__":

    print("--- Testing Genetic Algorithm Module ---")

    # --- 1. Setup a realistic environment for testing ---
    city_graph, node_positions, node_labels_map = create_city_graph()

    if city_graph is None:
        print("Error: graph_data.json not found. Run graph_builder.py first.")
        exit()

    # --- 2. Create the exact 'high_risk' scenario from our demo ---
    print("Creating a 'high_risk' test scenario...")
    test_changes = {
        'traffic': [('E-I', 'high'), ('M-P', 'high')],
        'weather': [],
        'road_quality': [('E-I', 'poor'), ('M-P', 'poor')]
    }

    # Update the graph with these changes
    city_graph = update_environment(city_graph, node_labels_map, test_changes)

    # --- 3. Define the start and end nodes for the test ---
    # We will test the exact 'A' to 'P' scenario
    START_NODE = 0  # 'A'
    END_NODE = 15  # 'P'

    print(f"Test running from '{node_labels_map[START_NODE]}' to '{node_labels_map[END_NODE]}'")

    # --- 4. Run the Genetic Algorithm ---
    best_route, best_risk = run_genetic_algorithm(
        graph=city_graph,
        start_node=START_NODE,
        end_node=END_NODE,
        pop_size=50,
        generations=50
    )

    # --- 5. Print the results ---
    print("\n--- GA Test Results ---")
    if best_route:
        # Helper function to convert [0, 5, ...] to "A -> F -> ..."
        def path_to_labels_test(path, labels):
            return " -> ".join([labels.get(node_id, '?') for node_id in path])


        print(f"Best Route Found: {path_to_labels_test(best_route, node_labels_map)}")
        print(f"Total Risk Score: {best_risk:.4f}")
    else:
        print("Error: Genetic algorithm did not find a route.")