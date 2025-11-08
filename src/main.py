# src/main.py

import networkx as nx
import matplotlib.pyplot as plt
import sys
import os  # Import os

# Import all the necessary components from our other modules
try:
    # We are in src/, so we import from the .py files directly
    from environment import (
        create_city_graph,
        update_environment,
        visualize_risk_matrix,
        visualize_risk_graph,
        visualize_graph  # Import this to show the initial map
    )
    from fuzzy_system import create_risk_assessment_system
    from genetic_algorithm import run_genetic_algorithm, calculate_fitness
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Please ensure all .py files (environment.py, fuzzy_system.py, etc.) are in the 'src' folder.")
    sys.exit(1)
except ModuleNotFoundError as e:
    print(f"Error: {e}. A module was not found.")
    print("Please ensure all .py files are in the 'src' folder.")
    sys.exit(1)


def get_user_start_node(node_labels):
    """
    Prompts the user to select a valid start node and returns its integer ID.
    """
    label_to_id = {label: id for id, label in node_labels.items()}

    print("\n--- Node Selection ---")
    available_nodes_str = ", ".join(sorted(label_to_id.keys()))
    print(f"Available nodes: {available_nodes_str}")

    while True:
        try:
            start_label = input(f"Enter a START node letter (e.g., 'A'): ").upper().strip()
            if start_label == 'P':
                print("Error: Start node cannot be the same as the end node ('P'). Try again.")
            elif start_label in label_to_id:
                return label_to_id[start_label]  # Return the integer ID
            else:
                print(f"Error: Node '{start_label}' is not in the graph. Please try again.")
        except EOFError:
            sys.exit("Input cancelled. Exiting.")
        except KeyboardInterrupt:
            sys.exit("\nProgram interrupted. Exiting.")


def get_user_end_node(node_labels, start_node_id):
    """
    Prompts the user to select a valid end node and returns its integer ID.
    """
    label_to_id = {label: id for id, label in node_labels.items()}
    start_node_label = node_labels[start_node_id]

    print(f"Start Node is set to: {start_node_label}")

    while True:
        try:
            end_label = input(f"Enter a END node letter (e.g., 'A'): ").upper().strip()
            if end_label == start_node_label:
                print(f"Error: Start node cannot be the same as the end node ('{start_node_label}'). Try again.")
            elif end_label in label_to_id:
                return label_to_id[end_label]  # Return the integer ID
            else:
                print(f"Error: Node '{end_label}' is not in the graph. Please try again.")
        except EOFError:
            sys.exit("Input cancelled. Exiting.")
        except KeyboardInterrupt:
            sys.exit("\nProgram interrupted. Exiting.")


# --- NEW FUNCTION ---
def get_user_scenario():
    """
    Interactively builds a 'changes' dictionary based on user input.
    """
    print("\n--- Create a 'What-If' Scenario ---")
    print("You will now specify changes to the 'perfect' road conditions.")

    changes_dict = {
        'traffic': [],
        'weather': [],
        'road_quality': []
    }

    # This helper function gets the changes for one factor
    def get_changes_for_factor(factor_name, conditions):
        while True:
            try:
                num_changes = int(input(f"\nHow many roads have changed '{factor_name}'? (Enter 0 or more): ").strip())
                break
            except ValueError:
                print("Invalid input. Please enter a number.")

        changes_list = []
        for i in range(num_changes):
            while True:
                road_str = input(f"  - Road {i + 1} (e.g., 'A-B'): ").upper().strip()
                if '-' in road_str and len(road_str.split('-')) == 2:
                    break
                print("Invalid format. Please use 'Node1-Node2' (e.g., 'E-I').")

            while True:
                condition_str = input(f"  - Condition for {road_str} ({', '.join(conditions)}): ").lower().strip()
                if condition_str in conditions:
                    break
                print(f"Invalid condition. Please enter one of: {', '.join(conditions)}")

            changes_list.append((road_str, condition_str))
        return changes_list

    # Get changes for each factor
    changes_dict['traffic'] = get_changes_for_factor('traffic', ['low', 'medium', 'high'])
    changes_dict['weather'] = get_changes_for_factor('weather', ['good', 'okay', 'bad'])
    changes_dict['road_quality'] = get_changes_for_factor('road_quality', ['good', 'average', 'poor'])

    print("\nCustom scenario created successfully.")
    return changes_dict


# --- END OF NEW FUNCTION ---

def visualize_final_routes(G, pos, labels, ga_route, dijkstra_route):
    """
    Visualizes the final comparison between the GA route and Dijkstra's route.
    """
    plt.figure(figsize=(16, 12))

    nx.draw(G, pos, labels=labels, with_labels=True, node_color='skyblue',
            node_size=800, font_size=10, font_weight='bold')
    edge_labels = nx.get_edge_attributes(G, 'distance')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)

    dijkstra_edges = list(zip(dijkstra_route, dijkstra_route[1:]))
    nx.draw_networkx_edges(G, pos, edgelist=dijkstra_edges, edge_color='r', width=3.0,
                           style='dashed')

    ga_edges = list(zip(ga_route, ga_route[1:]))
    nx.draw_networkx_edges(G, pos, edgelist=ga_edges, edge_color='g', width=3.5,
                           style='solid')

    plt.legend(handles=[
        plt.Line2D([0], [0], color='g', lw=3.5, label='Fuzzy-Genetic Route (Smartest)'),
        plt.Line2D([0], [0], color='r', lw=3.0, ls='--', label="Dijkstra's Route (Shortest)")
    ], loc='upper left', fontsize=12)

    plt.title("Route Comparison: Fuzzy-Genetic vs. Dijkstra's", fontsize=20)
    # CORRECTED PATH: Save to 'results/' not '../results/'
    plt.savefig('../results/final_route_comparison.png')
    print("\nFinal comparison image saved to 'results/final_route_comparison.png'.")
    plt.show()


def get_path_distance(G, path):
    """Calculates the total distance of a given path."""
    dist = 0.0
    for i in range(len(path) - 1):
        dist += G.edges[path[i], path[i + 1]]['distance']
    return round(dist, 2)


def path_to_labels(path, node_labels):
    """Converts a path of IDs [0, 4, 8] to a string 'A -> E -> I'."""
    if not path:
        return "No path found"
    return " -> ".join([node_labels.get(node_id, '?') for node_id in path])


# --- Main Program Execution ---
if __name__ == "__main__":

    # --- 1. SETUP THE ENVIRONMENT ---
    print("--- Phase 1: Loading Environment ---")

    # Ensure results directory exists
    if not os.path.exists('results'):
        os.makedirs('results')

    city_graph, node_positions, node_labels_map = create_city_graph()

    if city_graph is None:
        sys.exit("Failed to load graph. Please run 'python src/graph_builder.py' first.")

    risk_system = create_risk_assessment_system()

    # Show the user the map they are working with
    print("Displaying the city map with physical distances...")
    visualize_graph(city_graph, node_positions, node_labels_map, filename='city_map_initial.png')

    # --- 2. GET USER INPUT (START) ---
    START_NODE = get_user_start_node(node_labels_map)
    END_NODE = get_user_end_node(node_labels_map, START_NODE)

    start_label = node_labels_map[START_NODE]
    end_label = node_labels_map[END_NODE]
    print(f"\nSelected path: '{start_label}' to '{end_label}'")

    # --- 3. RUN THE "SHORTEST" PATH (DIJKSTRA'S) ---
    print("\n" + "=" * 50)
    print("--- Phase 2: Finding the 'Shortest' Path (Baseline) ---")

    try:
        route_dijkstra = nx.shortest_path(city_graph, source=START_NODE, target=END_NODE, weight='distance')
        dist_dijkstra = get_path_distance(city_graph, route_dijkstra)

        print(f"Dijkstra's algorithm finds the shortest path by *distance*.")
        print(f"  - Path: {path_to_labels(route_dijkstra, node_labels_map)}")
        print(f"  - Total Distance: {dist_dijkstra}")
    except nx.NetworkXNoPath:
        print(f"Error: No path found from {node_labels_map[START_NODE]} to {node_labels_map[END_NODE]}.")
        sys.exit()
    print("=" * 50)

    # --- 4. GET USER INPUT (SCENARIO) & UPDATE ENVIRONMENT ---
    custom_changes = get_user_scenario()

    print(f"\n--- Phase 3: Applying Custom Scenario ---")
    city_graph = update_environment(city_graph, node_labels_map, custom_changes)

    # Show the risk matrix *after* changes
    print("\nDisplaying the new Risk Matrix based on your changes...")
    visualize_risk_matrix(city_graph, risk_system, scenario_name="Custom")

    # Show the risk graph *after* changes
    print("\nDisplaying the new Risk Graph based on your changes...")
    visualize_risk_graph(city_graph, node_positions, node_labels_map, risk_system,
                         filename='city_map_with_risk_custom.png')

    # Calculate the risk of Dijkstra's "shortest" path in this new, "bad" environment
    fitness_dijkstra = calculate_fitness(route_dijkstra, city_graph, risk_system)
    risk_dijkstra = 1.0 / fitness_dijkstra

    print(f"\nCalculating the *actual fuzzy risk* of the 'shortest' (Dijkstra's) path...")
    print(f"  - Dijkstra's Path Risk Score: {risk_dijkstra:.2f}")
    print("=" * 50)

    # --- 5. RUN THE "SMARTEST" PATH (FUZZY-GENETIC) ---
    print(f"\n--- Phase 4: Finding the 'Smartest' Path (Fuzzy-Genetic) ---")
    print("Running Genetic Algorithm to find the path with the LOWEST FUZZY RISK...")

    best_route_ga, risk_ga = run_genetic_algorithm(
        graph=city_graph,
        start_node=START_NODE,
        end_node=END_NODE,
        pop_size=100,
        generations=50,
        mutation_rate=0.3,
        crossover_rate=0.8
    )
    dist_ga = get_path_distance(city_graph, best_route_ga)

    print("\nGenetic Algorithm finished.")
    print(f"  - Path: {path_to_labels(best_route_ga, node_labels_map)}")
    print(f"  - Total Distance: {dist_ga}")
    print(f"  - Fuzzy Risk Score: {risk_ga:.2f}")
    print("=" * 50)

    # --- 6. FINAL COMPARISON & VISUALIZATION ---
    print(f"\n--- Final Results for Your Custom Scenario ---")

    print(f"[Dijkstra's 'Shortest' Path]")
    print(f"  Route: {path_to_labels(route_dijkstra, node_labels_map)}")
    print(f"  Distance: {dist_dijkstra} | Actual Fuzzy Risk: {risk_dijkstra:.2f}")

    print(f"\n[Fuzzy-Genetic 'Smartest' Path]")
    print(f"  Route: {path_to_labels(best_route_ga, node_labels_map)}")
    print(f"  Distance: {dist_ga} | Actual Fuzzy Risk: {risk_ga:.2f}")

    print("\n--- Conclusion ---")
    if risk_ga < risk_dijkstra:
        print("The Fuzzy-Genetic Algorithm found a smarter, lower-risk route. SUCCESS!")
    else:
        print("The Fuzzy-Genetic and Dijkstra's routes had a similar risk score.")

    visualize_final_routes(city_graph, node_positions, node_labels_map, best_route_ga, route_dijkstra)