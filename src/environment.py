# src/environment.py

import networkx as nx
import matplotlib.pyplot as plt
import random
import numpy as np
import json
import os  # Import os to check for results directory

# Import the fuzzy system
try:
    from fuzzy_system import create_risk_assessment_system
except ImportError:
    # This allows the script to be run directly if fuzzy_system isn't found
    print("Warning: fuzzy_system.py not found. Only graph creation will be tested.")
    create_risk_assessment_system = None

# This map translates the user's "fuzzy" text input into a crisp number
CONDITION_MAP = {
    'traffic': {'low': 15.0, 'medium': 50.0, 'high': 90.0},
    'weather': {'good': 9.0, 'okay': 5.0, 'bad': 2.0},
    'road_quality': {'good': 9.0, 'average': 5.0, 'poor': 2.0}
}


def create_city_graph():
    """
    Loads the fixed city graph from the 'graph_data.json' file.
    """
    # Create results dir if it doesn't exist
    if not os.path.exists('results'):
        os.makedirs('results')

    try:
        # We are in src/, so we go up one level to find the json file
        with open('../graph_data.json', 'r') as f:
            graph_data = json.load(f)
    except FileNotFoundError:
        print("\n" + "=" * 50)
        print("Error: 'graph_data.json' not found in the root folder.")
        print("Please run 'python src/graph_builder.py' first to create it.")
        print("=" * 50 + "\n")
        return None, None, None
    except Exception as e:
        print(f"Error loading graph_data.json: {e}")
        return None, None, None

    G = nx.Graph()
    node_labels = {}

    for node in graph_data["nodes"]:
        G.add_node(node["id"], label=node["label"])
        node_labels[node["id"]] = node["label"]

    for edge in graph_data["edges"]:
        G.add_edge(edge["source"], edge["target"], distance=edge["distance"])

    pos_str_keys = graph_data["positions"]
    pos = {int(k): tuple(v) for k, v in pos_str_keys.items()}

    print("Fixed city graph loaded successfully from 'graph_data.json'.")
    return G, pos, node_labels


def update_environment(G, node_labels, changes_dict):
    """
    Updates the graph attributes based on a "changes" dictionary.
    1. Sets a "perfect normal" baseline.
    2. Applies all changes from the changes_dict.
    """
    if not G: return None

    print("\nUpdating environment with custom changes...")

    # 1. Set "perfect normal" baseline for all edges
    for u, v in G.edges():
        G.edges[u, v]['road_quality'] = 10.0  # Perfect quality
        G.edges[u, v]['weather'] = 10.0  # Perfect weather
        G.edges[u, v]['traffic'] = 10.0  # Very low traffic

    # 2. Create reverse map for node labels ('A' -> 0)
    label_to_id = {label: id for id, label in node_labels.items()}

    # 3. Apply user's custom changes
    for factor, changes_list in changes_dict.items():
        if factor not in CONDITION_MAP:
            print(f"Warning: Unknown factor '{factor}'. Skipping.")
            continue

        for road_str, condition in changes_list:
            # Parse the road string 'A-B'
            labels = road_str.upper().split('-')
            if len(labels) != 2:
                print(f"Warning: Skipping malformed road '{road_str}'.")
                continue

            u_label, v_label = labels[0], labels[1]

            # Find the integer IDs for the labels
            if u_label in label_to_id and v_label in label_to_id:
                u, v = label_to_id[u_label], label_to_id[v_label]

                if G.has_edge(u, v):
                    # Get the numeric value for the condition
                    value = CONDITION_MAP[factor].get(condition.lower())

                    if value is not None:
                        G.edges[u, v][factor] = value
                        print(f"  - Applied: Road {road_str} {factor} set to '{condition}' ({value})")
                    else:
                        print(f"Warning: Unknown condition '{condition}' for {factor}. Skipping.")
                else:
                    print(f"Warning: No direct road between {u_label}-{v_label}. Skipping.")
            else:
                print(f"Warning: Unknown node labels in '{road_str}'. Skipping.")

    print("Environment updated.")
    return G


def visualize_graph(G, pos, labels, filename='city_map.png'):
    """
    Draws the graph with distance labels.
    """
    if not G: return

    plt.figure(figsize=(16, 12))

    nx.draw(G, pos, labels=labels, with_labels=True, node_color='skyblue',
            node_size=800, font_size=10, font_weight='bold')

    edge_labels = nx.get_edge_attributes(G, 'distance')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)

    plt.title("Simulated City Map (with Distance)", fontsize=20)
    # CORRECTED PATH: Save to 'results/' not '../results/'
    plt.savefig(f'../results/{filename}')
    plt.show()


def visualize_risk_matrix(G, risk_system, scenario_name="Current"):
    """
    Calculates and visualizes the risk scores in a matrix.
    NEW: Saves the matrix to a .txt file.
    """
    if not G: return

    num_nodes = G.number_of_nodes()
    risk_matrix = np.full((num_nodes, num_nodes), np.nan)

    node_labels_map = nx.get_node_attributes(G, 'label')
    node_labels = [node_labels_map.get(i, f'N{i}') for i in range(num_nodes)]

    for u, v in G.edges():
        edge_data = G.get_edge_data(u, v)

        # Feed data into the fuzzy system
        risk_system.input['traffic'] = edge_data['traffic']
        risk_system.input['weather'] = edge_data['weather']
        risk_system.input['road_quality'] = edge_data['road_quality']

        risk_system.compute()
        segment_risk = risk_system.output['segment_risk']

        weighted_risk = segment_risk * edge_data['distance']

        risk_matrix[u, v] = weighted_risk
        risk_matrix[v, u] = weighted_risk

    # --- NEW: Save matrix to .txt file ---
    txt_filename = f'../results/risk_matrix_{scenario_name.lower().replace(" ", "_")}.txt'
    header = "Weighted Risk Matrix (Fuzzy Score * Distance)\nRows/Cols match node order: " + ", ".join(node_labels)
    np.savetxt(txt_filename, risk_matrix, fmt="%.2f", header=header, delimiter='\t')
    print(f"\nRisk matrix data saved to '{txt_filename}'")
    # --- End of new feature ---

    # Visualization
    plt.figure(figsize=(12, 10))
    cmap = plt.cm.YlOrRd
    cmap.set_bad(color='gainsboro')

    plt.imshow(risk_matrix, cmap=cmap, interpolation='nearest')
    plt.colorbar(label='Weighted Segment Risk (Fuzzy Score * Distance)')

    plt.title(f'Weighted Segment Risk Matrix ({scenario_name} Scenario)', fontsize=16)
    plt.xticks(np.arange(num_nodes), node_labels)
    plt.yticks(np.arange(num_nodes), node_labels)

    plt.xlabel('Destination Node')
    plt.ylabel('Start Node')

    plt.tight_layout()
    # CORRECTED PATH: Save to 'results/'
    plt.savefig(f'../results/risk_matrix_{scenario_name.lower().replace(" ", "_")}.png')
    plt.show()


# --- NEW FUNCTION ---
def visualize_risk_graph(G, pos, labels, risk_system, filename='city_map_with_risk.png'):
    """
    Draws the graph, replacing distance labels with weighted risk labels.
    """
    if not G: return

    print(f"\nGenerating risk graph and saving to 'results/{filename}'...")

    # 1. Calculate weighted risk for all edges
    risk_labels = {}
    for u, v in G.edges():
        edge_data = G.get_edge_data(u, v)

        # Feed data into the fuzzy system
        risk_system.input['traffic'] = edge_data['traffic']
        risk_system.input['weather'] = edge_data['weather']
        risk_system.input['road_quality'] = edge_data['road_quality']

        risk_system.compute()
        segment_risk = risk_system.output['segment_risk']

        weighted_risk = segment_risk * edge_data['distance']
        # Format the risk score for the label
        risk_labels[(u, v)] = f"{weighted_risk:.1f}"

    # 2. Draw the graph
    plt.figure(figsize=(16, 12))
    nx.draw(G, pos, labels=labels, with_labels=True, node_color='skyblue',
            node_size=800, font_size=10, font_weight='bold')

    # 3. Draw the new risk labels
    nx.draw_networkx_edge_labels(G, pos, edge_labels=risk_labels, font_size=8, font_color='firebrick')

    plt.title("Simulated City Map (with Weighted Risk)", fontsize=20)
    # CORRECTED PATH: Save to 'results/'
    plt.savefig(f'../results/{filename}')
    plt.show()


# --- END OF NEW FUNCTION ---


# This block allows you to run this script directly to test its functionality
if __name__ == "__main__":

    # The __main__ block is now simplified, as most functionality
    # will be driven by main.py. This just tests the graph loading.

    city_graph, node_positions, node_labels_map = create_city_graph()

    if city_graph:
        print("\n--- Testing Graph Visualization ---")
        visualize_graph(city_graph, node_positions, node_labels_map, filename='city_map_initial.png')

        # You can optionally test the new 'normal' state here
        if create_risk_assessment_system:
            print("\n--- Testing 'Normal' Scenario Risk Matrix ---")
            risk_sys = create_risk_assessment_system()
            # Create a dummy 'changes' dict to test the 'normal' baseline
            empty_changes = {'traffic': [], 'weather': [], 'road_quality': []}
            city_graph_normal = update_environment(city_graph, node_labels_map, empty_changes)
            visualize_risk_matrix(city_graph_normal, risk_sys, scenario_name="Normal_Baseline")
            visualize_risk_graph(city_graph_normal, node_positions, node_labels_map, risk_sys,
                                 filename='city_map_with_risk_normal.png')