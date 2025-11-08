# src/graph_builder.py

import json


def build_and_save_graph():
    """
    Defines the static graph data based on 'sc graph1.png'.
    Saves it to 'graph_data.json' in the project's root folder.
    """

    # Map node integers (0-16) to their letter labels (17 nodes total)
    node_labels = {
        0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H',
        8: 'I', 9: 'J', 10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O',
        15: 'P', 16: 'Q'
    }

    # Define all edges with their fixed distances from 'sc graph1.png'
    edges_with_distances = [
        (0, 11, 8), (0, 1, 5), (0, 5, 4), (0, 4, 4),  # A's edges
        (1, 2, 7),  # B's edge
        (2, 3, 3),  # C's edges
        (3, 6, 5), (3, 7, 7),  (3, 11, 6), # D's edges
        (4, 5, 5), (4, 8, 1),  (4, 14, 8), # E's edges
        (5, 6, 6), (5, 8, 3),  # F's edges
        (6, 9, 8), (6, 7, 10),  # G's edges (D-G, F-G, J-G, H-G)
        (7, 10, 9),  # H's edges (D-H, G-H, K-H)
        (8, 9, 14), (8, 12, 2),  # I's edges
        (9, 10, 6),  # J's edges
        (10, 13, 10),  # K's edges
        (12, 14, 2), (12, 15, 3), (12, 13, 2),  # M's edges
        (13, 16, 4),  # N's edges
        (15, 16, 1)  # P's edge (M-P, N-P, Q-P)
    ]

    # --- Manual (x, y) positions to match 'sc graph1.png' layout ---
    pos = {
        0: (5, 15),  # A
        1: (5, 10),  # B
        2: (5, 5),  # C
        3: (5, 0),  # D
        4: (7, 15),  # E
        5: (7, 10),  # F
        6: (7, 5),  # G
        7: (7, 0),  # H
        8: (9, 12.5),  # I
        9: (9, 7.5),  # J
        10: (9, 2.5),  # K
        11: (3, 7.5),  # L
        12: (11, 10),  # M
        13: (11, 5),  # N
        14: (13, 12.5),  # O
        15: (13, 7.5),  # P
        16: (13, 2.5)  # Q
    }
    # --- End of manual positions ---

    # --- Prepare all data for JSON export ---
    graph_data = {
        "nodes": [],
        "edges": [],
        "positions": {}
    }

    # Add node data
    for node_id, label in node_labels.items():
        graph_data["nodes"].append({"id": node_id, "label": label})

    # Add edge data
    for u, v, dist in edges_with_distances:
        graph_data["edges"].append({"source": u, "target": v, "distance": dist})

    # Add position data (convert tuples to lists for JSON)
    for node_id, coords in pos.items():
        if node_id in node_labels:
            graph_data["positions"][node_id] = list(coords)

            # --- Save the data to a JSON file in the root folder ---
    output_filename = '../graph_data.json'

    try:
        with open(output_filename, 'w') as f:
            json.dump(graph_data, f, indent=4)
        print(f"Graph data (with manual layout) successfully built and saved to '{output_filename}'")
    except Exception as e:
        print(f"Error saving graph data: {e}")


if __name__ == "__main__":
    build_and_save_graph()