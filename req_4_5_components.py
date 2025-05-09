import networkx as nx
import numpy as np


def network_components_analysis(G):

    print("\n \n Component gangs")
    top_compnent_count = 3

    components = sorted(nx.connected_components(G), key=len, reverse=True)
    if len(components) < 3:
        top_compnent_count = len(components)

    print("Number of components:", len(components))

    for i, component in enumerate(components[:top_compnent_count]):
        subgraph = G.subgraph(component)

        noun_count = sum(
            1 for node in subgraph if subgraph.nodes[node].get("type") == "noun"
        )
        adjective_count = sum(
            1 for node in subgraph if subgraph.nodes[node].get("type") == "adjective"
        )

        print(f"\nComponent {i + 1} (size: {subgraph.number_of_nodes()} nodes)")
        print(f" Nouns: {noun_count} ({noun_count / subgraph.number_of_nodes():.1%})")
        print(
            f"  Adjs: {adjective_count} ({adjective_count / subgraph.number_of_nodes():.1%})"
        )

        if adjective_count > 0:
            overall_ratio = noun_count / adjective_count
            print(f" noun-to-adj ratio: {overall_ratio:.2f}:1")

            if overall_ratio > 1.5:
                print("network is noun-dominated")
            elif overall_ratio < 0.67:
                print("network is adjective-dominated")
            else:
                print("network has a balanced noun-adjective distribution")
        else:
            print(" no adjs in this component")


def summarize_components(G):
    components = sorted(nx.connected_components(G), key=len, reverse=True)
    data = []

    for i, component in enumerate(components[:3]):
        subgraph = G.subgraph(component)
        nodes = subgraph.number_of_nodes()
        edges = subgraph.number_of_edges()
        size_str = f"{nodes}, {edges}"

        if nx.is_connected(subgraph):
            diameter = nx.diameter(subgraph)
            avg_path_length = nx.average_shortest_path_length(subgraph)
        else:
            diameter = "not connected"
            avg_path_length = "n/a"

        avg_degree_centrality = np.mean(list(nx.degree_centrality(subgraph).values()))
        data.append(
            (
                f"Component {i + 1}",
                size_str,
                "{:.2f}".format(avg_path_length),
                diameter,
                "{:.2f}".format(avg_degree_centrality),
            )
        )

    print("\nSummary of Top 3 Components:")
    print(
        "{:<15} {:<20} {:<25} {:<15} {:<20}".format(
            "Component",
            "Size (Nodes, Edges)",
            "Average Path Length",
            "Diameter",
            "Avg Degree Centrality",
        )
    )
    for row in data:
        print("{:<15} {:<20} {:<25} {:<15} {:<20}".format(*row))
