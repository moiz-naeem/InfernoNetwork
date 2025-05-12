import networkx as nx


def adj_noun_graph_properties_check(G):
    bipartite_checker = nx.is_bipartite(G)
    print(f"Bipartite: {bipartite_checker}")

    if nx.is_connected(G):
        diameter = nx.diameter(G)
        avg_path_length = nx.average_shortest_path_length(G)
        all_pairs = dict(nx.all_pairs_shortest_path_length(G))
        all_lengths = [
            length
            for start in all_pairs
            for end, length in all_pairs[start].items()
            if start != end
        ]
        print(f"Diameter: {diameter}")
        print(f"Average path length: {avg_path_length:.2f}")
        print(f"shortest path length: {min(all_lengths)}")
        print(f"Longest path length: {max(all_lengths)}")

    else:
        print("not connected hence infinite diameter")
        print("not connected hence cant calculate average path length")
        print("not connected hence cant calculate shortest/longest path length")

    clustering_coefficent = nx.clustering(G)
    avg_clustering = nx.average_clustering(G)
    print(f"Average clustering coefficient: {avg_clustering:.2f}")
    print(f"Minimum clustering coefficient: {min(clustering_coefficent.values()):.2f}")
    print(f"Maximum clustering coefficient: {max(clustering_coefficent.values()):.2f}")

    degrees = dict(G.degree())
    avg_degree = sum(degrees.values()) / len(degrees)
    print(f"Average degree: {avg_degree:.2f}")
    print(f"Minimum degree: {min(degrees.values())}")
    print(f"Maximum degree: {max(degrees.values())}")

    print(f"Number of nodes: {G.number_of_nodes()}")
    print(f"Number of edges: {G.number_of_edges()}")
    print(f"Density: {nx.density(G):.4f}")
