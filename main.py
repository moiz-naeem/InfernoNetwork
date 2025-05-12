from req_11_steiner import find_steiner_tree, reconstruct_weighted
from req_1_text_processing import process_text
from req_2_graph_and_visualization import (
    graph_for_adj_noun_occurrence,
    save_adjacency_matrix,
    visualize_network,
)
from req_3_basic_SNA import adj_noun_graph_properties_check
from req_4_5_components import network_components_analysis, summarize_components
from req_6_7_centralities import plot_centralities_power_law_fit
from req_8_clustering_coeff import (
    analyze_clustering_coefficient_distribution,
    compare_fits,
    fit_power_law,
    fit_truncated_power_law,
    plot_clustering_coefficient_distribution,
)
from req_9_community import detect_communities_louvain
from req_10_relationships import relation_evolution


if __name__ == "__main__":
    # requirement 1
    print("requirement 1")
    file_path = "divine_comedy.txt"
    lines, top_nouns, top_adjs = process_text(file_path)

    with open("top_nouns.txt", "w", encoding="utf-8") as f:
        f.write(",\n".join(top_nouns))
    with open("top_adjectives.txt", "w", encoding="utf-8") as f:
        f.write(",\n".join(top_adjs))

    # requirement 2
    print("requirement 2")
    network = graph_for_adj_noun_occurrence(lines, top_nouns, top_adjs)
    save_adjacency_matrix(network, "adjacency_matrix.txt")
    visualize_network(network)

    # requirement 3
    print("requirement 3")
    adj_noun_graph_properties_check(network)

    # requirement 4
    print("requirement 4")
    network_components_analysis(network)

    # requirement 5
    print("requirement 5")
    summarize_components(network)

    # requirement 6
    print("requirement 6")
    plot_centralities_power_law_fit(network)

    # requirement 7
    print("requirement 7")
    analyze_clustering_coefficient_distribution(network)

    # requirement 8
    print("requirement 8")
    bin_centers, histogram_vals = plot_clustering_coefficient_distribution(
        network)
    # Do these do anything?
    power_r2 = fit_power_law(bin_centers, histogram_vals)
    truncated_r2 = fit_truncated_power_law(bin_centers, histogram_vals)
    compare_fits(power_r2, truncated_r2)

    # requirement 9
    print("requirement 9")
    partition = detect_communities_louvain(network)

    # requirement 10
    print("requirement 10")
    relation_evolution(file_path, top_nouns, top_adjs)

    # requirement 11
    print("requirement 11")
    reconst_network = reconstruct_weighted(lines, top_nouns, top_adjs)
    find_steiner_tree(reconst_network)
