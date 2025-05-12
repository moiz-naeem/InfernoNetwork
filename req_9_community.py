import networkx as nx
from matplotlib import pyplot as plt
from networkx.algorithms import community as nx_community


def detect_communities_louvain(G: nx.Graph):

    print("\nNetwork Composition Before Community Detection:")
    total_nouns = sum(1 for node in G.nodes() if G.nodes[node]["type"] == "noun")
    total_adjs = sum(1 for node in G.nodes() if G.nodes[node]["type"] == "adjective")
    print(f"Total nodes: {len(G.nodes())} (Nouns: {total_nouns}, Adjs: {total_adjs})")
    print(f"Total edges: {len(G.edges())}")

    mixed_edges = sum(
        1 for u, v in G.edges() if G.nodes[u]["type"] != G.nodes[v]["type"]
    )
    print(f"Noun-Adjective edges: {mixed_edges} ({mixed_edges / len(G.edges()):.1%})")

    partition = nx_community.louvain_communities(G, resolution=0.8, seed=42)

    print("\nCommunity Analysis:")

    communities = sorted(partition, key=len, reverse=True)
    total_communities = len(communities)
    print("Total no. of communitites: ", total_communities)
    for i, comm in enumerate(communities[:total_communities]):
        nouns = [n for n in comm if G.nodes[n]["type"] == "noun"]
        adjs = [a for a in comm if G.nodes[a]["type"] == "adjective"]

        # for gephi
        for node in comm:
            G.nodes[node]["modularity_class"] = f"comm{i}"

        print(f"\nCommunity {i + 1} (Size: {len(comm)} nodes)")
        print(f"Nouns: {len(nouns)} ({len(nouns) / len(comm):.1%})")
        print(f"Adjs: {len(adjs)} ({len(adjs) / len(comm):.1%})")

        if len(nouns) == 0:
            print("  Pure adjective community")
        elif len(adjs) == 0:
            print("  Pure noun community")
        else:
            ratio = len(nouns) / len(adjs)
            if ratio > 1.5:
                print("  Strongly noun-dominated community")
            elif ratio < 0.67:
                print("  Strongly adjective-dominated community")
            else:
                print("  Balanced noun-adjective community")

    # write to gephi file
    nx.write_gexf(G, "communities.gexf")

    plt.figure(figsize=(14, 10))
    pos = nx.spring_layout(G, k=0.3, iterations=50, seed=42)

    nx.draw_networkx_edges(G, pos, alpha=0.1, edge_color="lightgray", width=0.5)

    mixed_edges = [
        (u, v) for u, v in G.edges() if G.nodes[u]["type"] != G.nodes[v]["type"]
    ]

    print("Mixed edges: ", len(mixed_edges))
    nx.draw_networkx_edges(
        G, pos, edgelist=mixed_edges, alpha=0.7, edge_color="yellow", width=1
    )

    for i, comm in enumerate(communities[:total_communities]):
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=list(comm),
            node_size=50,
            node_color=[plt.cm.tab20(i)] * len(comm),
            label=f"Community {i + 1}",
        )

    top_nodes = sorted(G.degree(), key=lambda x: x[1], reverse=True)[:15]
    nx.draw_networkx_labels(
        G,
        pos,
        labels={n: n for n, deg in top_nodes},
        font_size=9,
        bbox=dict(
            facecolor="white", alpha=0.6, edgecolor="black", boxstyle="round,pad=0.3"
        ),
    )

    plt.title(
        "Noun-Adjective Community Structure\n(yellow edges show noun-adjective connections)"
    )
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.axis("off")
    plt.savefig("noun_adj_comm.png")

    return partition
