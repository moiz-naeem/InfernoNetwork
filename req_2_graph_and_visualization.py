import nltk
import networkx as nx
from matplotlib import pyplot as plt
import numpy as np


def save_adjacency_matrix(G, filename):
    nodes = sorted(G.nodes())
    adj_matrix = nx.to_numpy_array(G, nodelist=nodes)
    np.savetxt(filename, adj_matrix, fmt="%d")


def graph_for_adj_noun_occurrence(lines, top_nouns, top_adjs):
    G = nx.Graph()

    overlaps = 0

    for noun in top_nouns:
        G.add_node(f"noun_{noun}", type="noun")
        if noun in top_adjs:
            overlaps += 1

    for adj in top_adjs:
        G.add_node(f"adj_{adj}", type="adjective")

    print(f"Overlap between adj and nouns: {overlaps}")

    for i in range(len(lines)):
        words = nltk.tokenize.word_tokenize(lines[i])

        # there are duplicates in both classes, need to change the name of the node
        line_nouns = [f"noun_{word}" for word in set(words) & set(top_nouns)]
        line_adjs = [f"adj_{word}" for word in set(words) & set(top_adjs)]
        line_words = line_adjs + line_nouns

        for word1 in line_words:
            for word2 in line_words:
                if word1 != word2:
                    if G.has_edge(word1, word2):
                        continue
                    else:
                        G.add_edge(word1, word2)

    # write to gephi file
    nx.write_gexf(G, "noun_adj.gexf")
    return G


def visualize_network(G: nx.Graph):
    pos = nx.spring_layout(G, k=0.5, iterations=50)

    nouns = [node for node in G.nodes() if G.nodes[node]["type"] == "noun"]
    adjs = [node for node in G.nodes() if G.nodes[node]["type"] == "adjective"]

    plt.figure(figsize=(20, 15))
    nx.draw_networkx_nodes(
        G, pos, nodelist=nouns, node_color="red", node_size=500, alpha=0.8
    )
    nx.draw_networkx_nodes(
        G, pos, nodelist=adjs, node_color="green", node_size=500, alpha=0.8
    )
    # edges = [(u, v) for u, v, d in G.edges(data=True) if d["weight"] > 1]
    edges = [(u, v) for u, v, d in G.edges(data=True)]
    nx.draw_networkx_edges(G, pos, edgelist=edges, width=1.0, alpha=0.2)

    degrees = dict(G.degree())
    top_nodes = {node: node for node, deg in degrees.items()}
    nx.draw_networkx_labels(G, pos, labels=top_nodes, font_size=8, font_weight="bold")

    plt.title("Noun-Adjective Co-occurrence Network", fontsize=16)
    plt.axis("off")
    plt.savefig("noun_adj_network.png", dpi=300, bbox_inches="tight")
    plt.close()
