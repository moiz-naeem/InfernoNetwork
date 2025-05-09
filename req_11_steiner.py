import random
from matplotlib import pyplot as plt
import nltk
import networkx as nx

from req_2_graph_and_visualization import save_adjacency_matrix, visualize_network
from req_3_basic_SNA import adj_noun_graph_properties_check


def reconstruct_weighted(lines, top_nouns, top_adjs):
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
                        G[word1][word2]["weight"] += 1
                    else:
                        G.add_edge(word1, word2, weight=1)

    return G


# modified slightly from https://github.com/GiliardGodoi/xsteiner/blob/main/xsteiner/main.py
def random_prim_st(G: nx.Graph, terminals):
    """
    Random Steiner Tree based on Prim's algorithms
    """
    P = nx.Graph()
    nodes = set(G.nodes)
    terminals = set(terminals)
    done = set()
    candidates = set()

    # v_star = random.choice(list(nodes))
    v_star = list(terminals)[0]
    for u in G.adj[v_star]:
        candidates.add((v_star, u))

    nodes.remove(v_star)
    terminals.discard(v_star)
    done.add(v_star)

    n_count = 0
    n_edges = G.number_of_edges()

    while terminals:
        if n_count > n_edges + 2:
            raise RuntimeError(
                f"An unexpected error occurred: {n_count} > {n_edges + 2}"
            )
        edge = random.choice(list(candidates))
        v, u = edge

        if (v in done) and (u in nodes):
            P.add_edge(u, v)
            nodes.remove(u)
            terminals.discard(u)
            done.add(u)
            for t in G.adj[u]:
                if t not in done:
                    candidates.add((u, t))
        # elif (u in done) and (v in nodes):
        #     print('is it necessary?')
        #     P.add_edge(v, u)
        #     nodes.remove(v)
        #     terminals.discard(v)
        #     done.add(v)
        #     for t in G.adj[v]:
        #         if t not in done:
        #             candidates.add((u, t))
        # else:
        #     pass # just ignore the edge
        candidates.remove(edge)

    return P


def find_steiner_tree(G: nx.Graph):
    save_adjacency_matrix(G, "recon_adjacency_matrix.txt")

    # 10 highest degree nodes
    sorted10 = dict(
        sorted(
            dict(nx.degree(G)).items(),
            key=lambda item: item[1],
            reverse=True,
        )[:10]
    )
    terminals = list(sorted10.keys())
    print(terminals)

    steiner = random_prim_st(G, terminals)

    print("________________________________")
    print("Steiner Graph")
    adj_noun_graph_properties_check(steiner)

    pos = nx.spring_layout(steiner)

    plt.figure(figsize=(12, 12))

    # Create node color map - red for terminals, blue for others
    node_colors = ["red" if node in terminals else "blue" for node in steiner.nodes()]

    # Draw the network with colored nodes
    nx.draw(steiner, pos, node_color=node_colors, node_size=100)
    nx.draw_networkx_labels(steiner, pos, font_size=8, font_weight="bold")

    plt.title("Steiner tree, 10 highest degree nodes as terminals", fontsize=16)
    plt.axis("off")
    plt.savefig("steiner_tree.png", dpi=300, bbox_inches="tight")
    plt.close()
