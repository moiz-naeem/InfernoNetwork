import nltk
import networkx as nx;
import numpy as np;
import matplotlib.pyplot as plt;
from nltk import extract, WordNetLemmatizer
from nltk.corpus import stopwords

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download("maxent_ne_chunker")
nltk.download("words")
nltk.download('maxent_ne_chunker_tab')
nltk.download('stopwords')


def extract_using_tag(tagged_pos, tag, p_o_s):
    lammatizer = WordNetLemmatizer();
    words = []

    noise = {"d", "”" , "“" , "’" , "s" , "st" , "er" , "n" , "th" , "d." , "t", "en", "'", "ne", "e", "ll", "re", "o"}
    for word, pos in tagged_pos:
        word = word.lower()
        if pos != tag or word in noise:
            continue
        lemma = lammatizer.lemmatize(word, pos=p_o_s)

        words.append(lemma)
    return words

def process_text(file_path):
    with open(file_path, 'r', encoding='utf-8') as divine_comedy:

        read = divine_comedy.read().strip()
        all_word_token = nltk.tokenize.word_tokenize(read)
        all_sentence_token = nltk.tokenize.sent_tokenize(read)
        # print(all_word_token)
        words_token_tagged_by_pos = nltk.pos_tag(all_word_token)
        print(
            words_token_tagged_by_pos
        )
        adjectives = extract_using_tag(words_token_tagged_by_pos, "JJ", "a")
        nouns = extract_using_tag(words_token_tagged_by_pos, "NN", "n")
        noun_frequencis = nltk.FreqDist(nouns)
        adj_frequencies = nltk.FreqDist(adjectives)
        top_100_nouns = [word for word, count in noun_frequencis.most_common(100)]
        top_100_adjs = [word for word, count in adj_frequencies.most_common(100)]
        print("\n top 100 most frequent nouns: ")
        for word in top_100_nouns:
            print(f"{word}")
        print("\n top 100 most frequent adjective: ")
        for word in top_100_adjs:
            print(f"{word}")
        print("\nTotal adjectives:", len(adjectives))

    return all_sentence_token, top_100_nouns, top_100_adjs

def graph_for_adj_noun_occurrence(sentences, noun_nodes, adj_nodes):
    G = nx.Graph()
    for noun in noun_nodes:
        G.add_node(noun, type="noun" )
    for adj in adj_nodes:
        G.add_node(adj, type="adjectve" )

    for sentence in sentences:
        words  = nltk.tokenize.word_tokenize(sentence.lower())
        sentence_nouns = set(words) & set(noun_nodes)
        sentence_adjs = set(words) & set(adj_nodes)

        for noun in sentence_nouns:
            for adj in sentence_adjs:
                if G.has_edge(noun, adj):
                    G[noun][adj]["weight"] += 1;
                else:
                    G.add_edge(noun, adj, weight=1);

    return G;


def visualize_network(G):
    pos = nx.spring_layout(G, k=0.5, iterations=50);

    nouns = [node for node in G.nodes() if G.nodes[node]['type'] == 'noun']
    adjs = [node for node in G.nodes() if G.nodes[node]['type'] == 'adjective']

    plt.figure(figsize=(20, 15))
    nx.draw_networkx_nodes(G, pos, nodelist=nouns, node_color='red', node_size=500, alpha=0.8)
    nx.draw_networkx_nodes(G, pos, nodelist=adjs, node_color='green', node_size=500, alpha=0.8)
    edges = [(u, v) for u, v, d in G.edges(data=True) if d['weight'] > 1]
    nx.draw_networkx_edges(G, pos, edgelist=edges, width=1.0, alpha=0.2)

    degrees = dict(G.degree())
    top_nodes = {node: node for node, deg in degrees.items() if deg > 2}
    nx.draw_networkx_labels(G, pos, labels=top_nodes, font_size=10, font_weight='bold')


    plt.title("Noun-Adjective Co-occurrence Network", fontsize=16)
    plt.axis('off')
    plt.savefig('noun_adj_network.png', dpi=300, bbox_inches='tight')

    plt.show()

def save_adjacency_matrix(G, filename):

    nodes = sorted(G.nodes())
    adj_matrix = nx.to_numpy_array(G, nodelist=nodes)
    np.savetxt(filename, adj_matrix, fmt='%d')


def adj_noun_graph_properties_check(G):

    print("\nSkibidi Sammaakko")

    bipartite_checker = nx.is_bipartite(G)
    print(f"bipartite: {bipartite_checker}")

    if nx.is_connected(G):
        diameter = nx.diameter(G)
        avg_path_length = nx.average_shortest_path_length(G)
        all_pairs = dict(nx.all_pairs_shortest_path_length(G))
        all_lengths = [length for start in all_pairs for end, length in all_pairs[start].items() if start != end]
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
    print(f" Average clustering coefficient: {avg_clustering:.2f}")
    print(f"Minimum clustering coefficient: {min(clustering_coefficent.values()):.2f}")
    print(f" Maximum clustering coefficient: {max(clustering_coefficent.values()):.2f}")

    degrees = dict(G.degree())
    avg_degree = sum(degrees.values()) / len(degrees)
    print(f"Average degree: {avg_degree:.2f}")
    print(f"Minimum degree: {min(degrees.values())}")
    print(f"Maximum degree: {max(degrees.values())}")


    print(f"\nAdditional yap:")
    print(f" Number of nodes: {G.number_of_nodes()}")
    print(f"number of edges: {G.number_of_edges()}")
    print(f"density: {nx.density(G):.4f}")

def network_components_analysis(G):


    print("\n \n component gangs")
    top_compnent_count = 3;

    components = sorted(nx.connected_components(G), key=len, reverse=True)
    if(len(components) < 3):
        top_compnent_count = len(components)

    print("Number of components:", len(components))

    for i, component in enumerate(components[:top_compnent_count]):
        subgraph = G.subgraph(component)

        noun_count = sum(1 for node in subgraph if subgraph.nodes[node].get('type') == 'noun')
        adjective_count = sum(1 for node in subgraph if subgraph.nodes[node].get('type') == 'adjectve')


        print(f"\ncoomponent {i + 1} (size: {subgraph.number_of_nodes()} nodes)")
        print(f" mouns: {noun_count} ({noun_count / subgraph.number_of_nodes():.1%})")
        print(f"  adjs: {adjective_count} ({adjective_count / subgraph.number_of_nodes():.1%})")


        if adjective_count > 0:
            overall_ratio = noun_count/adjective_count;
            print(f" noun-to-adj ratio: {overall_ratio:.2f}:1")

            if overall_ratio > 1.5:
                print("network is noun-dominated")
            elif overall_ratio < 0.67:
                print("network is adjective-dominated")
            else:
                print("network has a balanced noun-adjective distribution")
        else:
            print(" no adjs in this component")


if __name__ == "__main__":
    file_path = 'divine_comedy.txt'


    sentences, top_nouns, top_adjs = process_text(file_path)

    with open('top_nouns.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(top_nouns))
    with open('top_adjectives.txt', 'w',  encoding='utf-8') as f:
        f.write('\n'.join(top_adjs))

    network = graph_for_adj_noun_occurrence(sentences, top_nouns, top_adjs)

    save_adjacency_matrix(network, 'adjacency_matrix.txt')

    visualize_network(network);
    adj_noun_graph_properties_check(network);
    network_components_analysis(network);
