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



if __name__ == "__main__":
    file_path = 'divine_comedy.txt'


    sentences, top_nouns, top_adjs = process_text(file_path)

    with open('top_nouns.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(top_nouns))
    with open('top_adjectives.txt', 'w',  encoding='utf-8') as f:
        f.write('\n'.join(top_adjs))

    network = graph_for_adj_noun_occurrence(sentences, top_nouns, top_adjs)

    save_adjacency_matrix(network, 'adjacency_matrix.txt')

    visualize_network(network)
