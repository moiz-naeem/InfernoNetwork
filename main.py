import nltk
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from networkx.algorithms import community as nx_community
from nltk import WordNetLemmatizer
from scipy.stats import linregress
from scipy.optimize import curve_fit

nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("wordnet")
nltk.download("omw-1.4")
nltk.download("averaged_perceptron_tagger_eng")
nltk.download("maxent_ne_chunker")
nltk.download("words")
nltk.download("maxent_ne_chunker_tab")
nltk.download("stopwords")


def extract_using_tag(tagged_pos, tag, p_o_s):
    lemmatizer = WordNetLemmatizer()
    words = []

    noise = {
        "d",
        "”",
        "“",
        "’",
        "‘",
        "s",
        "st",
        "er",
        "n",
        "th",
        "d.",
        "t",
        "en",
        "'",
        "ne",
        "e",
        "ll",
        "re",
        "o",
        "i",  # misclassified as both noun and adjective
        "whence",  # not a noun nor adjective
        "forth",  # not a noun nor adjective
        "thence",  # not a noun nor adjective
        "hence",  # not a noun nor adjective
        "aught",  # not a noun nor adjective
        "rous",  # part of a adjective e.g. thund'rous
        "such",  # not an adjective
        "spake",  # archaic for spoke, a verb
        "mine",  # not noun nor adj
        "ken",  # pronoun
    }

    for word, pos in tagged_pos:
        word = word.lower()
        if pos != tag or word in noise:
            continue
        lemma = lemmatizer.lemmatize(word, pos=p_o_s)

        words.append(lemma)
    return words


def read_clean_text(word_list):
    for i in range(len(word_list)):
        current_word = word_list[i]
        # strip all possbile endings
        cleaned_word: str = current_word.strip("“”;,.?!;:").lower()

        if cleaned_word == "thou" or cleaned_word == "thee":
            cleaned_word = "you"
        elif cleaned_word == "thine":
            cleaned_word = "yours"
        elif cleaned_word == "thy":
            cleaned_word = "your"
        elif cleaned_word == "ye":
            cleaned_word = "you"
        elif cleaned_word == "e’en":
            cleaned_word = "even"
        elif cleaned_word == "e’er":
            cleaned_word = "ever"
        elif cleaned_word == "o’er":
            cleaned_word = "over"
        elif cleaned_word == "heav’n":
            cleaned_word = "heaven"
        elif cleaned_word == "oft":
            cleaned_word = "often"
        elif cleaned_word == "hath":
            cleaned_word = "have"
        elif cleaned_word == "lo":
            cleaned_word = "look"
        elif cleaned_word == "doth":
            cleaned_word = "do"
        elif cleaned_word == "’gainst":
            cleaned_word = "against"
        elif cleaned_word.endswith("’d"):
            start = cleaned_word.split("’")[0]
            cleaned_word = start + "ed"

        # check case of the word
        if current_word[0].isupper():
            cleaned_word = cleaned_word.capitalize()

        # Add the original end to the word
        if "," in current_word:
            word_list[i] = cleaned_word + ","

        elif "!" in current_word:
            word_list[i] = cleaned_word + "!"

        elif "." in current_word:
            word_list[i] = cleaned_word + "."

        elif "?" in current_word:
            word_list[i] = cleaned_word + "?"

        elif ";" in current_word:
            word_list[i] = cleaned_word + ";"

        elif ":" in current_word:
            word_list[i] = cleaned_word + ":"

        else:
            word_list[i] = cleaned_word

    read = " ".join(word_list)

    return read


def process_text(file_path):
    lines = []
    # clean each line separately, lines are needed later
    with open(file_path, "r", encoding="utf-8") as text:
        for line in text:
            if line == "":
                continue
            line = read_clean_text(line.split())
            lines.append(line)

    # join lines to tokenize
    read = "\n".join(lines)

    with open("cleaned.txt", "w") as new:
        new.write(read)
        new.close()

    all_word_token = nltk.tokenize.word_tokenize(read)

    words_token_tagged_by_pos = nltk.pos_tag(all_word_token)

    adjectives = extract_using_tag(words_token_tagged_by_pos, "JJ", "a")
    nouns = extract_using_tag(words_token_tagged_by_pos, "NN", "n")

    noun_frequencis = nltk.FreqDist(nouns)
    adj_frequencies = nltk.FreqDist(adjectives)

    top_100_nouns = [word for word, count in noun_frequencis.most_common(100)]
    top_100_adjs = [word for word, count in adj_frequencies.most_common(100)]

    return (lines, top_100_nouns, top_100_adjs)


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
                        # G[word1][word2]["weight"] += 1
                    else:
                        # G.add_edge(word1, word2, weight=1)
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


def save_adjacency_matrix(G, filename):
    nodes = sorted(G.nodes())
    adj_matrix = nx.to_numpy_array(G, nodelist=nodes)
    np.savetxt(filename, adj_matrix, fmt="%d")


def adj_noun_graph_properties_check(G):

    print("\nSkibidi Sammaakko")

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

    print(f"\nAdditional yap:")
    print(f"Number of nodes: {G.number_of_nodes()}")
    print(f"Number of edges: {G.number_of_edges()}")
    print(f"Density: {nx.density(G):.4f}")


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
        size = (nodes, edges)
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


def plot_centralities_power_law_fit(G):

    degree_centrality = nx.degree_centrality(G)
    closeness_centrality = nx.closeness_centrality(G)
    betweenness_centrality = nx.betweenness_centrality(G)

    plt.figure(figsize=(18, 5))
    plt.subplot(1, 3, 1)
    plt.hist(degree_centrality.values(), bins=30, color="skyblue", edgecolor="black")
    plt.title("Dgree Centrality Distribution")
    plt.xlabel("Degree Centrality")
    plt.ylabel("Frequency")

    plt.subplot(1, 3, 2)
    plt.hist(
        closeness_centrality.values(), bins=30, color="lightgreen", edgecolor="black"
    )
    plt.title("Closeness Centrality Distribution")
    plt.xlabel("Closeness Centrality")
    plt.ylabel("Frequency")

    plt.subplot(1, 3, 3)
    plt.hist(
        betweenness_centrality.values(), bins=30, color="lightcoral", edgecolor="black"
    )
    plt.title("Betweeness Centrality Distribution")
    plt.xlabel("Betweenness Centrality")
    plt.ylabel("Frequency")

    plt.tight_layout()
    plt.savefig("centralitites.png", dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(18, 5))

    plt.subplot(1, 3, 1)
    degree_r2, fit = analyze_power_law(
        degree_centrality, "skyblue", "Degree", "Very weak evidence for power-law fit"
    )

    plt.subplot(1, 3, 2)
    closeness_r2, fit = analyze_power_law(
        closeness_centrality,
        "lightgreen",
        "Closeness",
        "Very weak evidence for power-law fit",
    )

    plt.subplot(1, 3, 3)
    betwenness_r2, fit = analyze_power_law(
        betweenness_centrality,
        "black",
        "Betweenness",
        "Very weak evidence for power-law fit",
    )

    plt.tight_layout()
    plt.savefig("power_law_fits.png", dpi=300, bbox_inches="tight")

    print("\n\nPower Law Fit R-squared Values:")
    print(f"Degree Centrality: {degree_r2:.3f} --> {fit}")
    print(f"Closeness Centrality: {closeness_r2:.3f}--> {fit}")
    print(f"Betweenness Centrality: {betwenness_r2:.3f}--> {fit} \n\n")


def analyze_power_law(centrality_values, color, name, fit):

    values = np.array(list(centrality_values.values()))
    values = values[values > 0]

    log_values = np.log10(values)
    log_ranks = np.log10(np.arange(1, len(values) + 1))

    slope, intercept, r_value, _, _ = linregress(log_ranks, log_values)
    r_squared = r_value**2
    if r_squared > 0.7 and r_squared < 0.85:
        fit = "Moderate evidence for power-law fit"
    elif r_squared > 0.85:
        fit = "Very strong evidence for power-law fit"

    plt.scatter(log_ranks, log_values, color=color, alpha=0.6)
    plt.plot(log_ranks, intercept + slope * log_ranks, "r-.", linewidth=2)
    plt.title(f"{name} Centrality (R²={r_squared:.2f})")
    plt.xlabel("Log(Rank)")
    plt.ylabel("Log(Centrality)")

    return r_squared, fit


def analyze_clustering_coefficient_distribution(G):

    clustering_coeffs = list(nx.clustering(G).values())

    plt.figure(figsize=(12, 10))

    plt.subplot(2, 2, 1)
    counts, bins, _ = plt.hist(
        clustering_coeffs, bins=10, color="skyblue", edgecolor="black"
    )
    plt.title("Clustering Coefficient Distribution (10 bins)")
    plt.xlabel("clustering Coefficient")
    plt.ylabel("frequency")

    bin_centers = 0.5 * (bins[1:] + bins[:-1])
    nonzero_indices = counts > 0
    x_nonzero = bin_centers[nonzero_indices]
    y_nonzero = counts[nonzero_indices]

    plt.subplot(2, 2, 2)
    log_x = np.log10(x_nonzero)
    log_y = np.log10(y_nonzero)
    plt.scatter(log_x, log_y, color="blue", alpha=0.7)

    slope, intercept, r_value, _, _ = linregress(log_x, log_y)
    power_law_r_squared = r_value**2
    power_law_fit = intercept + slope * log_x
    plt.plot(log_x, power_law_fit, "r-", linewidth=2)
    plt.title(f"Power Law Fit (log-log) - R² = {power_law_r_squared:.3f}")
    plt.xlabel("Log(Clustering Coefficient)")
    plt.ylabel("Log(Frequency)")

    def truncated_power_law(x, a, b, c):
        return a * (x**b) * np.exp(-c * x)

    plt.subplot(2, 2, 3)
    p0 = [1.0, slope, 0.1]
    params, covariance = curve_fit(
        truncated_power_law, x_nonzero, y_nonzero, p0=p0, maxfev=10000
    )
    a_fit, b_fit, c_fit = params

    y_fit = truncated_power_law(x_nonzero, *params)
    residuals = y_nonzero - y_fit
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y_nonzero - np.mean(y_nonzero)) ** 2)
    trunc_power_law_r_squared = 1 - (ss_res / ss_tot)

    plt.scatter(x_nonzero, y_nonzero, color="green", alpha=0.7)
    x_fit = np.linspace(min(x_nonzero), max(x_nonzero), 100)
    y_fit = truncated_power_law(x_fit, *params)
    plt.plot(x_fit, y_fit, "r-", linewidth=2)
    plt.title(f"Truncated Power Law Fit - R² = {trunc_power_law_r_squared:.3f}")
    plt.xlabel("Clustering Coefficient")
    plt.ylabel("Frequency")

    plt.subplot(2, 2, 4)
    plt.scatter(log_x, log_y, color="purple", alpha=0.7, label="Data")
    plt.plot(
        log_x,
        power_law_fit,
        "r-",
        linewidth=2,
        label=f"Power Law (R²={power_law_r_squared:.3f})",
    )

    log_y_trunc = np.log10(truncated_power_law(10**log_x, *params))
    plt.plot(
        log_x,
        log_y_trunc,
        "g--",
        linewidth=2,
        label=f"Trunc Power Law (R²={trunc_power_law_r_squared:.3f})",
    )
    plt.title("Comparison of Fits (log-log)")
    plt.xlabel("Log(Clustering Coefficient)")
    plt.ylabel("Log(Frequency)")
    plt.legend()

    if trunc_power_law_r_squared > power_law_r_squared:
        better_model = "Truncated Power Law"
        better_r2 = trunc_power_law_r_squared
    else:
        better_model = "Pure Power Law"
        better_r2 = power_law_r_squared

    plt.tight_layout()
    plt.savefig("clustering_coeff_distribution.png", dpi=300, bbox_inches="tight")

    print("\nClustering Coefficient Distribution Analysis:")
    print(f"Power Law Fit R² = {power_law_r_squared:.3f}, exponent = {slope:.3f}")
    print(f"Truncated Power Law Fit R² = {trunc_power_law_r_squared:.3f}")
    print(f"Parameters: a = {a_fit:.3f}, b = {b_fit:.3f}, c = {c_fit:.3f}")
    print(f"Better fit: {better_model} (R² = {better_r2:.3f})")

    if better_r2 > 0.85:
        fit_quality = "Very strong evidence"
    elif better_r2 > 0.7:
        fit_quality = "Moderate evidence"
    else:
        fit_quality = "Weak evidence"
    print(f"{fit_quality} for {better_model.lower()} fit")


def plot_clustering_coefficient_distribution(G):
    clustering_coefficients = list(nx.clustering(G).values())
    hist, bins = np.histogram(clustering_coefficients, bins=10, density=True)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    plt.figure(figsize=(8, 6))
    plt.bar(
        bin_centers,
        hist,
        width=(bins[1] - bins[0]),
        color="skyblue",
        edgecolor="black",
        alpha=0.7,
    )
    plt.xlabel("Clustering Coefficient")
    plt.ylabel("Probability Density")
    plt.title("Clustering Coefficient Distribution")

    return bin_centers, hist


def fit_power_law(bin_centers, hist):
    mask = (bin_centers > 0) & (hist > 0)
    log_x = np.log10(bin_centers[mask])
    log_y = np.log10(hist[mask])

    if len(log_x) < 2:
        print("Not enough data points for power law fit")
        return -np.inf

    slope, intercept, r_value, _, _ = linregress(log_x, log_y)
    r_squared = r_value**2

    plt.figure(figsize=(8, 6))
    plt.scatter(log_x, log_y, color="blue", label="Data", alpha=0.7)
    plt.plot(
        log_x,
        slope * log_x + intercept,
        color="red",
        label=f"Fit: $R^2 = {r_squared:.3f}$",
    )
    plt.xlabel("Log10(Bin Centers)")
    plt.ylabel("Log10(Frequency)")
    plt.title("Power Law Fit")
    plt.legend()

    print(
        f"Power-law fit: slope={slope:.3f}, intercept={intercept:.3f}, R-squared={r_squared:.3f}"
    )
    return r_squared


def fit_truncated_power_law(bin_centers, histogram_values):
    def truncated_power_law(clustering_coeff, exponent, decay_rate):
        return (clustering_coeff**exponent) * np.exp(-decay_rate * clustering_coeff)

    valid_data_mask = (bin_centers > 0) & (histogram_values > 0)
    clustering_coeffs = bin_centers[valid_data_mask]
    frequencies = histogram_values[valid_data_mask]

    if len(clustering_coeffs) < 2:
        print("Insufficient non-zero data points for truncated power law fitting")
        return -np.inf

    try:
        optimal_params, _ = curve_fit(
            truncated_power_law, clustering_coeffs, frequencies, p0=[-2, 1], maxfev=5000
        )
        exponent, decay_rate = optimal_params

        predicted_values = truncated_power_law(clustering_coeffs, *optimal_params)
        residual_errors = frequencies - predicted_values
        sum_squared_residuals = np.sum(residual_errors**2)
        total_variance = np.sum((frequencies - np.mean(frequencies)) ** 2)
        r_squared = 1 - (sum_squared_residuals / total_variance)

        plt.figure(figsize=(8, 6))
        plt.scatter(
            clustering_coeffs,
            frequencies,
            color="blue",
            label="Observed Data",
            alpha=0.7,
        )

        fit_x_values = np.linspace(min(clustering_coeffs), max(clustering_coeffs), 100)
        plt.plot(
            fit_x_values,
            truncated_power_law(fit_x_values, *optimal_params),
            color="green",
            label=f"Fit (R²={r_squared:.3f})\nExponent={exponent:.3f}, Decay={decay_rate:.3f}",
        )

        plt.xlabel("Clustering Coefficient")
        plt.ylabel("Probability Density")
        plt.title("Truncated Power Law Fit to Clustering Distribution")
        plt.legend()

        print(f"Truncated power law fit results:")
        print(f"exponent (a): {exponent:.3f}")
        print(f"decay rate (b): {decay_rate:.3f}")
        print(f"R²: {r_squared:.3f}")

        return r_squared

    except Exception as error:
        print(f"Failed to fit truncated power law: {error}")
        return -np.inf


def compare_fits(power_r2, truncated_r2):
    print("\nComparison of fits:")
    print(f"Power law R-squared: {power_r2:.3f}")
    print(f"Truncated power law R-squared: {truncated_r2:.3f}")

    if power_r2 > 0.85 and power_r2 > truncated_r2:
        print("Strong evidence for a power-law distribution.")
    elif truncated_r2 > 0.85 and truncated_r2 > power_r2:
        print("Strong evidence for a truncated power-law distribution.")
    elif power_r2 > truncated_r2:
        print("Better fit to power law, but not strong evidence.")
    elif truncated_r2 > power_r2:
        print("Better fit to truncated power law, but not strong evidence.")
    else:
        print("Inconclusive results.")


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


def relation_evolution(file_path, top_nouns, top_adjs):
    lines = []

    noun_noun = []
    noun_adj = []
    adj_adj = []

    with open(file_path, "r", encoding="utf-8") as text:
        for line in text:
            line = read_clean_text(line.split())
            if line == "":
                continue
            lines.append(line)

    for i in range(len(lines)):
        adj_this_line = 0
        noun_this_line = 0

        current_line = lines[i]
        all_word_token = nltk.tokenize.word_tokenize(current_line)

        words_token_tagged_by_pos = nltk.pos_tag(all_word_token)

        adjectives = extract_using_tag(words_token_tagged_by_pos, "JJ", "a")
        nouns = extract_using_tag(words_token_tagged_by_pos, "NN", "n")

        for adj in adjectives:
            if adj in top_adjs:
                adj_this_line += 1

        for noun in nouns:
            if noun in top_nouns:
                noun_this_line += 1

        # relationships on this line
        noun_noun.append(max(0, noun_this_line * (noun_this_line - 1) / 2))
        noun_adj.append(noun_this_line * adj_this_line)
        adj_adj.append(max(0, adj_this_line * (adj_this_line - 1) / 2))

        if i != 0:
            noun_noun[i] += noun_noun[(i - 1)]
            noun_adj[i] += noun_adj[(i - 1)]
            adj_adj[i] += adj_adj[(i - 1)]

    # Plot the relationships
    plt.figure(figsize=(18, 15))

    # Plot noun-noun relationships
    plt.subplot(3, 1, 1)
    plt.plot(range(len(lines)), noun_noun, "r-", linewidth=1.5)
    plt.title("Noun-Noun Relationships", fontsize=16)
    plt.ylabel("Number of Relationships")
    plt.grid(True, alpha=0.3)

    # Plot noun-adj relationships
    plt.subplot(3, 1, 2)
    plt.plot(range(len(lines)), noun_adj, "g-", linewidth=1.5)
    plt.title("Noun-Adjective Relationships", fontsize=16)
    plt.ylabel("Number of Relationships")
    plt.grid(True, alpha=0.3)

    # Plot adj-adj relationships
    plt.subplot(3, 1, 3)
    plt.plot(range(len(lines)), adj_adj, "b-", linewidth=1.5)
    plt.title("Adjective-Adjective Relationships", fontsize=16)
    plt.xlabel("Line Number")
    plt.ylabel("Number of Relationships")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("relationship_evolution.png", dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    file_path = "divine_comedy.txt"

    lines, top_nouns, top_adjs = process_text(file_path)

    with open("top_nouns.txt", "w", encoding="utf-8") as f:
        f.write(",\n".join(top_nouns))
    with open("top_adjectives.txt", "w", encoding="utf-8") as f:
        f.write(",\n".join(top_adjs))

    network = graph_for_adj_noun_occurrence(lines, top_nouns, top_adjs)

    save_adjacency_matrix(network, "adjacency_matrix.txt")

    visualize_network(network)
    adj_noun_graph_properties_check(network)
    network_components_analysis(network)
    summarize_components(network)
    plot_centralities_power_law_fit(network)

    analyze_clustering_coefficient_distribution(network)

    # do these do anything?

    bin_centers, histogram_vals = plot_clustering_coefficient_distribution(network)
    power_r2 = fit_power_law(bin_centers, histogram_vals)
    truncated_r2 = fit_truncated_power_law(bin_centers, histogram_vals)
    compare_fits(power_r2, truncated_r2)

    partition = detect_communities_louvain(network)

    # relationship evolution
    relation_evolution(file_path, top_nouns, top_adjs)
