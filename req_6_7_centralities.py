from matplotlib import pyplot as plt
import networkx as nx
import numpy as np
from scipy.stats import linregress


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
