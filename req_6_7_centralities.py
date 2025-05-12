from matplotlib import pyplot as plt
import networkx as nx
import numpy as np
from scipy.stats import linregress


def analyze_power_law(centrality_values, color, name):
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
    else:
        fit = "Very weak evidence for power-law fit"

    plt.figure(figsize=(6, 5))
    plt.scatter(log_ranks, log_values, color=color, alpha=0.6)
    plt.plot(log_ranks, intercept + slope * log_ranks, "r-.", linewidth=2)
    plt.title(f"{name} Centrality (R²={r_squared:.2f})")
    plt.xlabel("Log(Rank)")
    plt.ylabel("Log(Centrality)")
    plt.tight_layout()
    plt.savefig(f"{name.lower()}_power_law_fit.png",
                dpi=300, bbox_inches="tight")
    plt.close()

    return r_squared, fit


def plot_centralities_power_law_fit(G):
    degree_centrality = nx.degree_centrality(G)
    closeness_centrality = nx.closeness_centrality(G)
    betweenness_centrality = nx.betweenness_centrality(G)

    # Save individual histograms
    plt.figure()
    plt.hist(degree_centrality.values(), bins=30,
             color="skyblue", edgecolor="black")
    plt.title("Degree Centrality Distribution")
    plt.xlabel("Degree Centrality")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig("degree_centrality_distribution.png",
                dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.hist(closeness_centrality.values(), bins=30,
             color="lightgreen", edgecolor="black")
    plt.title("Closeness Centrality Distribution")
    plt.xlabel("Closeness Centrality")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig("closeness_centrality_distribution.png",
                dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.hist(betweenness_centrality.values(), bins=30,
             color="lightcoral", edgecolor="black")
    plt.title("Betweenness Centrality Distribution")
    plt.xlabel("Betweenness Centrality")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig("betweenness_centrality_distribution.png",
                dpi=300, bbox_inches="tight")
    plt.close()

    # Save individual power-law fits
    degree_r2, degree_fit = analyze_power_law(
        degree_centrality, "skyblue", "Degree")
    closeness_r2, closeness_fit = analyze_power_law(
        closeness_centrality, "lightgreen", "Closeness")
    betweenness_r2, betweenness_fit = analyze_power_law(
        betweenness_centrality, "black", "Betweenness")

    print("\n\nPower Law Fit R-squared Values:")
    print(f"Degree Centrality: {degree_r2:.3f} --> {degree_fit}")
    print(f"Closeness Centrality: {closeness_r2:.3f} --> {closeness_fit}")
    print(
        f"Betweenness Centrality: {betweenness_r2:.3f} --> {betweenness_fit}\n\n")
