import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def compare_fits(power_r2, truncated_r2):
    print("\nModel Fit Comparison:")
    print(f"Power-law R²: {power_r2:.4f}")
    print(f"Truncated Power-law R²: {truncated_r2:.4f}")

    if power_r2 > truncated_r2:
        print("Power-law fit is better.")
    elif truncated_r2 > power_r2:
        print("Truncated power-law fit is better.")
    else:
        print("Both fits are equally good.")


def analyze_clustering_coefficient_distribution(G):
    clustering_coeffs = list(nx.clustering(G).values())
    counts, bins, _ = plt.hist(
        clustering_coeffs, bins=10, color="skyblue", edgecolor="black"
    )
    plt.title("Clustering Coefficient Distribution (10 bins)")
    plt.xlabel("Clustering Coefficient")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig("clustering_histogram.png", dpi=300, bbox_inches="tight")
    plt.close()

    bin_centers = 0.5 * (bins[1:] + bins[:-1])
    return bin_centers, counts


def fit_power_law(x, y):
    # Power law fitting function
    def power_law(x, a, b):
        return a * (x**b)

    # Perform curve fitting
    params, _ = curve_fit(power_law, x, y, maxfev=10000)
    a_fit, b_fit = params

    # Calculate R²
    y_fit = power_law(x, *params)
    residuals = y - y_fit
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    power_r2 = 1 - (ss_res / ss_tot)

    return power_r2


def fit_truncated_power_law(x, y):
    # Truncated power law fitting function
    def truncated_power_law(x, a, b, c):
        return a * (x**b) * np.exp(-c * x)

    # Remove zeros or very small values from x and y for fitting
    nonzero_indices = (x > 0) & (y > 0)
    x_nonzero = x[nonzero_indices]
    y_nonzero = y[nonzero_indices]

    # Perform curve fitting
    p0 = [1.0, -2.0, 0.1]  # Initial guess for the parameters
    params, _ = curve_fit(truncated_power_law, x_nonzero,
                          y_nonzero, p0=p0, maxfev=10000)
    a_fit, b_fit, c_fit = params

    # Calculate R²
    y_fit = truncated_power_law(x_nonzero, *params)
    residuals = y_nonzero - y_fit
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y_nonzero - np.mean(y_nonzero))**2)
    truncated_r2 = 1 - (ss_res / ss_tot)

    return truncated_r2
