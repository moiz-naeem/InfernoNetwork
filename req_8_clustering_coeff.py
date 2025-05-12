import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import linregress


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
    counts, bins = np.histogram(clustering_coeffs, bins=10)
    bin_centers = 0.5 * (bins[1:] + bins[:-1])
    nonzero_indices = counts > 0
    x_nonzero = bin_centers[nonzero_indices]
    y_nonzero = counts[nonzero_indices]

    # --- Figure 1: Histogram ---
    plt.figure()
    plt.hist(clustering_coeffs, bins=10, color="skyblue", edgecolor="black")
    plt.title("Clustering Coefficient Distribution (10 bins)")
    plt.xlabel("Clustering Coefficient")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig("figs/clustering_hist.png", dpi=300)
    plt.close()

    # --- Figure 2: Power Law Fit (log-log) ---
    log_x = np.log10(x_nonzero)
    log_y = np.log10(y_nonzero)
    slope, intercept, r_value, _, _ = linregress(log_x, log_y)
    power_law_r_squared = r_value**2
    power_law_fit = intercept + slope * log_x

    plt.figure()
    plt.scatter(log_x, log_y, color="blue", alpha=0.7)
    plt.plot(log_x, power_law_fit, "r-", linewidth=2)
    plt.title(f"Power Law Fit (log-log) - R² = {power_law_r_squared:.3f}")
    plt.xlabel("Log(Clustering Coefficient)")
    plt.ylabel("Log(Frequency)")
    plt.tight_layout()
    plt.savefig("figs/clustering_powerlaw.png", dpi=300)
    plt.close()

    # --- Figure 3: Truncated Power Law ---
    def truncated_power_law(x, a, b, c):
        return a * (x**b) * np.exp(-c * x)

    p0 = [1.0, slope, 0.1]
    params, _ = curve_fit(truncated_power_law, x_nonzero,
                          y_nonzero, p0=p0, maxfev=10000)
    a_fit, b_fit, c_fit = params
    y_fit = truncated_power_law(x_nonzero, *params)
    residuals = y_nonzero - y_fit
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y_nonzero - np.mean(y_nonzero)) ** 2)
    trunc_power_law_r_squared = 1 - (ss_res / ss_tot)

    plt.figure()
    plt.scatter(x_nonzero, y_nonzero, color="green", alpha=0.7)
    x_fit = np.linspace(min(x_nonzero), max(x_nonzero), 100)
    y_fit = truncated_power_law(x_fit, *params)
    plt.plot(x_fit, y_fit, "r-", linewidth=2)
    plt.title(
        f"Truncated Power Law Fit - R² = {trunc_power_law_r_squared:.3f}")
    plt.xlabel("Clustering Coefficient")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig("figs/clustering_truncated.png", dpi=300)
    plt.close()

    # --- Figure 4: Comparison in Log-Log ---
    plt.figure()
    plt.scatter(log_x, log_y, color="purple", alpha=0.7, label="Data")
    plt.plot(log_x, power_law_fit, "r-", linewidth=2,
             label=f"Power Law (R²={power_law_r_squared:.3f})")
    log_y_trunc = np.log10(truncated_power_law(10**log_x, *params))
    plt.plot(log_x, log_y_trunc, "g--", linewidth=2,
             label=f"Trunc Power Law (R²={trunc_power_law_r_squared:.3f})")
    plt.title("Comparison of Fits (log-log)")
    plt.xlabel("Log(Clustering Coefficient)")
    plt.ylabel("Log(Frequency)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("figs/clustering_comparison.png", dpi=300)
    plt.close()

    # --- Console Output ---
    print("\nClustering Coefficient Distribution Analysis:")
    print(
        f"Power Law Fit R² = {power_law_r_squared:.3f}, exponent = {slope:.3f}")
    print(f"Truncated Power Law Fit R² = {trunc_power_law_r_squared:.3f}")
    print(f"Parameters: a = {a_fit:.3f}, b = {b_fit:.3f}, c = {c_fit:.3f}")

    if trunc_power_law_r_squared > power_law_r_squared:
        better_model = "Truncated Power Law"
        better_r2 = trunc_power_law_r_squared
    else:
        better_model = "Pure Power Law"
        better_r2 = power_law_r_squared

    print(f"Better fit: {better_model} (R² = {better_r2:.3f})")

    if better_r2 > 0.85:
        fit_quality = "Very strong evidence"
    elif better_r2 > 0.7:
        fit_quality = "Moderate evidence"
    else:
        fit_quality = "Weak evidence"
    print(f"{fit_quality} for {better_model.lower()} fit")

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
