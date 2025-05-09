import networkx as nx
from matplotlib import pyplot as plt
import numpy as np
from scipy.stats import linregress
from scipy.optimize import curve_fit


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


# Are these needed?


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
