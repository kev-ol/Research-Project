import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from results import extract_cov_mfvi, cov2corr


"""UQF (Uncertainty Quantification Factor)"""

def plot_uqf_boxplot(results_dict, methods=("mfvi", "ssvi_i", "ssvi_c")):
    """Boxplot of UQF values, pooled across seeds, for each method.

    Parameters
    ----------
    results_dict : dict
        Mapping from seed to a dict containing, for each entry in `methods`,
        a sub-dict with key "uqf" (array_like of float, the per-country UQF
        values for that seed and method).
    methods : sequence of str, optional
        Method names to include. Default is ("mfvi", "ssvi_i", "ssvi_c").

    Returns
    -------
    None
        Displays the matplotlib figure; nothing is returned.
    """
    seeds = list(results_dict.keys())
    pooled = {method: np.concatenate([results_dict[seed][method]["uqf"] for seed in seeds]) for method in methods}

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.boxplot([pooled[m] for m in methods], tick_labels=methods)
    ax.set_ylabel("UQF")
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()


"""Correlation Mean Absolute Error (MAE)"""

def plot_corr_mae_boxplots(results_dict, N, K, Z_width):
    """Boxplots of mean absolute delta_c correlation error vs Gibbs, pooled
    across countries and seeds, for each VI method.

    Parameters
    ----------
    results_dict : dict
        Mapping from seed to a single-seed results dict (see
        `results.corr_mae_table` for its required structure).
    N : int
        Number of endogenous variables.
    K : int
        Number of regressors per equation.
    Z_width : int
        Number of non-exchangeable regressors per equation.

    Returns
    -------
    None
        Displays the matplotlib figure; nothing is returned.
    """
    seeds = list(results_dict.keys())
    method_names = ["MFVI", "SSVI-I", "SSVI-C"]
    pooled = {name: [] for name in method_names}

    for seed in seeds:
        results = results_dict[seed]
        C = results["C"]
        V_delta = results["mfvi"]["results"]["V_delta"]
        cov_mfvi = extract_cov_mfvi(V_delta, N, K, Z_width, C)
        cov_ssvi_i = results["ssvi_i"]["results"]["cov_deltac"]
        cov_ssvi_c = results["ssvi_c"]["results"]["cov_deltac"]
        cov_true = results["cov_true"]
        methods = {"MFVI": cov_mfvi, "SSVI-I": cov_ssvi_i, "SSVI-C": cov_ssvi_c}

        for c in range(C):
            corr_true = cov2corr(cov_true[c])
            np.fill_diagonal(corr_true, np.nan)
            for name, cov in methods.items():
                corr_method = cov2corr(cov[c])
                np.fill_diagonal(corr_method, np.nan)
                diff = corr_method - corr_true
                pooled[name].append(np.nanmean(np.abs(diff)))

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.boxplot([pooled[name] for name in method_names], tick_labels=method_names)
    ax.set_ylabel("Mean abs correlation error vs Gibbs")
    ax.grid(axis='y', alpha=0.3)
    fig.suptitle("δ_c correlation structure error (pooled across countries & seeds)", fontsize=14)
    plt.tight_layout()
    plt.show()


"""Accuracy Measure (Faes et al. 2011)"""

def plot_accuracy_boxplots(results_faes, method_name, C):
    """Boxplots of Faes accuracy per parameter block, for one VI method vs Gibbs.

    Parameters
    ----------
    results_faes : dict
        Output of `results.compute_faes_scores`.
    method_name : str
        Label for the VI method, used in the figure title.
    C : int
        Number of countries.

    Returns
    -------
    None
        Displays the matplotlib figure; nothing is returned.
    """
    country_labels = [f'C{c+1}' for c in range(C)]

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # beta_c: one box per country
    beta_c_data = [results_faes['beta_c'][c] for c in range(C)]
    axes[0, 0].boxplot(beta_c_data, labels=country_labels)
    axes[0, 0].set_title(r'$\beta_c$')
    axes[0, 0].set_ylabel('Accuracy (%)')

    # gamma_c: one box per country
    gamma_c_data = [results_faes['gamma_c'][c] for c in range(C)]
    axes[0, 1].boxplot(gamma_c_data, labels=country_labels)
    axes[0, 1].set_title(r'$\gamma_c$')
    axes[0, 1].set_ylabel('Accuracy (%)')

    # Sigma_c diagonals: one box per country
    sigma_c_data = [results_faes['Sigma_c'][c] for c in range(C)]
    axes[0, 2].boxplot(sigma_c_data, labels=country_labels)
    axes[0, 2].set_title(r'$\Sigma_c$ diagonals')
    axes[0, 2].set_ylabel('Accuracy (%)')

    # beta_0: single box over all N*K coefficients
    beta_0_data = [results_faes['beta_0']]
    axes[1, 0].boxplot(beta_0_data, labels=[r'$\beta_0$'])
    axes[1, 0].set_ylabel('Accuracy (%)')

    # lambda: single value, shown as a point
    axes[1, 1].scatter([1], [results_faes['lam']], s=100, zorder=5)
    axes[1, 1].set_xlim(0.5, 1.5)
    axes[1, 1].set_xticks([1])
    axes[1, 1].set_xticklabels([r'$\lambda$'])
    axes[1, 1].set_ylabel('Accuracy (%)')

    axes[1, 2].set_visible(False)

    # data-driven y-limits per subplot, except lambda (kept at full 0-100 range)
    boxplot_axes_data = {
        (0, 0): beta_c_data,
        (0, 1): gamma_c_data,
        (0, 2): sigma_c_data,
        (1, 0): beta_0_data,
    }

    for (row, col), data in boxplot_axes_data.items():
        ax = axes[row, col]
        all_vals = np.concatenate([np.asarray(d) for d in data])
        lo, hi = np.min(all_vals), np.max(all_vals)
        pad = max((hi - lo) * 0.1, 1.0)
        ax.set_ylim(max(0, lo - pad), min(100, hi + pad))

    axes[1, 1].set_ylim(0, 100)  # lambda: keep full range

    for ax in axes.flat:
        if ax.get_visible():
            ax.grid(axis='y', alpha=0.3)

    fig.suptitle(f'Faes et al. Accuracy: {method_name} vs Gibbs', fontsize=14)
    plt.tight_layout()
    plt.show()


def plot_accuracy_boxplots_pooled(results_dict, method_name, method_key):
    """Same layout as plot_accuracy_boxplots, but pooled across seeds

    Parameters
    ----------
    results_dict : dict
        Mapping from seed to a per-seed results dict, each containing key
        "C" (int) and `method_key` (dict with key "faes", the output of
        `results.compute_faes_scores`).
    method_name : str
        Label for the VI method, used in the figure title.
    method_key : str
        Key into each seed's results dict identifying this method's results.

    Returns
    -------
    None
        Displays the matplotlib figure; nothing is returned.
    """
    seeds = list(results_dict.keys())
    C = results_dict[seeds[0]]["C"]

    faes_all = [results_dict[seed][method_key]["faes"] for seed in seeds]

    beta_c_data = [np.concatenate([faes['beta_c'][c] for faes in faes_all]) for c in range(C)]
    gamma_c_data = [np.concatenate([faes['gamma_c'][c] for faes in faes_all]) for c in range(C)]
    sigma_c_data = [np.concatenate([faes['Sigma_c'][c] for faes in faes_all]) for c in range(C)]
    beta_0_data = [np.concatenate([faes['beta_0'] for faes in faes_all])]
    lam_data = [faes['lam'] for faes in faes_all]  # one value per seed

    country_labels = [f'C{c+1}' for c in range(C)]
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    axes[0, 0].boxplot(beta_c_data, labels=country_labels)
    axes[0, 0].set_title(r'$\beta_c$')
    axes[0, 0].set_ylabel('Accuracy (%)')

    axes[0, 1].boxplot(gamma_c_data, labels=country_labels)
    axes[0, 1].set_title(r'$\gamma_c$')
    axes[0, 1].set_ylabel('Accuracy (%)')

    axes[0, 2].boxplot(sigma_c_data, labels=country_labels)
    axes[0, 2].set_title(r'$\Sigma_c$ diagonals')
    axes[0, 2].set_ylabel('Accuracy (%)')

    axes[1, 0].boxplot(beta_0_data, labels=[r'$\beta_0$'])
    axes[1, 0].set_ylabel('Accuracy (%)')

    axes[1, 1].scatter([1] * len(lam_data), lam_data, s=80, zorder=5, alpha=0.6)
    axes[1, 1].set_xlim(0.5, 1.5)
    axes[1, 1].set_xticks([1])
    axes[1, 1].set_xticklabels([r'$\lambda$'])
    axes[1, 1].set_ylabel('Accuracy (%)')

    axes[1, 2].set_visible(False)

    boxplot_axes_data = {
        (0, 0): beta_c_data, (0, 1): gamma_c_data,
        (0, 2): sigma_c_data, (1, 0): beta_0_data,
    }
    for (row, col), data in boxplot_axes_data.items():
        ax = axes[row, col]
        all_vals = np.concatenate([np.asarray(d) for d in data])
        lo, hi = np.min(all_vals), np.max(all_vals)
        pad = max((hi - lo) * 0.1, 1.0)
        ax.set_ylim(max(0, lo - pad), min(100, hi + pad))

    axes[1, 1].set_ylim(0, 100)

    for ax in axes.flat:
        if ax.get_visible():
            ax.grid(axis='y', alpha=0.3)

    fig.suptitle(f'Faes et al. Accuracy (pooled across {len(seeds)} seeds): {method_name} vs Gibbs', fontsize=14)
    plt.tight_layout()
    plt.show()


"""Impulse Response Functions"""

def plot_irfs_comparison(gibbs_irfs, vi_irfs, country_names, variable_names, vi_label="VI"):
    """
    Plot a grid of impulse response comparisons between Gibbs and a VI method.

    Parameters
    ----------
    gibbs_irfs : numpy.ndarray of shape (n_draws_gibbs, C, H+1, N)
        IRFs from `results.compute_irfs` using Gibbs posterior draws —
        plotted as blue fanchart (5-95 percentile, 5% steps) + black solid
        median.
    vi_irfs : numpy.ndarray of shape (n_draws_vi, C, H+1, N)
        IRFs from `results.compute_irfs` using a VI method's posterior
        draws — plotted as red solid median + red dashed 5/95 percentiles.
    country_names : sequence of length C of str
        Country labels, used as column titles.
    variable_names : sequence of length N of str
        Variable labels, used as row y-labels.
    vi_label : str, optional
        Label for the VI method, used in the figure title. Default is "VI".

    Returns
    -------
    None
        Displays the matplotlib figure; nothing is returned.
    """
    n_draws_g, C, H_plus_1, N = gibbs_irfs.shape
    horizons = np.arange(H_plus_1)

    gibbs_median = np.median(gibbs_irfs, axis=0)          # (C, H+1, N)
    vi_median = np.median(vi_irfs, axis=0)                # (C, H+1, N)
    vi_p5 = np.percentile(vi_irfs, 5, axis=0)
    vi_p95 = np.percentile(vi_irfs, 95, axis=0)

    # fanchart bands in 5% steps: (5,95), (10,90), ..., (45,55)
    band_pairs = [(p, 100 - p) for p in range(5, 50, 5)]

    fig, axes = plt.subplots(N, C, figsize=(3 * C, 2.2 * N), sharex=True)

    for n in range(N):
        for c in range(C):
            ax = axes[n, c]

            for lo, hi in band_pairs:
                lo_band = np.percentile(gibbs_irfs[:, c, :, n], lo, axis=0)
                hi_band = np.percentile(gibbs_irfs[:, c, :, n], hi, axis=0)
                ax.fill_between(horizons, lo_band, hi_band,
                                 color="tab:blue", alpha=0.08)

            ax.plot(horizons, gibbs_median[c, :, n], color="black", linewidth=1.2)

            ax.plot(horizons, vi_median[c, :, n], color="red", linewidth=1.2)
            ax.plot(horizons, vi_p5[c, :, n], color="red", linestyle="--", linewidth=1)
            ax.plot(horizons, vi_p95[c, :, n], color="red", linestyle="--", linewidth=1)

            ax.axhline(0, color="gray", linewidth=0.7, linestyle=":")

            if n == 0:
                ax.set_title(country_names[c])
            if c == 0:
                ax.set_ylabel(variable_names[n])
            if n == N - 1:
                ax.set_xlabel("Horizon")
    fig.suptitle(f"Gibbs vs {vi_label}: Impulse Response Comparison", y=1.02)
    plt.tight_layout()
    plt.show()


def plot_wasserstein_grid_comparison(distances_dict, country_names, variable_names):
    """
    Plot a grid comparison of Wasserstein-distance curves for multiple VI
    methods against Gibbs.

    Layout matches the IRF grid: rows = variables, columns = countries.
    Each panel overlays one line per method.

    Parameters
    ----------
    distances_dict : dict
        Mapping from method label (str) to numpy.ndarray of shape
        (C, H+1, N) from `results.compute_wasserstein_curve`, e.g.
        {"mfvi": wass_mfvi, "SSVI_I": wass_ssvi_i, "SSVI_C": wass_ssvi_c}.
    country_names : sequence of length C of str
        Country labels, used as column titles.
    variable_names : sequence of length N of str
        Variable labels, used as row y-labels.

    Returns
    -------
    None
        Displays the matplotlib figure; nothing is returned.
    """
    method_names = list(distances_dict.keys())
    C, H_plus_1, N = next(iter(distances_dict.values())).shape
    horizons = np.arange(H_plus_1)

    colors = plt.cm.tab10(np.linspace(0, 1, len(method_names)))

    fig, axes = plt.subplots(N, C, figsize=(3 * C, 2.2 * N), sharex=True)

    for n in range(N):
        for c in range(C):
            ax = axes[n, c]
            for method, color in zip(method_names, colors):
                ax.plot(horizons, distances_dict[method][c, :, n],
                         color=color, linewidth=1.2, label=method)
            ax.axhline(0, color="gray", linewidth=0.7, linestyle=":")

            if n == 0:
                ax.set_title(country_names[c])
            if c == 0:
                ax.set_ylabel(variable_names[n])
            if n == N - 1:
                ax.set_xlabel("Horizon")

    axes[0, 0].legend(loc="upper right", fontsize=8)
    fig.suptitle("Wasserstein distance: Gibbs vs VI methods", y=1.02)
    plt.tight_layout()
    plt.show()


"""Effect of lambda on coefficient means"""

def plot_conditional_mean_grid(methods, n_bins=10, title=None):
    """Plot, for several methods, the conditional mean of a coefficient given
    lambda, binned into quantiles of lambda, one line per country.

    Parameters
    ----------
    methods : sequence of tuple
        Each tuple is (name, lam, coef_by_country, country_names,
        beta0_samples):
        name : str
            Method label, used as the subplot title.
        lam : array_like of shape (n_samples,)
            Posterior samples of lambda for this method.
        coef_by_country : sequence of length C of array_like of shape (n_samples,)
            Posterior samples of a single coefficient, per country.
        country_names : sequence of length C of str
            Country labels, used in the legend.
        beta0_samples : array_like of shape (n_samples,) or None
            Posterior samples of the corresponding beta_0 coefficient, or
            None to skip plotting it.
    n_bins : int, optional
        Number of quantile bins of lambda to average within. Default is 10.
    title : str or None, optional
        Overall figure title. Default is None.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure.
    axes : numpy.ndarray of matplotlib.axes.Axes
        The created 2x2 grid of axes.
    """
    fig, axes = plt.subplots(2, 2, figsize=(10, 9), sharey=True)
    handles, labels = None, None

    for ax, (name, lam, coef_by_country, country_names, beta0_samples) in zip(axes.flat, methods):
        for c, cname in enumerate(country_names):
            bins = pd.qcut(lam, q=n_bins)
            df = pd.DataFrame({"lam": lam, "coef": coef_by_country[c]})
            grouped = df.groupby(bins, observed=True)
            ax.plot(grouped["lam"].mean(), grouped["coef"].mean(), marker="o", label=cname)

        if beta0_samples is not None:
            bins = pd.qcut(lam, q=n_bins)
            df = pd.DataFrame({"lam": lam, "coef": beta0_samples})
            grouped = df.groupby(bins, observed=True)
            ax.plot(grouped["lam"].mean(), grouped["coef"].mean(), marker="s",
                     color="black", linestyle="--", label=r"$\beta_0$")

        ax.set_xlabel(r"$\lambda$")
        ax.set_ylabel("conditional mean")
        ax.set_title(name)
        if handles is None:
            handles, labels = ax.get_legend_handles_labels()

    if title:
        fig.suptitle(title, y=0.98)

    fig.legend(handles, labels, loc="lower center", ncol=len(labels), bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=[0, 0.06, 1, 0.94])
    return fig, axes


def plot_conditional_mean_by_seed(results_by_seed, k, n_bins=10):
    """For each seed, plot the conditional mean of `beta_c[:, :, k]` (and the
    matching `beta_0[k]`) given lambda, for every method, using
    `plot_conditional_mean_grid`.

    Parameters
    ----------
    results_by_seed : dict
        Mapping from seed to a results dict containing, for each of "mfvi",
        "ssvi_i", "ssvi_c", "gibbs", a sub-dict (keyed "samples" for VI
        methods, "results" for "gibbs") with keys "beta_c" (array_like,
        shape (n_samples, C, N*K)), "beta_0" (array_like, shape
        (n_samples, N*K)), and "lam" (array_like, shape (n_samples,)); and
        key "config" (object with attribute `country_names`).
    k : int
        Index into the flattened N*K coefficient vector to plot.
    n_bins : int, optional
        Number of quantile bins of lambda to average within. Default is 10.

    Returns
    -------
    dict
        Mapping from seed to the matplotlib.figure.Figure produced for that
        seed.
    """
    figs = {}
    for seed, results in results_by_seed.items():
        arr = {
            method: np.array(results[method]["samples" if method != "gibbs" else "results"]["beta_c"])
            for method in ["mfvi", "ssvi_i", "ssvi_c", "gibbs"]
        }
        beta0 = {
            method: np.array(results[method]["samples" if method != "gibbs" else "results"]["beta_0"])
            for method in ["mfvi", "ssvi_i", "ssvi_c", "gibbs"]
        }
        lam = {
            method: results[method]["samples" if method != "gibbs" else "results"]["lam"]
            for method in ["mfvi", "ssvi_i", "ssvi_c", "gibbs"]
        }
        country_names = results["config"].country_names

        methods = [
            (name.upper().replace("_", "-"), lam[name],
             [arr[name][:, c, k] for c in range(arr[name].shape[1])],
             country_names, beta0[name][:, k])
            for name in ["mfvi", "ssvi_i", "ssvi_c", "gibbs"]
        ]

        fig, axes = plot_conditional_mean_grid(
            methods, n_bins=n_bins,
            title=f"Conditional mean of beta_c[{k}] given λ, by country (seed {seed})"
        )
        figs[seed] = fig
    return figs


"""Comparison to true parameters"""

def plot_lambda_intervals(results_by_seed, true_by_seed, methods, levels=(0.5, 0.8, 0.95), ax=None):
    """Plot, per seed and method, credible intervals for lambda at several
    levels alongside the true simulating value.

    Parameters
    ----------
    results_by_seed : dict
        Mapping from seed to a dict of method -> dict (keyed "results" for
        "gibbs", "samples" otherwise) containing "lam" (array_like of
        float).
    true_by_seed : dict
        Mapping from seed to a dict containing "lam" (float), the true
        simulating value of lambda for that seed.
    methods : sequence of str
        Method names to include.
    levels : sequence of float, optional
        Central credible-interval levels to plot. Default is (0.5, 0.8, 0.95).
    ax : matplotlib.axes.Axes or None, optional
        Axes to draw on; a new figure and axes are created if None.

    Returns
    -------
    matplotlib.axes.Axes
        The axes the intervals were drawn on.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    seeds = list(results_by_seed.keys())
    n_methods = len(methods)
    linewidths = {levels[0]: 8, levels[1]: 4, levels[2]: 1.5}

    for m_idx, method in enumerate(methods):
        result_key = "results" if method == "gibbs" else "samples"
        for s_idx, seed in enumerate(seeds):
            samples = np.array(results_by_seed[seed][method][result_key]["lam"])
            true_val = true_by_seed[seed]["lam"]
            y = s_idx * (n_methods + 1) + m_idx

            for level in levels:
                alpha = 1 - level
                lo, hi = np.quantile(samples, [alpha / 2, 1 - alpha / 2])
                lo = max(lo, 1e-8)  # guard against zero/negative on log scale
                ax.plot([lo, hi], [y, y], color=f"C{m_idx}",
                        linewidth=linewidths[level], solid_capstyle="round",
                        label=method if (s_idx == 0 and level == levels[0]) else None)

            ax.scatter([true_val], [y], color="black", marker="x", s=60, zorder=5)

    ax.set_xscale("log")
    ax.set_yticks([s_idx * (n_methods + 1) + n_methods / 2 for s_idx in range(len(seeds))])
    ax.set_yticklabels([f"seed {s}" for s in seeds])
    for s_idx in range(len(seeds)):
        ax.axhline(s_idx * (n_methods + 1) - 0.5, color="grey", linewidth=0.5, alpha=0.5)
    ax.set_xlabel(r"$\lambda$ (log scale)")
    ax.legend()
    return ax


"""Model Diagnostics"""

def plot_diagnostics(ssvi_i_trace, ssvi_c_trace, ssvi_i_ess, ssvi_c_ess, rhat, title_suffix=""):
    """Plot ULA trace, ESS-across-iterations, and R-hat diagnostics for
    SSVI-I and SSVI-C.

    Parameters
    ----------
    ssvi_i_trace : array_like of shape (n_steps,)
        log(lambda) ULA trace for SSVI-I (typically the last outer iteration).
    ssvi_c_trace : array_like of shape (n_steps,)
        log(lambda) ULA trace for SSVI-C (typically the last outer iteration).
    ssvi_i_ess : sequence of float
        Effective sample size at each CAVI iteration, for SSVI-I.
    ssvi_c_ess : sequence of float
        Effective sample size at each CAVI iteration, for SSVI-C.
    rhat : array_like of shape (D,)
        R-hat value for each scalar parameter component (e.g. from
        `gibbs._compute_diagnostics`).
    title_suffix : str, optional
        Text appended to each subplot title. Default is "".

    Returns
    -------
    None
        Displays the matplotlib figure and prints an R-hat summary; nothing
        is returned.
    """
    fig, axes = plt.subplots(1, 4, figsize=(19, 4))

    axes[0].plot(ssvi_i_trace)
    axes[0].set_title(f"SSVI-I: log(lambda) ULA trace{title_suffix}")

    axes[1].plot(ssvi_c_trace)
    axes[1].set_title(f"SSVI-C: log(lambda) ULA trace{title_suffix}")

    axes[2].plot(ssvi_i_ess, marker="o", label="SSVI-I")
    axes[2].plot(ssvi_c_ess, marker="o", label="SSVI-C")
    axes[2].set_xlabel("CAVI iteration")
    axes[2].set_ylabel("ESS")
    axes[2].set_title(f"ESS across CAVI iterations{title_suffix}")
    axes[2].legend()

    axes[3].hist(rhat, bins=50)
    axes[3].axvline(1.01, color='red', linestyle='--', label='common threshold (1.01)')
    axes[3].set_xlabel('R-hat')
    axes[3].set_ylabel('count')
    axes[3].set_title(f"R-hat distribution{title_suffix}")
    axes[3].legend()

    fig.tight_layout()
    plt.show()

    print(f"max R-hat: {rhat.max():.4f}, min R-hat: {rhat.min():.4f}, "
          f"# > 1.01: {(rhat > 1.01).sum()} / {len(rhat)}")
