from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import to_rgba
from results import extract_cov_mfvi, cov2corr
from matplotlib.patches import Patch

### Preparations for Figures ###

FIGURES_DIR = Path("Figures")


def _save_fig(fig, save_name):
    """Save fig as a PDF to Figures/{save_name}.pdf (creating the folder
    if needed). No operationif `save_name` is None."""
    if save_name is None:
        return
    FIGURES_DIR.mkdir(exist_ok=True)
    fig.savefig(FIGURES_DIR / f"{save_name}.pdf", bbox_inches="tight")


## Method colour/ordering convention
# All comparison figures should order methods as MFVI, SSVI-I, SSVI-C, Gibbs
# and colour them Gibbs=yellow (gold), MFVI=blue, SSVI-I=green, SSVI-C=red.

METHOD_ORDER = ["mfvi", "ssvi_i", "ssvi_c", "gibbs"]
METHOD_LABELS = {"mfvi": "MFVI", "ssvi_i": "SSVI-I", "ssvi_c": "SSVI-C", "gibbs": "Gibbs"}
METHOD_COLOURS = {"mfvi": "tab:blue", "ssvi_i": "tab:green", "ssvi_c": "tab:red", "gibbs": "gold"}


def _method_key(label):
    """Normalize a method label or key (e.g. "SSVI-I", "ssvi_i", "SSVI_I")
    to its canonical key in METHOD_ORDER."""
    return str(label).strip().lower().replace("-", "_").replace(" ", "_")


def _method_colour(label, default="tab:gray"):
    """Look up the convention colour for a method label or key."""
    return METHOD_COLOURS.get(_method_key(label), default)


def _ordered_methods(labels_or_keys):
    """Sort an iterable of method labels/keys into the canonical MFVI,
    SSVI-I, SSVI-C, Gibbs order; unrecognized entries are moved to the end,
    keeping their relative order."""
    items = list(labels_or_keys)

    def sort_key(item):
        key = _method_key(item)
        return METHOD_ORDER.index(key) if key in METHOD_ORDER else len(METHOD_ORDER)

    return sorted(items, key=sort_key)


MEDIAN_PROPS = dict(color="black", linewidth=2)  # kept boxplot medianopaque so it reads over any fill, incl. gold


def _shade_boxes(bp, labels_or_keys, alpha=0.35):
    """Lightly fill each box of a boxplot result
    with its own method's convention colour. Alpha is baked into the
    facecolor only, so box edges and the median line (drawn separately,
    on top, per MEDIAN_PROPS) stay fully opaque."""
    for patch, item in zip(bp["boxes"], labels_or_keys):
        patch.set_facecolor(to_rgba(_method_colour(item), alpha))
        patch.set_edgecolor("black")


def _shade_all_boxes(bp, color, alpha=0.35):
    """Lightly fill every box of a boxplot result
    with a single convention colour (for single-method figures). Alpha is
    baked into the facecolor only, as in _shade_boxes."""
    for patch in bp["boxes"]:
        patch.set_facecolor(to_rgba(color, alpha))
        patch.set_edgecolor("black")


### Uncertainty Quantification Factor ###

def plot_uqf_boxplot(results_dict, methods=("mfvi", "ssvi_i", "ssvi_c"), save_name=None):
    """Boxplot of UQF values, pooled across seeds, for each method.
    Saves to Figures/{save_name}.pdf if save_name is given."""
    methods = _ordered_methods(methods)
    seeds = list(results_dict.keys())
    pooled = {method: np.concatenate([results_dict[seed][method]["uqf"] for seed in seeds]) for method in methods}

    display_labels = [METHOD_LABELS.get(m, m.upper().replace("_", "-")) for m in methods]

    fig, ax = plt.subplots(figsize=(6, 5))
    bp = ax.boxplot([pooled[m] for m in methods], tick_labels=display_labels, patch_artist=True, medianprops=MEDIAN_PROPS)
    _shade_boxes(bp, methods)
    ax.set_ylabel("UQF", fontsize=22)
    ax.tick_params(labelsize=25)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    _save_fig(fig, save_name)
    plt.show()

### Mean Absolute Error of Off-Diagonal Correlations ###

def plot_corr_mae_boxplots(results_dict, N, K, Z_width, save_name=None):
    """Boxplots of mean absolute delta_c correlation error vs Gibbs, pooled
    across countries and seeds, for each VI method.
    Saves to Figures/{save_name}.pdf if save_name is given."""
    seeds = list(results_dict.keys())
    method_keys = ["mfvi", "ssvi_i", "ssvi_c"]
    method_names = [METHOD_LABELS[k] for k in method_keys]
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
    bp = ax.boxplot([pooled[name] for name in method_names], tick_labels=method_names, patch_artist=True, medianprops=MEDIAN_PROPS)
    _shade_boxes(bp, method_keys)
    ax.set_ylabel("MAE", fontsize=22)
    ax.tick_params(labelsize=25)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    _save_fig(fig, save_name)
    plt.show()

### Accuracy Measure ###

def plot_accuracy_boxplots(results_faes, method_name, C, save_name=None):
    """Boxplots of accuracy per parameter block, for one VI method vs Gibbs.
    Saves to Figures/{save_name}.pdf if save_name is given."""
    country_labels = [f'C{c+1}' for c in range(C)]
    color = _method_colour(method_name)

    fig, axes = plt.subplots(1, 5, figsize=(25, 6))

    # beta_c: one box per country
    beta_c_data = [results_faes['beta_c'][c] for c in range(C)]
    bp0 = axes[0].boxplot(beta_c_data, labels=country_labels, patch_artist=True, medianprops=MEDIAN_PROPS)
    _shade_all_boxes(bp0, color)
    axes[0].set_title(r'$\beta_c$', fontsize=35)
    axes[0].set_ylabel('Accuracy (%)', fontsize=25)

    # gamma_c: one box per country
    gamma_c_data = [results_faes['gamma_c'][c] for c in range(C)]
    bp1 = axes[1].boxplot(gamma_c_data, labels=country_labels, patch_artist=True, medianprops=MEDIAN_PROPS)
    _shade_all_boxes(bp1, color)
    axes[1].set_title(r'$\gamma_c$', fontsize=35)

    # Sigma_c diagonals: one box per country
    sigma_c_data = [results_faes['Sigma_c'][c] for c in range(C)]
    bp2 = axes[2].boxplot(sigma_c_data, labels=country_labels, patch_artist=True, medianprops=MEDIAN_PROPS)
    _shade_all_boxes(bp2, color)
    axes[2].set_title(r'$\Sigma_c$ diagonals', fontsize=35)

    # beta_0: single box over all N*K coefficients
    beta_0_data = [results_faes['beta_0']]
    bp3 = axes[3].boxplot(beta_0_data, patch_artist=True, medianprops=MEDIAN_PROPS)
    _shade_all_boxes(bp3, color)
    axes[3].set_title(r'$\beta_0$', fontsize=35)
    axes[3].set_xticks([])

    # lambda: single value, shown as a point
    axes[4].scatter([1], [results_faes['lam']], s=250, zorder=5, color=color, edgecolor="black", linewidth=0.6)
    axes[4].set_xlim(0.5, 1.5)
    axes[4].set_xticks([])
    axes[4].set_title(r'$\lambda$', fontsize=35)

    # data-driven y-limits per subplot, except lambda (kept at full 0-100 range)
    boxplot_axes_data = {
        0: beta_c_data,
        1: gamma_c_data,
        2: sigma_c_data,
        3: beta_0_data,
    }

    for idx, data in boxplot_axes_data.items():
        ax = axes[idx]
        all_vals = np.concatenate([np.asarray(d) for d in data])
        lo, hi = np.min(all_vals), np.max(all_vals)
        pad = max((hi - lo) * 0.1, 1.0)
        ax.set_ylim(max(0, lo - pad), min(100, hi + pad))

    axes[4].set_ylim(0, 100)  # lambda: keep full range

    for ax in axes.flat:
        ax.grid(axis='y', alpha=0.3)
        ax.tick_params(labelsize=25)

    plt.tight_layout()
    _save_fig(fig, save_name)
    plt.show()

def plot_accuracy_boxplots_pooled(results_dict, method_name, method_key, save_name=None):
    """Boxplots of accuracy per parameter block, for one VI method vs Gibbs, pooled across seeds and countries.
    Saves to Figures/{save_name}.pdf if save_name is given."""
    seeds = list(results_dict.keys())
    C = results_dict[seeds[0]]["C"]

    faes_all = [results_dict[seed][method_key]["faes"] for seed in seeds]

    beta_c_data = [np.concatenate([faes['beta_c'][c] for c in range(C) for faes in faes_all])]
    gamma_c_data = [np.concatenate([faes['gamma_c'][c] for c in range(C) for faes in faes_all])]
    sigma_c_data = [np.concatenate([faes['Sigma_c'][c] for c in range(C) for faes in faes_all])]
    beta_0_data = [np.concatenate([faes['beta_0'] for faes in faes_all])]

    color = METHOD_COLOURS.get(_method_key(method_key), _method_colour(method_name))

    fig, axes = plt.subplots(1, 4, figsize=(20, 6))

    bp0 = axes[0].boxplot(beta_c_data, patch_artist=True, medianprops=MEDIAN_PROPS)
    _shade_all_boxes(bp0, color)
    axes[0].set_title(r'$\beta_c$', fontsize=35)
    axes[0].set_ylabel('Accuracy (%)', fontsize=25)
    axes[0].set_xticks([])

    bp1 = axes[1].boxplot(gamma_c_data, patch_artist=True, medianprops=MEDIAN_PROPS)
    _shade_all_boxes(bp1, color)
    axes[1].set_title(r'$\gamma_c$', fontsize=35)
    axes[1].set_xticks([])

    bp2 = axes[2].boxplot(sigma_c_data, patch_artist=True, medianprops=MEDIAN_PROPS)
    _shade_all_boxes(bp2, color)
    axes[2].set_title(r'$\Sigma_c$ diagonals', fontsize=35)
    axes[2].set_xticks([])

    bp3 = axes[3].boxplot(beta_0_data, patch_artist=True, medianprops=MEDIAN_PROPS)
    _shade_all_boxes(bp3, color)
    axes[3].set_title(r'$\beta_0$', fontsize=35)
    axes[3].set_xticks([])

    boxplot_axes_data = {
        0: beta_c_data,
        1: gamma_c_data,
        2: sigma_c_data,
        3: beta_0_data,
    }

    for idx, data in boxplot_axes_data.items():
        ax = axes[idx]
        all_vals = np.concatenate([np.asarray(d) for d in data])
        lo, hi = np.min(all_vals), np.max(all_vals)
        pad = max((hi - lo) * 0.1, 1.0)
        ax.set_ylim(max(0, lo - pad), min(100, hi + pad))

    for ax in axes.flat:
        ax.grid(axis='y', alpha=0.3)
        ax.tick_params(labelsize=25)

    plt.tight_layout()
    _save_fig(fig, save_name)
    plt.show()

def plot_lambda_accuracy_pooled(results_dict, method_pairs=(("MFVI", "mfvi"), ("SSVI-I", "ssvi_i"), ("SSVI-C", "ssvi_c")), save_name=None):
    """Scatter of pooled lambda accuracy, one column per VI method, with one point per seed.
    Saves to Figures/{save_name}.pdf if save_name is given."""
    ordered_keys = _ordered_methods([key for _, key in method_pairs])
    method_pairs = sorted(method_pairs, key=lambda pair: ordered_keys.index(pair[1]))
    seeds = list(results_dict.keys())

    fig, ax = plt.subplots(figsize=(8, 6))

    for i, (label, key) in enumerate(method_pairs, start=1):
        lam_data = [results_dict[seed][key]["faes"]['lam'] for seed in seeds]
        ax.scatter([i] * len(lam_data), lam_data, s=250, zorder=5, alpha=0.85,
                   color=_method_colour(key), edgecolor="black", linewidth=0.6)

    ax.set_xlim(0.5, len(method_pairs) + 0.5)
    ax.set_xticks(range(1, len(method_pairs) + 1))
    ax.set_xticklabels([label for label, _ in method_pairs], fontsize=16)
    ax.set_ylabel('Accuracy (%)', fontsize=22)
    ax.set_ylim(0, 100)
    ax.tick_params(labelsize=25)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    _save_fig(fig, save_name)
    plt.show()


 ### Impulse Response Functions ###

def plot_irfs_comparison(gibbs_irfs, vi_irfs, country_names, variable_names, vi_label="VI", save_name=None):
    """Plot a grid of impulse response comparisons between Gibbs and a VI method.
    Saves to Figures/{save_name}.pdf if save_name is given."""
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

            ax.plot(horizons, gibbs_median[c, :, n], color="black", linewidth=1.2, label="Gibbs (median)")

            ax.plot(horizons, vi_median[c, :, n], color="red", linewidth=1.2, label=f"{vi_label} (median)")
            ax.plot(horizons, vi_p5[c, :, n], color="red", linestyle="--", linewidth=1, label=f"{vi_label} (5/95%)")
            ax.plot(horizons, vi_p95[c, :, n], color="red", linestyle="--", linewidth=1)

            ax.axhline(0, color="gray", linewidth=0.7, linestyle=":")

            if n == 0:
                ax.set_title(country_names[c], fontsize=20)
            if c == 0:
                ax.set_ylabel(variable_names[n], fontsize=20)
            if n == N - 1:
                ax.set_xlabel("Horizon", fontsize=15)
            ax.tick_params(labelsize=15)

    plt.tight_layout()
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fanchart_patch = Patch(color="tab:blue", alpha=0.3)

    ordered_handles = [fanchart_patch] + handles
    ordered_labels = ["Gibbs (5-95% fanchart)"] + labels

    fig.legend(ordered_handles, ordered_labels, loc="lower center", ncol=len(ordered_labels), fontsize=20,
               bbox_to_anchor=(0.5, -0.05))
    _save_fig(fig, save_name)
    plt.show()

def plot_wasserstein_grid_comparison(distances_dict, country_names, variable_names, save_name=None):
    """Plot a grid comparison of Wasserstein-distance curves for multiple VI
    methods against Gibbs.
    Saves to Figures/{save_name}.pdf if save_name is given."""
    method_names = _ordered_methods(distances_dict.keys())
    C, H_plus_1, N = next(iter(distances_dict.values())).shape
    horizons = np.arange(H_plus_1)

    fig, axes = plt.subplots(N, C, figsize=(3 * C, 2.2 * N), sharex=True)

    for n in range(N):
        for c in range(C):
            ax = axes[n, c]
            for method in method_names:
                ax.plot(horizons, distances_dict[method][c, :, n],
                         color=_method_colour(method), linewidth=2.2, label=method)
            ax.axhline(0, color="gray", linewidth=0.7, linestyle=":")

            if n == 0:
                ax.set_title(country_names[c], fontsize=20)
            if c == 0:
                ax.set_ylabel(variable_names[n], fontsize=20)
            if n == N - 1:
                ax.set_xlabel("Horizon", fontsize=15)
            ax.tick_params(labelsize=15)

    plt.tight_layout()
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), fontsize=20,
               bbox_to_anchor=(0.5, -0.05))
    _save_fig(fig, save_name)
    plt.show()

### Comparison to true parameters ###

def plot_lambda_intervals(results_by_seed, true_by_seed, methods, levels=(0.5, 0.8, 0.95), ax=None, save_name=None):
    """Plot, per seed and method, credible intervals for lambda at several nominal
    levels alongside the true data generating value.
    Saves to Figures/{save_name}.pdf if save_name is given."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    methods = _ordered_methods(methods)
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
                ax.plot([lo, hi], [y, y], color=_method_colour(method),
                        linewidth=linewidths[level], solid_capstyle="round",
                        label=METHOD_LABELS.get(method, method) if (s_idx == 0 and level == levels[0]) else None)

            ax.scatter([true_val], [y], color="black", marker="x", s=100, zorder=5)

    ax.set_xscale("log")
    ax.set_yticks([s_idx * (n_methods + 1) + n_methods / 2 for s_idx in range(len(seeds))])
    ax.set_yticklabels([f"seed {s}" for s in seeds], fontsize=18)
    for s_idx in range(len(seeds)):
        ax.axhline(s_idx * (n_methods + 1) - 0.5, color="grey", linewidth=0.5, alpha=0.5)
    ax.set_xlabel(r"$\lambda$ (log scale)", fontsize=22)
    ax.tick_params(axis="x", labelsize=18)

    method_handles, method_labels = ax.get_legend_handles_labels()
    level_handles = [Line2D([0], [0], color="black", linewidth=linewidths[level],
                             solid_capstyle="round", label=f"{int(level * 100)}% CI")
                      for level in levels]

    spacer = Line2D([0], [0], color="none", label="")
    all_handles = method_handles + [spacer] + level_handles
    ax.legend(handles=all_handles, fontsize=18, title_fontsize=19,
              loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0)

    ax.figure.tight_layout()
    _save_fig(ax.figure, save_name)
    return ax

### Model Diagnostics ###

def plot_diagnostics(ssvi_i_trace, ssvi_c_trace, ssvi_i_ess, ssvi_c_ess, rhat, title_suffix=""):
    """Plot final ULA trace, and ESS-across-iterations for SSVI-I and SSVI-C, alongside R-hat diagnostics Gibbs sampler."""
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

### Relationship between lambda and beta0 ###

def plot_lambda_beta0_correlation_scatter(gibbs_results, save_name=None):
    """Scatter of correlations between lambda and each
    component of beta_0, from Gibbs posterior draws.
    Saves to Figures/{save_name}.pdf if save_name is given."""
    lam = np.asarray(gibbs_results["lam"])
    beta_0 = np.asarray(gibbs_results["beta_0"])
    corrs = np.array([np.corrcoef(lam, beta_0[:, k])[0, 1] for k in range(beta_0.shape[1])])

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(range(len(corrs)), corrs, s=50, alpha=0.8)
    ax.axhline(0, color="gray", linewidth=1, linestyle="--")
    ax.set_xlabel(r"$\beta_0$ component index", fontsize=20)
    ax.set_ylabel(r"Correlation with $\lambda$", fontsize=20)
    ax.tick_params(labelsize=15)
    plt.tight_layout()
    _save_fig(fig, save_name)
    plt.show()
