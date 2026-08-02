from dataclasses import dataclass, field
import pickle
from pathlib import Path
from IPython.display import display
import numpy as np
import matplotlib.pyplot as plt
import time
from mfvi import run_mfvi
from ssvi_i import run_ssvi_i
from ssvi_c import run_ssvi_c
from gibbs import run_gibbs
from data_prep import prep_data
from results import *

@dataclass
class PipelineConfig:
    name: str                      # used for cache filename + plot titles
    country_names: list
    variable_names: list
    sign_pattern: tuple = ((2, 2, 1.0), (3, 2, -1.0), (2, 3, 1.0), (3, 3, 1.0))
    # method hyperparams
    ssvi_i_kwargs: dict = field(default_factory=lambda: dict(n_steps=1000, s=0.1, n_burnin=100, epsilon=0.05))
    ssvi_c_kwargs: dict = field(default_factory=lambda: dict(n_steps=1000, s=0.1, n_burnin=100))
    gibbs_kwargs: dict = field(default_factory=lambda: dict(n_chains=4, n_steps=10000, n_burnin=2000))
    n_draws: int = 10000
    H: int = 36


def run_pipeline(Y, W, Z1, Z2, C, N, N_w, T, K, Z_width, L, L_w, L_z1, L_z2,
                  config: PipelineConfig, Lambda=None, cache_dir="cache", force_recompute=False):
    cache_path = Path(cache_dir) / f"{config.name}.pkl"
    if cache_path.exists() and not force_recompute:
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    mfvi_pack, ssvi_i_pack, gibbs_pack = prep_data(
        Y, W, Z1, Z2, C, N, N_w, T, K, Z_width, L, L_w, L_z1, L_z2, Lambda
    )

    t0 = time.perf_counter()
    results_mfvi, ELBO_mfvi = run_mfvi(mfvi_pack, Z_width, C, N, K, T)
    mfvi_samples = sample_from_mfvi(results_mfvi, mfvi_pack, C, N, K, T)
    print(f"MFVI COMPLETE ({time.perf_counter() - t0:.1f}s)")

    t0 = time.perf_counter()
    results_ssvi_i, ELBO_ssvi_i, ess_i, log_lams_i = run_ssvi_i(ssvi_i_pack, Z_width, C, N, K, T, **config.ssvi_i_kwargs)
    ssvi_i_samples = sample_from_ssvi_i(results_ssvi_i, ssvi_i_pack, C, N, K, T)
    print(f"SSVI-I COMPLETE ({time.perf_counter() - t0:.1f}s)")

    t0 = time.perf_counter()
    results_ssvi_c, ELBO_ssvi_c, ess_c, log_lams_c = run_ssvi_c(ssvi_i_pack, Z_width, C, N, K, T, **config.ssvi_c_kwargs)
    ssvi_c_samples = sample_from_ssvi_c(results_ssvi_c, ssvi_i_pack, C, N, K, T)
    print(f"SSVI-C COMPLETE ({time.perf_counter() - t0:.1f}s)")

    t0 = time.perf_counter()
    results_gibbs, ess, rhat = run_gibbs(gibbs_pack, C, N, K, Z_width, T, **config.gibbs_kwargs)
    print(f"GIBBS COMPLETE ({time.perf_counter() - t0:.1f}s)")

    cov_true = compute_cov_true(results_gibbs, C)
    cov_mfvi = extract_cov_mfvi_pipeline(results_mfvi, mfvi_pack, C)

    gibbs_faes_arrays = prepare_gibbs_faes_arrays(results_gibbs)

    rng = np.random.default_rng(0)

    beta_gibbs = np.array(results_gibbs["beta_c"])
    sigma_gibbs = np.array(results_gibbs["Sigma_c"])
    idx_gibbs = rng.choice(beta_gibbs.shape[0], size=config.n_draws, replace=False)
    irfs_gibbs, _ = compute_irfs(
        beta_gibbs[idx_gibbs], sigma_gibbs[idx_gibbs],
        N=N, L=L, K=K, C=C, H=config.H, sign_pattern=config.sign_pattern, seed=4,
    )

    def build_method_dict(results_method, samples, cov, seed, elbo=None, diagnostics=None):
        beta = np.array(samples["beta_c"])
        sigma = np.array(samples["Sigma_c"])
        idx = rng.choice(beta.shape[0], size=config.n_draws, replace=False)
        irfs_method, _ = compute_irfs(
            beta[idx], sigma[idx], N=N, L=L, K=K, C=C,
            H=config.H, sign_pattern=config.sign_pattern, seed=seed,
        )
        d = dict(
            results=results_method, samples=samples,
            uqf=compute_uqf(cov_true, cov, C),
            faes=compute_faes_scores(samples, gibbs_faes_arrays),
            irfs=irfs_method,
            wasserstein=compute_wasserstein_curve(irfs_gibbs, irfs_method),
        )
        if elbo is not None:
            d["elbo"] = elbo
        if diagnostics is not None:
            d["diagnostics"] = diagnostics
        return d

    results = dict(
        config=config, C=C, cov_true=cov_true,
        mfvi=build_method_dict(
            results_mfvi, mfvi_samples, cov_mfvi, seed=1, elbo=ELBO_mfvi,
        ),
        ssvi_i=build_method_dict(
            results_ssvi_i, ssvi_i_samples, results_ssvi_i['cov_deltac'], seed=2,
            elbo=ELBO_ssvi_i, diagnostics=dict(ess=ess_i, log_lam_history=log_lams_i),
        ),
        ssvi_c=build_method_dict(
            results_ssvi_c, ssvi_c_samples, results_ssvi_c['cov_deltac'], seed=3,
            elbo=ELBO_ssvi_c, diagnostics=dict(ess=ess_c, log_lam_history=log_lams_c),
        ),
        gibbs=dict(
            results=results_gibbs,
            diagnostics=dict(rhat=rhat, ess=ess),
            irfs=irfs_gibbs,
        )
    )

    cache_path.parent.mkdir(exist_ok=True)
    tmp_path = cache_path.with_suffix(".tmp")
    with open(tmp_path, "wb") as f:
        pickle.dump(results, f)
    tmp_path.replace(cache_path)
    return results


def plot_pipeline_results(results, N, K, Z_width):
    """Produce every comparison plot (boxplots, IRFs, Wasserstein grid, UQF, MAE_corr) 
    for one run_pipeline result, ending with convergence diagnostics."""
    config = results["config"]
    C = results["C"]
    methods = [("MFVI", "mfvi"), ("SSVI-I", "ssvi_i"), ("SSVI-C", "ssvi_c")]

    for label, key in methods:
        plot_accuracy_boxplots(results[key]["faes"], label, C)

    for label, key in methods:
        plot_irfs_comparison(
            results["gibbs"]["irfs"], results[key]["irfs"],
            config.country_names, config.variable_names, vi_label=label,
        )

    wasserstein_labels = {"mfvi": "mfvi", "ssvi_i": "SSVI_I", "ssvi_c": "SSVI_C"}
    plot_wasserstein_grid_comparison(
        {wasserstein_labels[key]: results[key]["wasserstein"] for _, key in methods},
        config.country_names, config.variable_names,
    )

    lam_mfvi = results["mfvi"]["samples"]["lam"]
    lam_ssvi_i = results["ssvi_i"]["samples"]["lam"]
    lam_ssvi_c = results["ssvi_c"]["samples"]["lam"]
    lam_gibbs = results["gibbs"]["results"]["lam"]

    plt.figure(figsize=(8, 5))
    plt.hist(lam_ssvi_i, bins=50, density=True, alpha=0.5, label='SSVI_I')
    plt.hist(lam_ssvi_c, bins=50, density=True, alpha=0.5, label='SSVI_C')
    plt.hist(lam_mfvi, bins=50, density=True, alpha=0.5, label='mfvi')
    plt.hist(lam_gibbs, bins=50, density=True, alpha=0.5, label='Gibbs')
    plt.xlim(0, 0.0002)
    plt.xlabel('lambda')
    plt.ylabel('density')
    plt.legend()
    plt.title('SSVI-I vs MFVI vs Gibbs: posterior of lambda\n(mfvi peak truncated for visibility)')
    plt.show()

    # --- UQF ---
    uqf_table = pd.DataFrame({
        method: results[method]["uqf"]
        for method in ["mfvi", "ssvi_i", "ssvi_c"]
    }, index=config.country_names if hasattr(results["mfvi"]["uqf"], "__len__") else None)
    print("UQF (Cov(delta_c) accuracy):")
    display(uqf_table.round(4))

    # --- Correlation-structure MAE ---
    print("Correlation structure MAE:")
    display(corr_mae_table(results, N, K, Z_width))

    # --- Diagnostics (last) ---
    plot_diagnostics(
        results["ssvi_i"]["diagnostics"]["log_lam_history"][-1],
        results["ssvi_c"]["diagnostics"]["log_lam_history"][-1],
        results["ssvi_i"]["diagnostics"]["ess"],
        results["ssvi_c"]["diagnostics"]["ess"],
        results["gibbs"]["diagnostics"]["rhat"],
    )

def plot_pipeline_results_seed(results, true_params, N, K, Z_width, label=""):
    config = results[list(results.keys())[0]]["config"]
    C = results[list(results.keys())[0]]["C"]
    method_pairs = [("MFVI", "mfvi"), ("SSVI-I", "ssvi_i"), ("SSVI-C", "ssvi_c")]
    methods = ["mfvi", "ssvi_i", "ssvi_c", "gibbs"]

    # --- pooled accuracy ---
    for method_label, key in method_pairs:
        plot_accuracy_boxplots_pooled(results, method_label, key)

    # --- Wasserstein, per seed ---
    wasserstein_labels = {"mfvi": "MFVI", "ssvi_i": "SSVI-I", "ssvi_c": "SSVI-C"}
    for seed in results:
        plot_wasserstein_grid_comparison(
            {wasserstein_labels[key]: results[seed][key]["wasserstein"] for _, key in method_pairs},
            config.country_names, config.variable_names,
        )

    # --- UQF and MAE_corr, pooled ---
    plot_corr_mae_boxplots(results, N, K, Z_width)
    plot_uqf_boxplot(results)

    # --- comparison to real truth ---
    coverage_betac = coverage_table(results, true_params, "beta_c", methods)
    print(f"beta_c coverage ({label}):")
    display(coverage_betac)

    coverage_gammac = coverage_table(results, true_params, "gamma_c", methods)
    print(f"gamma_c coverage ({label}):")
    display(coverage_gammac)

    coverage_beta0 = coverage_table(results, true_params, "beta_0", methods, axis_type="vector")
    print(f"beta_0 coverage ({label}):")
    display(coverage_beta0)

    ax = plot_lambda_intervals(results, true_params, methods)
    ax.set_title(rf"Credible intervals for $\lambda$ vs. true value — {label}")
    plt.show()

    # --- diagnostics, per seed, last ---
    for seed in results:
        plot_diagnostics(
            results[seed]["ssvi_i"]["diagnostics"]["log_lam_history"][-1],
            results[seed]["ssvi_c"]["diagnostics"]["log_lam_history"][-1],
            results[seed]["ssvi_i"]["diagnostics"]["ess"],
            results[seed]["ssvi_c"]["diagnostics"]["ess"],
            results[seed]["gibbs"]["diagnostics"]["rhat"],
            title_suffix=f" (seed {seed})",
        )