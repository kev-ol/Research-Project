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
from figures import *
from figures import _save_fig  # not exported by `import *` (leading underscore)

# filename slugs for each VI method, used when saving figures
METHOD_SLUGS = {"mfvi": "mfvi", "ssvi_i": "ssvii", "ssvi_c": "ssvic"}

@dataclass
class PipelineConfig:
    """Configuration for one `run_pipeline` call: run identity, plot labels,
    and per-method hyperparameters.

    Attributes
    ----------
    name : str
        Run identifier and cache file stem (`{cache_dir}/{name}.pkl`).
    country_names : list of str
        Country labels in C-axis order.
    variable_names : list of str
        Endogenous variable labels in N-axis order.
    sign_pattern : tuple, optional
        Sign-identification pattern for IRF computation.
    ssvi_i_kwargs : dict, optional
        Kwargs forwarded to `run_ssvi_i`.
    ssvi_c_kwargs : dict, optional
        Kwargs forwarded to `run_ssvi_c`.
    gibbs_kwargs : dict, optional
        Kwargs forwarded to `run_gibbs`.
    n_samples : int, optional
        Posterior draws per method. Default is 10000.
    H : int, optional
        IRF horizon. Default is 36.
    """
    name: str
    country_names: list
    variable_names: list
    sign_pattern: tuple = ((2, 2, 1.0), (3, 2, -1.0), (2, 3, 1.0), (3, 3, 1.0))
    # method hyperparameters
    ssvi_i_kwargs: dict = field(default_factory=lambda: dict(n_steps=1000, s=0.1, n_burnin=100, epsilon=0.05))
    ssvi_c_kwargs: dict = field(default_factory=lambda: dict(n_steps=1000, s=0.5, n_burnin=100))
    gibbs_kwargs: dict = field(default_factory=lambda: dict(n_chains=4, n_steps=10000, n_burnin=2000))
    n_samples: int = 10000
    H: int = 36


def run_pipeline(Y, W, Z1, Z2, C, N, N_w, T, K, Z_width, L, L_w, L_z1, L_z2,
                  config: PipelineConfig, Lambda=None, cache_dir="cache", force_recompute=False, seed=None):
    """Run the full estimation pipeline (MFVI, SSVI-I, SSVI-C, Gibbs) on one
    dataset, compute comparison metrics, and cache results to disk.

    Parameters
    ----------
    Y, W, Z1, Z2 : numpy.ndarray
        Raw endogenous and exogenous panel data including leading lag periods.
    C, N, T, K, Z_width, L : int
        Model dimensions.
    L_w, L_z1, L_z2 : sequence of int
        Lags of W, Z1, Z2 included as regressors.
    config : PipelineConfig
        Run configuration.
    Lambda : numpy.ndarray of shape (C, N*K, N*K) or None, optional
        Pre-specified Minnesota-prior scale matrices. If None, built internally.
    cache_dir : str, optional
        Cache directory. Default is "cache".
    force_recompute : bool, optional
        Ignore existing cache and recompute. Default is False.
    seed : int or None, optional
        Top-level seed. Child seeds are spawned per stage for reproducibility.
        Cache is keyed by config.name only, not seed. Default is None.

    Returns
    -------
    dict
        Keys: 'config', 'C', 'cov_true', 'mfvi', 'ssvi_i', 'ssvi_c', 'gibbs',
        'runtime_total'. Each method dict contains 'results', 'samples', 'uqf',
        'faes', 'irfs', 'wasserstein', 'runtime'.
    """
    cache_path = Path(cache_dir) / f"{config.name}.pkl"
    # return saved data if applicable
    if cache_path.exists() and not force_recompute:
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    # preprocess data
    mfvi_pack, ssvi_i_pack, gibbs_pack = prep_data(
        Y, W, Z1, Z2, C, N, N_w, T, K, Z_width, L, L_w, L_z1, L_z2, Lambda
    )

    # independent child seeds for every stochastic stage, spawned once from
    # the top-level seed so the whole run is reproducible given (data, config, seed)
    (seed_mfvi, seed_ssvi_i, seed_ssvi_c, seed_gibbs,
     irf_seed_gibbs, irf_seed_mfvi, irf_seed_ssvi_i, irf_seed_ssvi_c) = np.random.SeedSequence(seed).spawn(8)
    rng_mfvi = np.random.default_rng(seed_mfvi)
    rng_ssvi_i = np.random.default_rng(seed_ssvi_i)
    rng_ssvi_c = np.random.default_rng(seed_ssvi_c)
    rng_gibbs = np.random.default_rng(seed_gibbs)

    # run mfvi and time it
    t0 = time.perf_counter()
    results_mfvi, ELBO_mfvi = run_mfvi(mfvi_pack, Z_width, C, N, K, T)
    mfvi_samples = sample_from_mfvi(results_mfvi, mfvi_pack, C, N, K, T, config.n_samples, rng=rng_mfvi)
    runtime_mfvi = time.perf_counter() - t0
    print(f"MFVI COMPLETE ({runtime_mfvi:.1f}s)")

    # run ssvi-i and time it
    t0 = time.perf_counter()
    results_ssvi_i, ELBO_ssvi_i, ess_i, log_lams_i = run_ssvi_i(ssvi_i_pack, Z_width, C, N, K, T, **config.ssvi_i_kwargs, rng=rng_ssvi_i)
    ssvi_i_samples = sample_from_ssvi_i(results_ssvi_i, ssvi_i_pack, C, N, K, T, config.n_samples, rng=rng_ssvi_i)
    runtime_ssvi_i = time.perf_counter() - t0
    print(f"SSVI-I COMPLETE ({runtime_ssvi_i:.1f}s)")

    # run ssvi-c and time it
    t0 = time.perf_counter()
    results_ssvi_c, ELBO_ssvi_c, ess_c, log_lams_c = run_ssvi_c(ssvi_i_pack, Z_width, C, N, K, T, **config.ssvi_c_kwargs, rng=rng_ssvi_c)
    ssvi_c_samples = sample_from_ssvi_c(results_ssvi_c, ssvi_i_pack, C, N, K, T, config.n_samples, rng=rng_ssvi_c)
    runtime_ssvi_c = time.perf_counter() - t0
    print(f"SSVI-C COMPLETE ({runtime_ssvi_c:.1f}s)")

    # run gibbs sampler and time it
    t0 = time.perf_counter()
    results_gibbs, ess, rhat = run_gibbs(gibbs_pack, C, N, K, Z_width, T, **config.gibbs_kwargs, rng=rng_gibbs)
    runtime_gibbs = time.perf_counter() - t0
    print(f"GIBBS COMPLETE ({runtime_gibbs:.1f}s)")

    # thin gibbs draws to config.n_samples for a consistent sample size
    # across all four methods. Every key except 'delta_c' is a numpy array;
    # 'delta_c' is a plain list of lists (see gibbs.run_gibbs), so it needs
    # list-style indexing instead of fancy indexing.
    rng_thin = np.random.default_rng(seed_gibbs)
    thin_idx = rng_thin.choice(len(results_gibbs["lam"]), size=config.n_samples, replace=False)
    results_gibbs = {
        key: val[thin_idx] if key != "delta_c" else [val[i] for i in thin_idx]
        for key, val in results_gibbs.items()
    }

    # extract delta_c covariances from gibbs and mfvi
    cov_true = compute_cov_true(results_gibbs, C)
    cov_mfvi = extract_cov_mfvi_pipeline(results_mfvi, mfvi_pack, C)

    # converting gibbs samples into arrays for accuracy metric
    gibbs_faes_arrays = prepare_gibbs_faes_arrays(results_gibbs)

    # gibbs IRFs, using the full (already-thinned) sample set directly
    irfs_gibbs, _ = compute_irfs(
        results_gibbs["beta_c"], results_gibbs["Sigma_c"],
        N=N, L=L, K=K, C=C, H=config.H, sign_pattern=config.sign_pattern, seed=irf_seed_gibbs,
    )

    # function for repeated construction of dictionaries across VI methods
    def build_method_dict(results_method, samples, cov, seed, elbo=None, diagnostics=None, runtime=None):
        """Assemble one VI method's entry in the pipeline results dict.

        Parameters
        ----------
        results_method : dict
            Raw params dict for this method.
        samples : dict
            Posterior samples for this method.
        cov : list of length C of numpy.ndarray
            Per-country delta_c covariance for UQF computation.
        seed : int or SeedSequence
            Seed for IRF sign-identification rotation.
        elbo : list of float or None, optional
            ELBO trace. Default is None.
        diagnostics : dict or None, optional
            Convergence diagnostics. Default is None.
        runtime : float or None, optional
            Estimation wall-clock time in seconds. Default is None.

        Returns
        -------
        dict
            Keys: 'results', 'samples', 'uqf', 'faes', 'irfs', 'wasserstein';
            plus 'elbo', 'diagnostics', 'runtime' if given.
        """
        beta = np.array(samples["beta_c"])
        sigma = np.array(samples["Sigma_c"])
        irfs_method, _ = compute_irfs(
            beta, sigma, N=N, L=L, K=K, C=C,
            H=config.H, sign_pattern=config.sign_pattern, seed=seed,
        )
        d = dict(
            results=results_method, samples=samples,
            uqf=[UQF(cov_true[c], cov[c]) for c in range(C)],
            faes=compute_faes_scores(samples, gibbs_faes_arrays),
            irfs=irfs_method,
            wasserstein=compute_wasserstein_curve(irfs_gibbs, irfs_method),
        )
        if elbo is not None:
            d["elbo"] = elbo
        if diagnostics is not None:
            d["diagnostics"] = diagnostics
        if runtime is not None:
            d["runtime"] = runtime
        return d

    # cconstruct results dictionary
    results = dict(
        config=config, C=C, cov_true=cov_true,
        mfvi=build_method_dict(
            results_mfvi, mfvi_samples, cov_mfvi, seed=irf_seed_mfvi, elbo=ELBO_mfvi,
            runtime=runtime_mfvi,
        ),
        ssvi_i=build_method_dict(
            results_ssvi_i, ssvi_i_samples, results_ssvi_i['cov_deltac'], seed=irf_seed_ssvi_i,
            elbo=ELBO_ssvi_i, diagnostics=dict(ess=ess_i, log_lam_history=log_lams_i),
            runtime=runtime_ssvi_i,
        ),
        ssvi_c=build_method_dict(
            results_ssvi_c, ssvi_c_samples, results_ssvi_c['cov_deltac'], seed=irf_seed_ssvi_c,
            elbo=ELBO_ssvi_c, diagnostics=dict(ess=ess_c, log_lam_history=log_lams_c),
            runtime=runtime_ssvi_c,
        ),
        gibbs=dict(
            results=results_gibbs,
            diagnostics=dict(rhat=rhat, ess=ess),
            irfs=irfs_gibbs,
            runtime=runtime_gibbs,
        )
    )
    # compute total runtime
    results["runtime_total"] = runtime_mfvi + runtime_ssvi_i + runtime_ssvi_c + runtime_gibbs

    # safely save data to disc
    cache_path.parent.mkdir(exist_ok=True)
    # dump to a tempopary file, then rename so cache_path is never left partially written
    tmp_path = cache_path.with_suffix(".tmp")
    with open(tmp_path, "wb") as f:
        pickle.dump(results, f)
    # replace whatever is there with new file
    tmp_path.replace(cache_path)
    return results


def plot_pipeline_results(results, N, K, Z_width, pct=0.25):
    """Produce all comparison plots and tables for one `run_pipeline` result."""
    config = results["config"]
    C = results["C"]
    methods = [("MFVI", "mfvi"), ("SSVI-I", "ssvi_i"), ("SSVI-C", "ssvi_c")]

    # plot accuracy boxplots
    for label, key in methods:
        plot_accuracy_boxplots(results[key]["faes"], label, C, save_name=f"accuracy-real-{METHOD_SLUGS[key]}")

    # plot IRFs
    for label, key in methods:
        plot_irfs_comparison(
            results["gibbs"]["irfs"], results[key]["irfs"],
            config.country_names, config.variable_names, vi_label=label,
            save_name=f"irf-real-{METHOD_SLUGS[key]}",
        )

    # plot wasserstein distances for IRFs
    wasserstein_labels = {"mfvi": "MFVI", "ssvi_i": "SSVI-I", "ssvi_c": "SSVI-C"}
    plot_wasserstein_grid_comparison(
        {wasserstein_labels[key]: results[key]["wasserstein"] for _, key in methods},
        config.country_names, config.variable_names,
        save_name="wasser-real",
    )

    # plot lambda marginal distributions
    lam_mfvi = results["mfvi"]["samples"]["lam"]
    lam_ssvi_i = results["ssvi_i"]["samples"]["lam"]
    lam_ssvi_c = results["ssvi_c"]["samples"]["lam"]
    lam_gibbs = results["gibbs"]["results"]["lam"]

    plt.rcParams.update({'font.size': 14})

    plt.figure(figsize=(8, 5))
    plt.hist(lam_mfvi, bins=100, density=True, alpha=0.5, label='MFVI', color=METHOD_COLOURS["mfvi"])
    plt.hist(lam_ssvi_i, bins=100, density=True, alpha=0.5, label='SSVI-I', color=METHOD_COLOURS["ssvi_i"])
    plt.hist(lam_ssvi_c, bins=100, density=True, alpha=0.5, label='SSVI-C', color=METHOD_COLOURS["ssvi_c"])
    plt.hist(lam_gibbs, bins=100, density=True, alpha=0.5, label='Gibbs', color=METHOD_COLOURS["gibbs"])
    # limit x-axis for readability
    plt.xlim(0, 0.0002)
    plt.xlabel(r'$\lambda$', fontsize=20)
    plt.ylabel('Density', fontsize=18)
    plt.legend(fontsize=18)
    plt.tick_params(labelsize=15)
    plt.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
    _save_fig(plt.gcf(), "lam-distr")
    plt.show()

    # print UQF table
    uqf_table = pd.DataFrame({
        method: results[method]["uqf"]
        for method in ["mfvi", "ssvi_i", "ssvi_c"]
    }, index=config.country_names if hasattr(results["mfvi"]["uqf"], "__len__") else None)
    print("UQF (Cov(delta_c) accuracy):")
    display(uqf_table.round(4))

    # print correlation MAE table
    print("Correlation structure MAE:")
    display(corr_mae_table(results, N, K, Z_width))

    # print runtime / iteration count table (VI methods only, Gibbs excluded)
    print("Runtime and iteration counts:")
    display(runtime_iteration_table(results))

    # print lambda dispersion-change tables (own range, then clipped to MFVI's range)
    delta_df = dispersion_change_real_data(results, pct=pct)
    print("Dispersion change (lambda effect on beta_c dispersion about beta_0):")
    display(summarise_dispersion_change(delta_df, group_cols=("method",)))

    delta_df_clipped = dispersion_change_real_data(results, pct=pct, clipped=True)
    print("Dispersion change (clipped to MFVI's lambda range):")
    display(summarise_dispersion_change(delta_df_clipped, group_cols=("method",)))

    plot_lambda_beta0_correlation_scatter(results['gibbs']['results'], save_name="beta0-lambda-corr")

    # algorithm diagnostics
    plot_diagnostics(
        results["ssvi_i"]["diagnostics"]["log_lam_history"][-1],
        results["ssvi_c"]["diagnostics"]["log_lam_history"][-1],
        results["ssvi_i"]["diagnostics"]["ess"],
        results["ssvi_c"]["diagnostics"]["ess"],
        results["gibbs"]["diagnostics"]["rhat"],
    )

def plot_pipeline_results_seed(results, true_params, N, K, Z_width, label="", scenario=None, pct=0.25):
    """Produce all pooled-across-seeds comparison plots and tables for a set of
    simulation results."""
    config = results[list(results.keys())[0]]["config"]
    C = results[list(results.keys())[0]]["C"]
    method_pairs = [("MFVI", "mfvi"), ("SSVI-I", "ssvi_i"), ("SSVI-C", "ssvi_c")]
    methods = ["mfvi", "ssvi_i", "ssvi_c", "gibbs"]

    # plot pooled accuracy
    for method_label, key in method_pairs:
        plot_accuracy_boxplots_pooled(
            results, method_label, key,
            save_name=f"accuracy-{scenario}-{METHOD_SLUGS[key]}" if scenario else None,
        )

    plot_lambda_accuracy_pooled(results, save_name=f"lam-accuracy-{scenario}" if scenario else None)

    # plot Wasserstein distances per seed
    wasserstein_labels = {"mfvi": "MFVI", "ssvi_i": "SSVI-I", "ssvi_c": "SSVI-C"}
    for seed in results:
        plot_wasserstein_grid_comparison(
            {wasserstein_labels[key]: results[seed][key]["wasserstein"] for _, key in method_pairs},
            config.country_names, config.variable_names,
        )

    # plot pooled UQF and MAE boxplots
    plot_corr_mae_boxplots(results, N, K, Z_width, save_name=f"mae-{scenario}" if scenario else None)
    plot_uqf_boxplot(results, save_name=f"uqf-{scenario}" if scenario else None)

    # coverage tables and barcharts for comparison to real truth
    coverage_betac = coverage_table(results, true_params, "beta_c", methods)
    print(f"beta_c coverage ({label}):")
    display(coverage_betac)

    coverage_gammac = coverage_table(results, true_params, "gamma_c", methods)
    print(f"gamma_c coverage ({label}):")
    display(coverage_gammac)

    coverage_beta0 = coverage_table(results, true_params, "beta_0", methods, axis_type="vector")
    print(f"beta_0 coverage ({label}):")
    display(coverage_beta0)

    # print mean runtime per method, averaged over seeds
    print(f"Mean runtime per method ({label}):")
    display(runtime_table_seed(results))

    # print lambda dispersion-change tables (own range, then clipped to MFVI's range)
    delta_df = dispersion_change_by_seed(results, pct=pct)
    print(f"Dispersion change (lambda effect on beta_c dispersion about beta_0) ({label}):")
    display(summarise_dispersion_change(delta_df, group_cols=("method",)))
    display(summarise_dispersion_change(delta_df, group_cols=("seed", "method")))

    delta_df_clipped = dispersion_change_by_seed(results, pct=pct, clipped=True)
    print(f"Dispersion change (clipped to MFVI's lambda range) ({label}):")
    display(summarise_dispersion_change(delta_df_clipped, group_cols=("method",)))

    ax = plot_lambda_intervals(
        results, true_params, methods,
        save_name=f"lam-calib-{scenario}" if scenario else None,
    )
    plt.show()

    # plot diagnostics, per seed
    for seed in results:
        plot_diagnostics(
            results[seed]["ssvi_i"]["diagnostics"]["log_lam_history"][-1],
            results[seed]["ssvi_c"]["diagnostics"]["log_lam_history"][-1],
            results[seed]["ssvi_i"]["diagnostics"]["ess"],
            results[seed]["ssvi_c"]["diagnostics"]["ess"],
            results[seed]["gibbs"]["diagnostics"]["rhat"],
            title_suffix=f" (seed {seed})",
        )
