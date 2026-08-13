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

@dataclass
class PipelineConfig:
    """Configuration for one `run_pipeline` run: run identity, plot labels,
    and per-method hyperparameters.

    Attributes
    ----------
    name : str
        Run identifier, used as the cache file's stem
        (`{cache_dir}/{name}.pkl`).
    country_names : list of str
        Country labels, in the same order as the C axis of the data.
    variable_names : list of str
        Endogenous variable labels, in the same order as the N axis of the data.
    sign_pattern : tuple of (int, int, float) triples, optional
        Sign-identification pattern passed to `results.compute_irfs`. Default
        is ``((2, 2, 1.0), (3, 2, -1.0), (2, 3, 1.0), (3, 3, 1.0))``.
    ssvi_i_kwargs : dict, optional
        Keyword arguments forwarded to `ssvi_i.run_ssvi_i`. Default is
        ``dict(n_steps=1000, s=0.1, n_burnin=100, epsilon=0.05)``.
    ssvi_c_kwargs : dict, optional
        Keyword arguments forwarded to `ssvi_c.run_ssvi_c`. Default is
        ``dict(n_steps=1000, s=0.1, n_burnin=100)``.
    gibbs_kwargs : dict, optional
        Keyword arguments forwarded to `gibbs.run_gibbs`. Default is
        ``dict(n_chains=4, n_steps=10000, n_burnin=2000)``.
    n_draws : int, optional
        Number of posterior draws used for IRF computation (per method).
        Default is 10000.
    H : int, optional
        IRF horizon passed to `results.compute_irfs`. Default is 36.
    """
    name: str
    country_names: list
    variable_names: list
    sign_pattern: tuple = ((2, 2, 1.0), (3, 2, -1.0), (2, 3, 1.0), (3, 3, 1.0))
    # method hyperparameters
    ssvi_i_kwargs: dict = field(default_factory=lambda: dict(n_steps=1000, s=0.1, n_burnin=100, epsilon=0.05))
    ssvi_c_kwargs: dict = field(default_factory=lambda: dict(n_steps=1000, s=0.1, n_burnin=100))
    gibbs_kwargs: dict = field(default_factory=lambda: dict(n_chains=4, n_steps=10000, n_burnin=2000))
    n_draws: int = 10000
    H: int = 36


def run_pipeline(Y, W, Z1, Z2, C, N, N_w, T, K, Z_width, L, L_w, L_z1, L_z2,
                  config: PipelineConfig, Lambda=None, cache_dir="cache", force_recompute=False, seed=None):
    """Run the full estimation pipeline (MFVI, SSVI-I, SSVI-C, Gibbs) on one
    dataset, compute comparison metrics (UQF, Faes accuracy, IRFs,
    Wasserstein distance), and cache the results to disk.

    Parameters
    ----------
    Y : numpy.ndarray of shape (C, T+L, N)
        Raw endogenous panel data, including the L extra leading periods
        needed to form lags.
    W : numpy.ndarray of shape (T+L, N_w)
        Raw exchangeable exogenous series, including the leading periods
        needed to form lags.
    Z1 : numpy.ndarray of shape (T+L, ...)
        Raw first non-exchangeable exogenous series.
    Z2 : numpy.ndarray of shape (T+L, ...)
        Raw second non-exchangeable exogenous series.
    C : int
        Number of countries.
    N : int
        Number of endogenous variables.
    N_w : int
        Number of exogenous W variables.
    T : int
        Number of usable (post-lag) time periods.
    K : int
        Total number of regressors per equation.
    Z_width : int
        Total number of non-exchangeable regressors per equation.
    L : int
        Number of endogenous lags.
    L_w : sequence of int
        Lags of W included as regressors.
    L_z1 : sequence of int
        Lags of Z1 included as regressors.
    L_z2 : sequence of int
        Lags of Z2 included as regressors.
    config : PipelineConfig
        Run configuration (name, labels, sign pattern, and per-method
        hyperparameters).
    Lambda : numpy.ndarray of shape (C, N*K, N*K) or None, optional
        Pre-specified Minnesota-prior scale matrices, forwarded to
        `data_prep.prep_data`. If None (default), built internally.
    cache_dir : str, optional
        Directory holding the pickled results cache. Default is "cache".
    force_recompute : bool, optional
        If True, ignore any existing cache file at
        `{cache_dir}/{config.name}.pkl` and recompute (overwriting it).
        Default is False.
    seed : int, numpy.random.SeedSequence, or None, optional
        Top-level seed controlling every stochastic stage of the pipeline
        (Gibbs sampling, the SSVI-I/SSVI-C Langevin chains, posterior-sample
        reconstruction, IRF sign-rotation search, and IRF draw
        subsampling). Independent child seeds are spawned from this value
        for each stage, so the same `seed` reproduces the pipeline's full
        output for identical inputs and `config`. If None (default), each
        stage draws fresh, non-reproducible entropy. Note the on-disk cache
        is keyed only by `config.name`, not by `seed` — reusing the same
        `name` with a different `seed` returns the cached result unless
        `force_recompute=True`.

    Returns
    -------
    dict
        Results dictionary with keys 'config' (PipelineConfig), 'C' (int),
        'cov_true' (list of length C of numpy.ndarray, the Gibbs-derived
        reference delta_c covariance per country), 'mfvi', 'ssvi_i', and
        'ssvi_c' (each a dict as built by the nested `build_method_dict`,
        with keys 'results', 'samples', 'uqf', 'faes', 'irfs',
        'wasserstein', 'runtime' (float, wall-clock seconds for that
        method's estimation step), and optionally 'elbo' and
        'diagnostics'), 'gibbs' (dict with keys 'results', 'diagnostics',
        'irfs', and 'runtime' (float)), and 'runtime_total' (float, the sum
        of the four methods' runtimes). If a cache file already exists and
        `force_recompute` is False, the cached dict is loaded and returned
        directly instead of being recomputed (so `runtime`/`runtime_total`
        reflect whenever it was originally computed, not the current call).
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
    (seed_mfvi, seed_ssvi_i, seed_ssvi_c, seed_gibbs, seed_subsample,
     irf_seed_gibbs, irf_seed_mfvi, irf_seed_ssvi_i, irf_seed_ssvi_c) = np.random.SeedSequence(seed).spawn(9)
    rng_mfvi = np.random.default_rng(seed_mfvi)
    rng_ssvi_i = np.random.default_rng(seed_ssvi_i)
    rng_ssvi_c = np.random.default_rng(seed_ssvi_c)
    rng_gibbs = np.random.default_rng(seed_gibbs)

    # run mfvi and time it
    t0 = time.perf_counter()
    results_mfvi, ELBO_mfvi = run_mfvi(mfvi_pack, Z_width, C, N, K, T)
    mfvi_samples = sample_from_mfvi(results_mfvi, mfvi_pack, C, N, K, T, rng=rng_mfvi)
    runtime_mfvi = time.perf_counter() - t0
    print(f"MFVI COMPLETE ({runtime_mfvi:.1f}s)")

    # run ssvi-i and time it
    t0 = time.perf_counter()
    results_ssvi_i, ELBO_ssvi_i, ess_i, log_lams_i = run_ssvi_i(ssvi_i_pack, Z_width, C, N, K, T, **config.ssvi_i_kwargs, rng=rng_ssvi_i)
    ssvi_i_samples = sample_from_ssvi_i(results_ssvi_i, ssvi_i_pack, C, N, K, T, rng=rng_ssvi_i)
    runtime_ssvi_i = time.perf_counter() - t0
    print(f"SSVI-I COMPLETE ({runtime_ssvi_i:.1f}s)")

    # run ssvi-c and time it
    t0 = time.perf_counter()
    results_ssvi_c, ELBO_ssvi_c, ess_c, log_lams_c = run_ssvi_c(ssvi_i_pack, Z_width, C, N, K, T, **config.ssvi_c_kwargs, rng=rng_ssvi_c)
    ssvi_c_samples = sample_from_ssvi_c(results_ssvi_c, ssvi_i_pack, C, N, K, T, rng=rng_ssvi_c)
    runtime_ssvi_c = time.perf_counter() - t0
    print(f"SSVI-C COMPLETE ({runtime_ssvi_c:.1f}s)")

    # run gibbs sampler and time it
    t0 = time.perf_counter()
    results_gibbs, ess, rhat = run_gibbs(gibbs_pack, C, N, K, Z_width, T, **config.gibbs_kwargs, rng=rng_gibbs)
    runtime_gibbs = time.perf_counter() - t0
    print(f"GIBBS COMPLETE ({runtime_gibbs:.1f}s)")

    # extract delta_c covariances from gibbs and mfvi
    cov_true = compute_cov_true(results_gibbs, C)
    cov_mfvi = extract_cov_mfvi_pipeline(results_mfvi, mfvi_pack, C)

    # converting gibbs samples into arrays for accuracy metric
    gibbs_faes_arrays = prepare_gibbs_faes_arrays(results_gibbs)

    rng = np.random.default_rng(seed_subsample)

    beta_gibbs = np.array(results_gibbs["beta_c"])
    sigma_gibbs = np.array(results_gibbs["Sigma_c"])
    # select 10,000 sample points for Gibbs IRFs
    idx_gibbs = rng.choice(beta_gibbs.shape[0], size=config.n_draws, replace=False)
    irfs_gibbs, _ = compute_irfs(
        beta_gibbs[idx_gibbs], sigma_gibbs[idx_gibbs],
        N=N, L=L, K=K, C=C, H=config.H, sign_pattern=config.sign_pattern, seed=irf_seed_gibbs,
    )

    # function for repeated construction of dictionaries across VI methods
    def build_method_dict(results_method, samples, cov, seed, elbo=None, diagnostics=None, runtime=None):
        """Assemble one VI method's entry in the pipeline results dict:
        posterior samples, UQF, Faes accuracy, IRFs, and Wasserstein
        distance against the Gibbs reference.

        Parameters
        ----------
        results_method : dict
            Raw output params dict for this method (e.g. the `params` dict
            returned by `mfvi.run_mfvi`, `ssvi_i.run_ssvi_i`, or
            `ssvi_c.run_ssvi_c`).
        samples : dict
            Posterior samples dict for this method, as returned by
            `results.sample_from_mfvi`, `results.sample_from_ssvi_i`, or
            `results.sample_from_ssvi_c`.
        cov : list of length C of numpy.ndarray
            Per-country delta_c covariance estimate for this method, used to
            compute the UQF against `cov_true`.
        seed : int or numpy.random.SeedSequence
            Seed passed to `results.compute_irfs` for this method's
            sign-identification rotation sampling — a child seed spawned
            from `run_pipeline`'s top-level `seed`.
        elbo : list of float or None, optional
            ELBO trace for this method; stored under key "elbo" if given.
            Default is None.
        diagnostics : dict or None, optional
            Convergence diagnostics for this method; stored under key
            "diagnostics" if given. Default is None.
        runtime : float or None, optional
            Wall-clock seconds this method's estimation step took (excluding
            downstream metric computation); stored under key "runtime" if
            given. Default is None.

        Returns
        -------
        dict
            Dictionary with keys 'results' (dict), 'samples' (dict), 'uqf'
            (list of length C of float), 'faes' (dict), 'irfs'
            (numpy.ndarray, shape (n_draws, C, H+1, N)), and 'wasserstein'
            (numpy.ndarray, shape (C, H+1, N)); plus 'elbo', 'diagnostics',
            and/or 'runtime' if the corresponding arguments were given.
        """
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


def plot_pipeline_results(results, N, K, Z_width):
    """Produce every comparison plot (boxplots, IRFs, Wasserstein grid, UQF, MAE_corr)
    for one run_pipeline result, ending with convergence diagnostics.

    Parameters
    ----------
    results : dict
        Output of `run_pipeline` for a single dataset.
    N : int
        Number of endogenous variables.
    K : int
        Number of regressors per equation.
    Z_width : int
        Number of non-exchangeable regressors per equation.

    Returns
    -------
    None
        Displays a sequence of matplotlib figures and printed/displayed
        tables; nothing is returned.
    """
    config = results["config"]
    C = results["C"]
    methods = [("MFVI", "mfvi"), ("SSVI-I", "ssvi_i"), ("SSVI-C", "ssvi_c")]

    # plot accuracy boxplots
    for label, key in methods:
        plot_accuracy_boxplots(results[key]["faes"], label, C)

    # plot IRFs
    for label, key in methods:
        plot_irfs_comparison(
            results["gibbs"]["irfs"], results[key]["irfs"],
            config.country_names, config.variable_names, vi_label=label,
        )

    # plot wasserstein distances for IRFs
    wasserstein_labels = {"mfvi": "mfvi", "ssvi_i": "SSVI_I", "ssvi_c": "SSVI_C"}
    plot_wasserstein_grid_comparison(
        {wasserstein_labels[key]: results[key]["wasserstein"] for _, key in methods},
        config.country_names, config.variable_names
    )

    # plot lambda marginal distributions
    lam_mfvi = results["mfvi"]["samples"]["lam"]
    lam_ssvi_i = results["ssvi_i"]["samples"]["lam"]
    lam_ssvi_c = results["ssvi_c"]["samples"]["lam"]
    lam_gibbs = results["gibbs"]["results"]["lam"]

    plt.figure(figsize=(8, 5))
    plt.hist(lam_ssvi_i, bins=50, density=True, alpha=0.5, label='SSVI_I')
    plt.hist(lam_ssvi_c, bins=50, density=True, alpha=0.5, label='SSVI_C')
    plt.hist(lam_mfvi, bins=50, density=True, alpha=0.5, label='mfvi')
    plt.hist(lam_gibbs, bins=50, density=True, alpha=0.5, label='Gibbs')
    # limit x-axis for readability
    plt.xlim(0, 0.0002)
    plt.xlabel('lambda')
    plt.ylabel('density')
    plt.legend()
    plt.title('SSVI-I vs MFVI vs Gibbs: posterior of lambda\n(mfvi peak truncated for visibility)')
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

    # algorithm diagnostics
    plot_diagnostics(
        results["ssvi_i"]["diagnostics"]["log_lam_history"][-1],
        results["ssvi_c"]["diagnostics"]["log_lam_history"][-1],
        results["ssvi_i"]["diagnostics"]["ess"],
        results["ssvi_c"]["diagnostics"]["ess"],
        results["gibbs"]["diagnostics"]["rhat"],
    )

def plot_pipeline_results_seed(results, true_params, N, K, Z_width, label=""):
    """Produce every pooled-across-seeds comparison plot (accuracy, Wasserstein,
    correlation MAE, UQF, coverage tables, lambda intervals) for a set of
    `run_pipeline` results keyed by seed, ending with per-seed convergence
    diagnostics.

    Parameters
    ----------
    results : dict
        Mapping from seed to a single-seed results dict, each the output of
        `run_pipeline`.
    true_params : dict
        Mapping from seed to the true simulating parameters for that seed
        (e.g. the `true_params` dict returned by `simulate.simulate_data`).
    N : int
        Number of endogenous variables.
    K : int
        Number of regressors per equation.
    Z_width : int
        Number of non-exchangeable regressors per equation.
    label : str, optional
        Text included in printed headers and the lambda-interval plot
        title. Default is "".

    Returns
    -------
    None
        Displays a sequence of matplotlib figures and printed/displayed
        tables; nothing is returned.
    """
    config = results[list(results.keys())[0]]["config"]
    C = results[list(results.keys())[0]]["C"]
    method_pairs = [("MFVI", "mfvi"), ("SSVI-I", "ssvi_i"), ("SSVI-C", "ssvi_c")]
    methods = ["mfvi", "ssvi_i", "ssvi_c", "gibbs"]

    # plot pooled accuracy
    for method_label, key in method_pairs:
        plot_accuracy_boxplots_pooled(results, method_label, key)

    # plot Wasserstein distances per seed
    wasserstein_labels = {"mfvi": "MFVI", "ssvi_i": "SSVI-I", "ssvi_c": "SSVI-C"}
    for seed in results:
        plot_wasserstein_grid_comparison(
            {wasserstein_labels[key]: results[seed][key]["wasserstein"] for _, key in method_pairs},
            config.country_names, config.variable_names,
        )

    # plot pooled UQF and MAE boxplots
    plot_corr_mae_boxplots(results, N, K, Z_width)
    plot_uqf_boxplot(results)

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

    ax = plot_lambda_intervals(results, true_params, methods)
    ax.set_title(rf"Credible intervals for $\lambda$ vs. true value — {label}")
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
