from dataclasses import dataclass, field
import pickle
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import arviz as az
from mfvi import run_mfvi
from ssvi_i import run_ssvi_i
from ssvi_c import run_ssvi_c
from gibbs import run_gibbs
from gibbs_og import run_gibbs_og
from data_prep import prep_data
from results import (
    sample_from_mfvi, sample_from_ssvi_i, sample_from_ssvi_c,
    compute_cov_true, extract_cov_mfvi, UQF, compute_uqf,
    prepare_gibbs_faes_arrays, compute_faes_scores, plot_accuracy_boxplots,
    compute_irfs, plot_irfs_comparison,
    compute_wasserstein_curve, plot_wasserstein_grid_comparison,
)

@dataclass
class PipelineConfig:
    name: str                      # used for cache filename + plot titles
    sign_pattern: tuple
    country_names: list
    variable_names: list
    # method hyperparams
    ssvi_i_kwargs: dict = field(default_factory=lambda: dict(n_steps=1000, step_size_init=0.01, s=0.2, n_burnin=100))
    ssvi_c_kwargs: dict = field(default_factory=lambda: dict(n_steps=1000, step_size_init=1, s=0.1, n_burnin=100))
    gibbs_kwargs: dict = field(default_factory=lambda: dict(n_chains=4, n_steps=10000, n_burnin=2000))
    n_draws: int = 10000
    H: int = 36


def run_pipeline(Y, W, Z1, Z2, C, N, N_w, T, K, Z_width, L, L_w, L_z1, L_z2,
                  config: PipelineConfig, cache_dir="cache", force_recompute=False):
    cache_path = Path(cache_dir) / f"{config.name}.pkl"
    if cache_path.exists() and not force_recompute:
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    mfvi_pack, ssvi_i_pack, gibbs_pack, gibbs_pack_og = prep_data(
        Y, W, Z1, Z2, C, N, N_w, T, K, Z_width, L, L_w, L_z1, L_z2,
    )

    results_mfvi, ELBO_mfvi = run_mfvi(mfvi_pack, Z_width, C, N, K, T)
    results_ssvi_i, ELBO_ssvi_i, ess_i, log_lams_i = run_ssvi_i(ssvi_i_pack, Z_width, C, N, K, T, **config.ssvi_i_kwargs)
    results_ssvi_c, ELBO_ssvi_c, ess_c, log_lams_c = run_ssvi_c(ssvi_i_pack, Z_width, C, N, K, T, **config.ssvi_c_kwargs)
    results_gibbs, ess, rhat = run_gibbs(gibbs_pack, C, N, K, Z_width, T, **config.gibbs_kwargs)

    cov_true = compute_cov_true(results_gibbs, C)
    cov_mfvi = extract_cov_mfvi(results_mfvi, mfvi_pack, C)

    mfvi_samples = sample_from_mfvi(results_mfvi, mfvi_pack, C, N, K, T)
    ssvi_i_samples = sample_from_ssvi_i(results_ssvi_i, ssvi_i_pack, C, N, K, T)
    ssvi_c_samples = sample_from_ssvi_c(results_ssvi_c, ssvi_i_pack, C, N, K, T)

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
    with open(cache_path, "wb") as f:
        pickle.dump(results, f)
    return results


def plot_pipeline_results(results):
    """Produce every comparison plot (boxplots, IRFs, Wasserstein grid) for one run_pipeline result."""
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
