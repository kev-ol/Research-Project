import numpy as np
from scipy.stats import invgamma, invwishart, gaussian_kde, wasserstein_distance
from scipy.linalg import eigh
from joblib import Parallel, delayed
import pandas as pd
from ssvi_i import calc_V_deltac, calc_mu_deltac
from ssvi_c import calc_V_beta02, calc_mu_beta02, calc_V_deltac2, calc_mu_deltac2


"""Posterior sample reconstruction from each method's VI output"""

def sample_from_mfvi(results_mfvi, mfvi_pack, C, N, K, T, n_samples=10000, rng=None):
    """Reconstruct posterior samples from the MFVI variational approximation.

    Parameters
    ----------
    results_mfvi : dict
        Output of `mfvi.run_mfvi`'s `params`, with (in order) keys
        'mu_delta' (numpy.ndarray, shape (size_delta,)), 'V_delta'
        (numpy.ndarray, shape (size_delta, size_delta)), 'v_bar' (float),
        's_bar' (float), and 'S_bar_sigma' (list of length C of
        numpy.ndarray, shape (N, N)).
    mfvi_pack : dict
        Data pack produced by `data_prep.prep_data`, must contain
        'idx_deltac' (list of length C of int) and 'size_deltac' (int).
    C : int
        Number of countries.
    N : int
        Number of endogenous variables.
    K : int
        Number of regressors per equation.
    T : int
        Number of time periods.
    n_samples : int, optional
        Number of posterior samples to draw. Default is 10000.
    rng : int, numpy.random.SeedSequence, numpy.random.Generator, or None, optional
        Source of randomness for the reconstructed posterior samples. If
        None (default), a fresh, non-reproducible generator is used.

    Returns
    -------
    dict
        Dictionary of reconstructed posterior samples with keys:
        'beta_0' (list of length n_samples of numpy.ndarray of shape
        (N*K,)), 'lam' (list of length n_samples of float), 'beta_c' (list
        of length n_samples of list of length C of numpy.ndarray of shape
        (N*K,)), 'gamma_c' (list of length n_samples of list of length C of
        numpy.ndarray of shape (N*Z_width,)), 'delta_c' (list of length
        n_samples of list of length C of numpy.ndarray of shape
        (size_deltac,)), and 'Sigma_c' (list of length n_samples of list of
        length C of numpy.ndarray of shape (N, N)).
    """
    rng = np.random.default_rng(rng)
    mu_delta, V_delta, v_bar, s_bar, S_bar_sigma = results_mfvi.values()

    idx_deltac = mfvi_pack["idx_deltac"]
    size_deltac = mfvi_pack["size_deltac"]

    # delta samples (beta_0, beta_c, gamma_c, delta_c all come from this)
    L = np.linalg.cholesky(V_delta)
    deltas = mu_delta + (L @ rng.normal(size=(len(mu_delta), n_samples))).T

    beta_0_samples = deltas[:, :idx_deltac[0]] 

    beta_c_samples = [deltas[:, idx_deltac[c]:idx_deltac[c] + N*K] for c in range(C)]  
    gamma_c_samples = [deltas[:, idx_deltac[c] + N*K:idx_deltac[c] + size_deltac] for c in range(C)] 
    delta_c_samples = [deltas[:, idx_deltac[c]:idx_deltac[c] + size_deltac] for c in range(C)]    

    # lambda samples (cheap, vectorize trivially)
    lam_samples = invgamma.rvs(s_bar/2, scale=v_bar/2, size=n_samples, random_state=rng)

    # Sigma_c samples (vectorized per country via scipy's size argument)
    Sigma_c_samples = [invwishart.rvs(T, S_bar_sigma[c], size=n_samples, random_state=rng) for c in range(C)]  

    return {
        'beta_0': list(beta_0_samples),
        'lam': list(lam_samples),
        'beta_c': [[beta_c_samples[c][n] for c in range(C)] for n in range(n_samples)],
        'gamma_c': [[gamma_c_samples[c][n] for c in range(C)] for n in range(n_samples)],
        'delta_c': [[delta_c_samples[c][n] for c in range(C)] for n in range(n_samples)],
        'Sigma_c': [[Sigma_c_samples[c][n] for c in range(C)] for n in range(n_samples)],
    }


def sample_from_ssvi_i(results_ssvi_i, ssvi_i_pack, C, N, K, T, n_samples=10000, rng=None):
    """Reconstruct posterior samples from the SSVI-I variational approximation.

    Parameters
    ----------
    results_ssvi_i : dict
        Output of `ssvi_i.run_ssvi_i`'s `params`, with (in order) keys
        'mu_beta0' (numpy.ndarray, shape (N*K,)), 'V_beta0' (numpy.ndarray,
        shape (N*K, N*K)), 'q_lambda' (numpy.ndarray, shape (n_chain,), the
        converged lambda ULA chain), 'S_bar_sigma' (list of length C of
        numpy.ndarray, shape (N, N)), and 'cov_deltac' (list of length C of
        numpy.ndarray, shape (size_deltac, size_deltac)).
    ssvi_i_pack : dict
        Data pack produced by `data_prep.prep_data`, must contain (in order)
        'Y' (numpy.ndarray, shape (C, T, N)), 'F' (sequence of length C of
        numpy.ndarray, shape (T, K+Z_width)), 'FF' (sequence of length C of
        numpy.ndarray, shape (K+Z_width, K+Z_width)), 'idx_deltac' (list of
        int), 'size_deltac' (int), 'Pc' (numpy.ndarray, shape
        (size_deltac, size_deltac)), 'Lambda_inv' (list of length C of
        numpy.ndarray, shape (N*K, N*K)), and 'Lambda_inv_sum'
        (numpy.ndarray, shape (N*K, N*K)).
    C : int
        Number of countries.
    N : int
        Number of endogenous variables.
    K : int
        Number of regressors per equation.
    T : int
        Number of time periods.
    n_samples : int, optional
        Number of posterior samples to draw. Default is 10000.
    rng : int, numpy.random.SeedSequence, numpy.random.Generator, or None, optional
        Source of randomness for the reconstructed posterior samples
        (including the lambda-KDE resampling). If None (default), a fresh,
        non-reproducible generator is used.

    Returns
    -------
    dict
        Dictionary of reconstructed posterior samples with keys: 'beta_0'
        (list of length n_samples of numpy.ndarray of shape (N*K,)), 'lam'
        (list of length n_samples of float), 'beta_c' (list of length
        n_samples of numpy.ndarray of shape (C, N*K)), 'gamma_c' (list of
        length n_samples of numpy.ndarray of shape (C, N*Z_width)),
        'delta_c' (list of length n_samples of numpy.ndarray of shape
        (C, size_deltac)), and 'Sigma_c' (list of length n_samples of
        numpy.ndarray of shape (C, N, N)).
    """
    rng = np.random.default_rng(rng)
    mu_beta0, V_beta0, q_lambda_chain, S_bar_sigma, cov_deltac = results_ssvi_i.values()
    mu_sigma_inv = [T * np.linalg.inv(S_bar_sigma[c]) for c in range(C)]
    Y, F, FF, idx_deltac, size_deltac, Pc, Lambda_inv, Lambda_inv_sum = ssvi_i_pack.values()

    # lambda: sample from a KDE fit to the converged ULA chain (log-space, since lambda > 0)
    lam_chain = np.asarray(q_lambda_chain)
    kde_lam = gaussian_kde(np.log(lam_chain))
    lam_samples = np.exp(kde_lam.resample(n_samples, seed=rng).flatten())

    # beta_0: independent draw, paired with lam_samples by index
    L_beta0 = np.linalg.cholesky(V_beta0)
    beta_0_samples = mu_beta0 + (L_beta0 @ rng.normal(size=(len(mu_beta0), n_samples))).T 

    # V_deltac, mu_deltac: both batched over the same (lam, beta0) pairs
    V_deltac = calc_V_deltac(lam_samples, mu_sigma_inv, FF, Lambda_inv, size_deltac, Pc, C, N, K)
    mu_deltac = calc_mu_deltac(lam_samples, beta_0_samples, V_deltac, mu_sigma_inv, Y, F, Lambda_inv, size_deltac, Pc, C, N, K)

    delta_c_samples_arr = np.empty((n_samples, C, size_deltac))
    for c in range(C):
        L_c = np.linalg.cholesky(V_deltac[c])          
        z = rng.normal(size=(n_samples, size_deltac))
        delta_c_samples_arr[:, c, :] = mu_deltac[c] + np.einsum('nij,nj->ni', L_c, z)

    beta_c_samples_arr = delta_c_samples_arr[:, :, :N*K]
    gamma_c_samples_arr = delta_c_samples_arr[:, :, N*K:]

    Sigma_c_samples = np.stack(
        [invwishart.rvs(T, S_bar_sigma[c], size=n_samples, random_state=rng) for c in range(C)], axis=1
    )

    return {
        'beta_0': list(beta_0_samples),
        'lam': list(lam_samples),
        'beta_c': list(beta_c_samples_arr),
        'gamma_c': list(gamma_c_samples_arr),
        'delta_c': list(delta_c_samples_arr),
        'Sigma_c': list(Sigma_c_samples),
    }


def sample_from_ssvi_c(results_ssvi_c, ssvi_i_pack, C, N, K, T, n_samples=10000, rng=None):
    """Reconstruct posterior samples from the SSVI-C variational approximation.

    Parameters
    ----------
    results_ssvi_c : dict
        Output of `ssvi_c.run_ssvi_c`'s `params`, with (in order) keys
        'q_lambda' (numpy.ndarray, shape (n_chain,), the converged lambda
        ULA chain), 'S_bar_sigma' (list of length C of numpy.ndarray, shape
        (N, N)), and 'cov_deltac' (list of length C of numpy.ndarray, shape
        (size_deltac, size_deltac)).
    ssvi_i_pack : dict
        Data pack produced by `data_prep.prep_data`, must contain (in order)
        'Y' (numpy.ndarray, shape (C, T, N)), 'F' (sequence of length C of
        numpy.ndarray, shape (T, K+Z_width)), 'FF' (sequence of length C of
        numpy.ndarray, shape (K+Z_width, K+Z_width)), 'idx_deltac' (list of
        int), 'size_deltac' (int), 'Pc' (numpy.ndarray, shape
        (size_deltac, size_deltac)), 'Lambda_inv' (list of length C of
        numpy.ndarray, shape (N*K, N*K)), and 'Lambda_inv_sum'
        (numpy.ndarray, shape (N*K, N*K)).
    C : int
        Number of countries.
    N : int
        Number of endogenous variables.
    K : int
        Number of regressors per equation.
    T : int
        Number of time periods.
    n_samples : int, optional
        Number of posterior samples to draw. Default is 10000.
    rng : int, numpy.random.SeedSequence, numpy.random.Generator, or None, optional
        Source of randomness for the reconstructed posterior samples
        (including the lambda-KDE resampling). If None (default), a fresh,
        non-reproducible generator is used.

    Returns
    -------
    dict
        Dictionary of reconstructed posterior samples with keys: 'beta_0'
        (list of length n_samples of numpy.ndarray of shape (N*K,)), 'lam'
        (list of length n_samples of float), 'beta_c' (list of length
        n_samples of numpy.ndarray of shape (C, N*K)), 'gamma_c' (list of
        length n_samples of numpy.ndarray of shape (C, N*Z_width)),
        'delta_c' (list of length n_samples of numpy.ndarray of shape
        (C, size_deltac)), and 'Sigma_c' (list of length n_samples of
        numpy.ndarray of shape (C, N, N)).
    """
    rng = np.random.default_rng(rng)
    q_lambda_chain, S_bar_sigma, cov_deltac = results_ssvi_c.values()
    mu_sigma_inv = [T * np.linalg.inv(S_bar_sigma[c]) for c in range(C)]
    Y, F, FF, idx_deltac, size_deltac, Pc, Lambda_inv, Lambda_inv_sum = ssvi_i_pack.values()

    # lambda: sample from a KDE fit to the converged ULA chain (log-space, since lambda > 0)
    lam_chain = np.asarray(q_lambda_chain)
    kde_lam = gaussian_kde(np.log(lam_chain))
    lam_samples = np.exp(kde_lam.resample(n_samples, seed=rng).flatten())

    # V_deltac(lambda), computed once and reused for both beta0 and delta_c
    V_deltac = calc_V_deltac2(lam_samples, mu_sigma_inv, FF, Lambda_inv, size_deltac, Pc, C, N, K)

    # beta_0 | lambda
    V_beta0 = calc_V_beta02(lam_samples, V_deltac, Lambda_inv, Lambda_inv_sum, C, N, K)
    mu_beta0 = calc_mu_beta02(lam_samples, V_deltac, mu_sigma_inv, V_beta0, Y, F, Lambda_inv, Pc, C, N, K)

    size_beta0 = mu_beta0.shape[-1]
    L_beta0 = np.linalg.cholesky(V_beta0)
    z_beta0 = rng.normal(size=(n_samples, size_beta0))
    beta_0_samples = mu_beta0 + np.einsum('nij,nj->ni', L_beta0, z_beta0)

    # delta_c | lambda, beta0
    mu_deltac = calc_mu_deltac2(lam_samples, beta_0_samples, V_deltac, mu_sigma_inv, Y, F, Lambda_inv, size_deltac, Pc, C, N, K)

    delta_c_samples_arr = np.empty((n_samples, C, size_deltac))
    for c in range(C):
        L_c = np.linalg.cholesky(V_deltac[c])
        z = rng.normal(size=(n_samples, size_deltac))
        delta_c_samples_arr[:, c, :] = mu_deltac[c] + np.einsum('nij,nj->ni', L_c, z)

    beta_c_samples_arr = delta_c_samples_arr[:, :, :N*K]
    gamma_c_samples_arr = delta_c_samples_arr[:, :, N*K:]

    Sigma_c_samples = np.stack(
        [invwishart.rvs(T, S_bar_sigma[c], size=n_samples, random_state=rng) for c in range(C)], axis=1
    )

    return {
        'beta_0': list(beta_0_samples),
        'lam': list(lam_samples),
        'beta_c': list(beta_c_samples_arr),
        'gamma_c': list(gamma_c_samples_arr),
        'delta_c': list(delta_c_samples_arr),
        'Sigma_c': list(Sigma_c_samples),
    }


"""UQF (Uncertainty Quantification Factor)"""

def compute_cov_true(results_gibbs, C):
    """Empirical delta_c covariance from Gibbs draws, treated as ground truth.

    Parameters
    ----------
    results_gibbs : dict
        Output of `gibbs.run_gibbs`'s `post_burnin_samples`, must contain key
        'delta_c' (list of length n_draws of list of length C of
        numpy.ndarray of shape (size_deltac,)).
    C : int
        Number of countries.

    Returns
    -------
    list of length C of numpy.ndarray of shape (size_deltac, size_deltac)
        Empirical covariance of the delta_c draws, per country.
    """
    cov_true = []
    for c in range(C):
        delta_c_draws = np.array([results_gibbs["delta_c"][t][c] for t in range(len(results_gibbs["delta_c"]))])
        cov_true.append(np.cov(delta_c_draws.T))
    return cov_true


def extract_cov_mfvi_pipeline(results_mfvi, mfvi_pack, C):
    """Per-country delta_c covariance block sliced out of MFVI's full V_delta (used within pipeline).

    Parameters
    ----------
    results_mfvi : dict
        Output of `mfvi.run_mfvi`'s `params`, must contain key 'V_delta'
        (numpy.ndarray, shape (size_delta, size_delta)).
    mfvi_pack : dict
        Data pack produced by `data_prep.prep_data`, must contain
        'idx_deltac' (list of length C of int) and 'size_deltac' (int).
    C : int
        Number of countries.

    Returns
    -------
    list of length C of numpy.ndarray of shape (size_deltac, size_deltac)
        Per-country delta_c covariance blocks sliced from V_delta.
    """
    idx_deltac = mfvi_pack["idx_deltac"]
    size_deltac = mfvi_pack["size_deltac"]
    V_delta = results_mfvi["V_delta"]

    cov_mfvi = []
    for c in range(C):
        start = idx_deltac[c]
        cov_mfvi.append(V_delta[start:start + size_deltac, start:start + size_deltac])
    return cov_mfvi

def cov2corr(cov):
    """Convert a covariance matrix to a correlation matrix.

    Parameters
    ----------
    cov : numpy.ndarray of shape (D, D)
        Covariance matrix.

    Returns
    -------
    numpy.ndarray of shape (D, D)
        Corresponding correlation matrix.
    """
    d = np.sqrt(np.diag(cov))
    d[d == 0] = 1e-12
    return cov / np.outer(d, d)

def extract_cov_mfvi(V_delta, N, K, Z_width, C):
    """Slice per-country delta_c covariance blocks directly out of MFVI's full
    V_delta, without needing the full `mfvi_pack` (used outside of pipeline).

    Parameters
    ----------
    V_delta : numpy.ndarray of shape (size_delta, size_delta)
        Full joint covariance matrix over [beta_0, delta_1, ..., delta_C],
        as returned by `mfvi.calc_V_delta_naive`.
    N : int
        Number of endogenous variables.
    K : int
        Number of regressors per equation.
    Z_width : int
        Number of non-exchangeable regressors per equation.
    C : int
        Number of countries.

    Returns
    -------
    list of length C of numpy.ndarray of shape (N*K + N*Z_width, N*K + N*Z_width)
        Per-country delta_c covariance blocks.
    """
    size_beta0 = N * K
    size_deltac = N * K + N * Z_width
    idx_deltac = [size_beta0 + c * size_deltac for c in range(C)]
    return [V_delta[idx_deltac[c]:idx_deltac[c]+size_deltac,
                     idx_deltac[c]:idx_deltac[c]+size_deltac] for c in range(C)]


def UQF(cov_true, cov_est):
    """Compute the Uncertainty Quantification Factor (UQF) between a reference
    covariance matrix and an estimated covariance matrix.

    Parameters
    ----------
    cov_true : numpy.ndarray of shape (D, D)
        Reference ("true") covariance matrix, e.g. from Gibbs draws.
    cov_est : numpy.ndarray of shape (D, D)
        Estimated covariance matrix from a variational method.

    Returns
    -------
    float
        UQF, defined as 1 / (largest generalized eigenvalue of the pencil
        (cov_true, cov_est)).
    """
    eigenvalues = eigh(cov_true, cov_est, eigvals_only=True)
    return 1 / np.max(eigenvalues)


def compute_uqf(cov_true, cov_est_list, C):
    """Compute the Uncertainty Quantification Factor (UQF) for every country.

    Parameters
    ----------
    cov_true : list of length C of numpy.ndarray of shape (D, D)
        Reference ("true") covariance matrix per country.
    cov_est_list : list of length C of numpy.ndarray of shape (D, D)
        Estimated covariance matrix per country.
    C : int
        Number of countries.

    Returns
    -------
    list of length C of float
        UQF value for each country.
    """
    return [UQF(cov_true[c], cov_est_list[c]) for c in range(C)]

"""Correlation Mean Absolute Error (MAE)"""

def corr_mae_table(results, N, K, Z_width):
    """Compute, per country, the mean absolute correlation error of each VI
    method's delta_c correlation structure against Gibbs.

    Parameters
    ----------
    results : dict
        Single-seed results dict, with keys "C" (int), "mfvi" (dict with
        "results" sub-dict containing "V_delta"), "ssvi_i" (dict with
        "results" sub-dict containing "cov_deltac"), "ssvi_c" (dict with
        "results" sub-dict containing "cov_deltac"), "cov_true" (list of
        length C of numpy.ndarray, the reference covariance per country),
        and "config" (object with attribute `country_names`, a sequence of
        length C of str).
    N : int
        Number of endogenous variables.
    K : int
        Number of regressors per equation.
    Z_width : int
        Number of non-exchangeable regressors per equation.

    Returns
    -------
    pandas.DataFrame
        Rows indexed by country name, columns ["MFVI", "SSVI-I", "SSVI-C"],
        values the mean absolute correlation error vs Gibbs, rounded to 4
        decimal places.
    """
    C = results["C"]
    method_names = ["MFVI", "SSVI-I", "SSVI-C"]
    V_delta = results["mfvi"]["results"]["V_delta"]
    cov_mfvi = extract_cov_mfvi(V_delta, N, K, Z_width, C)
    cov_ssvi_i = results["ssvi_i"]["results"]["cov_deltac"]
    cov_ssvi_c = results["ssvi_c"]["results"]["cov_deltac"]
    cov_true = results["cov_true"]
    methods = {"MFVI": cov_mfvi, "SSVI-I": cov_ssvi_i, "SSVI-C": cov_ssvi_c}

    country_names = results["config"].country_names
    table = {name: [] for name in method_names}

    for c in range(C):
        corr_true = cov2corr(cov_true[c])
        np.fill_diagonal(corr_true, np.nan)
        for name, cov in methods.items():
            corr_method = cov2corr(cov[c])
            np.fill_diagonal(corr_method, np.nan)
            diff = corr_method - corr_true
            table[name].append(np.nanmean(np.abs(diff)))

    return pd.DataFrame(table, index=country_names).round(4)

"""Accuracy Measure (Faes et al. 2011)"""

def faes_accuracy(vi_samples, gibbs_samples, positive_support=False, grid_size=500):
    """Compute the Faes et al. (2011) accuracy score between a VI method's
    approximate posterior samples and Gibbs (reference) posterior samples for
    a single scalar parameter.

    Parameters
    ----------
    vi_samples : array_like of shape (T_vi,)
        1-d samples from a VI method's approximate posterior.
    gibbs_samples : array_like of shape (T_gibbs,)
        1-d samples from the Gibbs (reference) posterior for the same
        parameter.
    positive_support : bool, optional
        If True, both sample sets are log-transformed before density
        estimation (for parameters with positive support, e.g. lambda or
        Sigma_c diagonals). Default is False.
    grid_size : int, optional
        Number of grid points used to numerically integrate the density
        overlap. Default is 500.

    Returns
    -------
    float
        Faes et al. (2011) accuracy score, in percent (100 = perfect overlap
        of the two KDE-estimated densities).
    """
    vi_samples = np.asarray(vi_samples)
    gibbs_samples = np.asarray(gibbs_samples)

    if positive_support:
        vi_samples = np.log(vi_samples)
        gibbs_samples = np.log(gibbs_samples)

    kde_q = gaussian_kde(vi_samples)
    kde_p = gaussian_kde(gibbs_samples)

    # Bounds per paper: smallest/largest value across sample sets, no padding
    lo = min(vi_samples.min(), gibbs_samples.min())
    hi = max(vi_samples.max(), gibbs_samples.max())
    grid = np.linspace(lo, hi, grid_size)

    iae = np.trapezoid(np.abs(kde_q(grid) - kde_p(grid)), grid)
    return 100 * (1 - 0.5 * iae)


def _faes_grid(vi_arr, gibbs_arr, positive_support=False, n_jobs=-1):
    """Compute Faes accuracy scores for a (country, dimension) grid of scalar
    parameters, in parallel.

    Parameters
    ----------
    vi_arr : numpy.ndarray of shape (T, C, D)
        VI method draws x countries x scalar-dim.
    gibbs_arr : numpy.ndarray of shape (T, C, D)
        Gibbs draws x countries x scalar-dim.
    positive_support : bool, optional
        Passed through to `faes_accuracy`. Default is False.
    n_jobs : int, optional
        Number of parallel jobs passed to `joblib.Parallel`. Default is -1
        (use all available cores).

    Returns
    -------
    numpy.ndarray of shape (C, D)
        Faes accuracy score for each (country, dimension) pair.
    """
    T, C, D = vi_arr.shape
    pairs = [(c, d) for c in range(C) for d in range(D)]

    results = Parallel(n_jobs=n_jobs)(
        delayed(faes_accuracy)(vi_arr[:, c, d], gibbs_arr[:, c, d], positive_support)
        for c, d in pairs
    )
    out = np.empty((C, D))
    for (c, d), val in zip(pairs, results):
        out[c, d] = val
    return out


def _faes_vec(vi_arr, gibbs_arr, positive_support=False, n_jobs=-1):
    """Compute Faes accuracy scores for a vector of scalar parameters (no
    country axis), in parallel.

    Parameters
    ----------
    vi_arr : numpy.ndarray of shape (T, D)
        VI method draws x scalar-dim.
    gibbs_arr : numpy.ndarray of shape (T, D)
        Gibbs draws x scalar-dim.
    positive_support : bool, optional
        Passed through to `faes_accuracy`. Default is False.
    n_jobs : int, optional
        Number of parallel jobs passed to `joblib.Parallel`. Default is -1
        (use all available cores).

    Returns
    -------
    numpy.ndarray of shape (D,)
        Faes accuracy score for each dimension.
    """
    T, D = vi_arr.shape
    results = Parallel(n_jobs=n_jobs)(
        delayed(faes_accuracy)(vi_arr[:, d], gibbs_arr[:, d], positive_support)
        for d in range(D)
    )
    return np.array(results)


def prepare_gibbs_faes_arrays(gibbs_samples):
    """Convert Gibbs sample lists to arrays once, for reuse across every VI
    method's Faes scoring.

    Parameters
    ----------
    gibbs_samples : dict
        Output of `gibbs.run_gibbs`'s `post_burnin_samples`, must contain
        keys 'beta_c' (array_like, shape (T, C, N*K)), 'gamma_c' (array_like,
        shape (T, C, N*Z_width)), 'beta_0' (array_like, shape (T, N*K)),
        'Sigma_c' (array_like, shape (T, C, N, N)), and 'lam' (array_like,
        shape (T,)).

    Returns
    -------
    dict
        Dictionary with keys 'beta_c' (numpy.ndarray, shape (T, C, N*K)),
        'gamma_c' (numpy.ndarray, shape (T, C, N*Z_width)), 'beta_0'
        (numpy.ndarray, shape (T, N*K)), 'Sigma_c' (numpy.ndarray, shape
        (T, C, N), the diagonal entries of Sigma_c), and 'lam' (array_like,
        shape (T,), passed through unchanged).
    """
    beta_c_gibbs = np.array(gibbs_samples['beta_c'])    
    gamma_c_gibbs = np.array(gibbs_samples['gamma_c']) 
    beta_0_gibbs = np.array(gibbs_samples['beta_0'])        

    # Sigma_c diagonals: extract diag once, vectorized, before parallel calls
    Sigma_c_gibbs_full = np.array(gibbs_samples['Sigma_c'])  
    Sigma_c_gibbs = np.diagonal(Sigma_c_gibbs_full, axis1=2, axis2=3)

    return {
        'beta_c': beta_c_gibbs,
        'gamma_c': gamma_c_gibbs,
        'beta_0': beta_0_gibbs,
        'Sigma_c': Sigma_c_gibbs,
        'lam': gibbs_samples['lam'],
    }


def compute_faes_scores(vi_samples, gibbs_arrays):
    """Faes accuracy of a VI method's samples against Gibbs, for every
    parameter block.

    Parameters
    ----------
    vi_samples : dict
        Samples dict for one VI method, as returned by `sample_from_mfvi`,
        `sample_from_ssvi_i`, or `sample_from_ssvi_c`, with keys 'beta_c',
        'gamma_c', 'beta_0', 'Sigma_c', and 'lam'.
    gibbs_arrays : dict
        Output of `prepare_gibbs_faes_arrays`, computed once and shared
        across methods so the Gibbs conversion isn't redone per method.

    Returns
    -------
    dict
        Dictionary with keys 'beta_c' (numpy.ndarray, shape (C, N*K)),
        'gamma_c' (numpy.ndarray, shape (C, N*Z_width)), 'beta_0'
        (numpy.ndarray, shape (N*K,)), 'lam' (float), and 'Sigma_c'
        (numpy.ndarray, shape (C, N)) — the Faes accuracy score for each
        parameter component.
    """
    beta_c_vi = np.array(vi_samples['beta_c'])         
    gamma_c_vi = np.array(vi_samples['gamma_c'])    
    beta_0_vi = np.array(vi_samples['beta_0']) 

    # Sigma_c diagonals: extract diag once, vectorized, before parallel calls
    Sigma_c_vi_full = np.array(vi_samples['Sigma_c'])      
    Sigma_c_vi = np.diagonal(Sigma_c_vi_full, axis1=2, axis2=3)    

    scores = {}
    scores['beta_c'] = _faes_grid(beta_c_vi, gibbs_arrays['beta_c'])
    scores['gamma_c'] = _faes_grid(gamma_c_vi, gibbs_arrays['gamma_c'])
    scores['beta_0'] = _faes_vec(beta_0_vi, gibbs_arrays['beta_0'])
    # positive support therefore log-space
    scores['lam'] = faes_accuracy(vi_samples['lam'], gibbs_arrays['lam'], positive_support=True)
    scores['Sigma_c'] = _faes_grid(Sigma_c_vi, gibbs_arrays['Sigma_c'], positive_support=True)
    return scores


"""Impulse Response Functions"""

def _build_lag_matrices(beta_c, N, L, K):
    """Reshape a country's stacked beta_c coefficient vector into per-lag
    coefficient matrices.

    Matches kron(I_N, X_c) @ beta_c convention, where each equation's
    K-length block has columns ordered [y_lags (lag-major, N*L), w_lags
    (rest)].

    Parameters
    ----------
    beta_c : numpy.ndarray of shape (N*K,)
        Equation-stacked coefficient vector for one country, one posterior draw.
    N : int
        Number of endogenous variables.
    L : int
        Number of endogenous lags.
    K : int
        Number of regressors per equation.

    Returns
    -------
    list of length L of numpy.ndarray of shape (N, N)
        A_list, where `A_list[l-1]` is A_l, the coefficient matrix on
        y_{t-l} in y_t = A_1 y_{t-1} + ... + A_L y_{t-L} + ...; i.e.
        `A_list[l-1][i, j]` is the coefficient of equation i on variable j
        at lag l.
    """
    A_list = [np.zeros((N, N)) for _ in range(L)]
    for i in range(N):
        eq_block = beta_c[i * K: i * K + N * L]          # drop w_lags tail
        eq_lags = eq_block.reshape(L, N)                  # [lag, variable]
        for l in range(L):
            A_list[l][i, :] = eq_lags[l, :]
    return A_list


def _build_companion(A_list, N, L):
    """Stack A_1..A_L into the top block row of the NL x NL companion matrix.

    Parameters
    ----------
    A_list : list of length L of numpy.ndarray of shape (N, N)
        Per-lag coefficient matrices, as returned by `_build_lag_matrices`.
    N : int
        Number of endogenous variables.
    L : int
        Number of endogenous lags.

    Returns
    -------
    numpy.ndarray of shape (N*L, N*L)
        Companion-form matrix, with [A_1 A_2 ... A_L] as its top block row
        and shift-down identity blocks below.
    """
    NL = N * L
    Acomp = np.zeros((NL, NL))
    Acomp[:N, :] = np.hstack(A_list)          # [A_1 A_2 ... A_L]
    if L > 1:
        Acomp[N:, :NL - N] = np.eye(NL - N)   # shift-down identity blocks
    return Acomp


def _draw_admissible_G(Sigma_c, sign_pattern, max_tries=1000, rng=None):
    """Draw a structural impact matrix G_c whose 2x2 rotated block satisfies a
    given sign pattern, by rejection sampling over random rotations.

    Finds G_c = P_c @ Q, Q = blockdiag(I_{N-2}, V) with V a random 2x2
    rotation matrix parameterized by a single angle theta ~ Uniform(0, 2*pi),
    such that G_c satisfies the full sign pattern on the 2x2 rotated block
    (rows 2,3 x columns 2,3, per eq. 11: interest/exchange rows against
    both the monetary policy shock column and the remaining column):
        G_c[row, col] * sign > 0   for each (row, col, sign) in sign_pattern

    Parameters
    ----------
    Sigma_c : numpy.ndarray of shape (N, N)
        Covariance matrix for one country, one posterior draw.
    sign_pattern : sequence of (int, int, float) tuples
        (row, col, sign) triples; each entry constrains
        `G_c[row, col] * sign > 0`.
    shock_idx : int, optional
        Column index (0-based) of the monetary policy shock in G_c. Default is 2.
    max_tries : int, optional
        Maximum number of rejection-sampling attempts. Default is 1000.
    rng : numpy.random.Generator or None, optional
        Random number generator; a new default generator is created if None.

    Returns
    -------
    G_c : numpy.ndarray of shape (N, N)
        Admissible structural impact matrix.
    n_tries : int
        Number of rotation draws needed to find an admissible `G_c`.

    Raises
    ------
    RuntimeError
        If no admissible rotation is found within `max_tries` attempts.
    """
    if rng is None:
        rng = np.random.default_rng()

    N = Sigma_c.shape[0]
    P_c = np.linalg.cholesky(Sigma_c)

    for attempt in range(1, max_tries + 1):
        theta = rng.uniform(0, 2 * np.pi)
        c, s = np.cos(theta), np.sin(theta)
        V = np.array([[c, -s],
                      [s,  c]])

        Q = np.eye(N)
        Q[N - 2:, N - 2:] = V

        G_c = P_c @ Q

        ok = all(G_c[row, col] * sgn > 0 for row, col, sgn in sign_pattern)
        if ok:
            return G_c, attempt

    raise RuntimeError(f"No admissible rotation found in {max_tries} tries.")


def compute_irfs(beta_samples, sigma_samples, N, L, K, C, H=25,
                  shock_idx=2,
                  sign_pattern=((2, 2, 1.0), (3, 2, -1.0), (2, 3, 1.0), (3, 3, 1.0)),
                  max_tries=1000, seed=None):
    """
    Compute impulse responses to a monetary policy shock for each posterior
    draw and country, stopping at the first admissible rotation per draw.

    Parameters
    ----------
    beta_samples : numpy.ndarray of shape (n_draws, C, N*K)
        Posterior draws of beta_c, per draw and country.
    sigma_samples : numpy.ndarray of shape (n_draws, C, N, N)
        Posterior draws of Sigma_c, per draw and country.
    N : int
        Number of endogenous variables.
    L : int
        Number of lags.
    K : int
        Total columns of X_c (N*L + w-lag columns); only the first N*L
        coefficients per equation are used (lag block), the rest (w_lags,
        analogous to gamma_c) are dropped as they cancel in the IRF.
    C : int
        Number of countries.
    H : int, optional
        Horizon (IRF computed for h = 0, ..., H). Default is 25.
    shock_idx : int, optional
        Column index (0-based) of the monetary policy shock in G_c.
        Default 2 assumes ordering [output, price, interest rate, exchange
        rate] per eq. (11), so monetary policy is the 3rd shock -> index 2.
    sign_pattern : tuple of (int, int, float) triples, optional
        (row, col, sign) triples checked against G_c, covering the full 2x2
        rotated block (both columns 2 and 3, per eq. 11), not just the
        monetary policy shock's own column.
    max_tries : int, optional
        Max rejection-sampling attempts per (draw, country). Default is 1000.
    seed : int or None, optional
        RNG seed for reproducibility. Default is None.

    Returns
    -------
    irfs : numpy.ndarray of shape (n_draws, C, H+1, N)
        `irfs[d, c, h, :]` is the response of the N endogenous variables at
        horizon h to a unit monetary policy shock, for draw d, country c.
    n_tries : numpy.ndarray of shape (n_draws, C), dtype int
        Number of rotation draws needed for each (draw, country) pair.
    """
    rng = np.random.default_rng(seed)
    n_draws = beta_samples.shape[0]
    NL = N * L

    irfs = np.zeros((n_draws, C, H + 1, N))
    n_tries = np.zeros((n_draws, C), dtype=int)

    for d in range(n_draws):
        for c in range(C):
            beta_c = beta_samples[d, c]
            Sigma_c = sigma_samples[d, c]

            # reduced-form dynamics
            A_list = _build_lag_matrices(beta_c, N, L, K)
            Acomp = _build_companion(A_list, N, L)

            # structural impact matrix
            G_c, tries = _draw_admissible_G(
                Sigma_c, sign_pattern=sign_pattern, max_tries=max_tries, rng=rng
            )
            n_tries[d, c] = tries
            impact = G_c[:, shock_idx]           # period-0 response, length N

            # propagate through companion form
            Y = np.zeros(NL)
            Y[:N] = impact
            irfs[d, c, 0, :] = Y[:N]
            for h in range(1, H + 1):
                Y = Acomp @ Y
                irfs[d, c, h, :] = Y[:N]

    return irfs, n_tries


def compute_wasserstein_curve(gibbs_irfs, vi_irfs):
    """
    Compute the Wasserstein distance between Gibbs and VI IRF draw
    distributions, at every country/horizon/variable.

    Parameters
    ----------
    gibbs_irfs : numpy.ndarray of shape (n_draws_gibbs, C, H+1, N)
        IRFs from `compute_irfs` using Gibbs posterior draws.
    vi_irfs : numpy.ndarray of shape (n_draws_vi, C, H+1, N)
        IRFs from `compute_irfs` using a VI method's posterior draws.

    Returns
    -------
    numpy.ndarray of shape (C, H+1, N)
        Wasserstein distance between the Gibbs and VI empirical draw
        distributions at each country/horizon/variable.
    """
    _, C, H_plus_1, N = gibbs_irfs.shape
    distances = np.zeros((C, H_plus_1, N))

    for c in range(C):
        for h in range(H_plus_1):
            for n in range(N):
                distances[c, h, n] = wasserstein_distance(
                    gibbs_irfs[:, c, h, n], vi_irfs[:, c, h, n]
                )
    return distances


"""Effect of lambda on coefficient means"""

def lambda_dispersion_change(lam, coef_c_k, beta0_k_samples, pct=0.2,
                              clip_lo=None, clip_hi=None):
    """Percentage change in across-country dispersion about beta_0 between
    the top and bottom pct of the lambda posterior, for one coefficient,
    plus the same change normalised by the relative lambda range spanned
    (elasticity).

    Parameters
    ----------
    lam : array_like, shape (n_samples,)
    coef_c_k : array_like, shape (C, n_samples)
        Samples of one coefficient, per country.
    beta0_k_samples : array_like, shape (n_samples,)
        Matching draws of beta_0 for this coefficient.
    pct : float, optional
        Fraction defining the low/high lambda tails, computed within the
        (possibly clipped) lambda values. Default is 0.2.
    clip_lo : float or None, optional
        If given (together with clip_hi), restrict to lambda values in
        [clip_lo, clip_hi] before binning (e.g. another method's own
        range), instead of using this method's full posterior. Default
        is None (no clipping).
    clip_hi : float or None, optional
        See clip_lo.

    Returns
    -------
    delta_norm : float
        (Disp_high - Disp_low) / Disp_low, or np.nan if Disp_low is zero
        or no samples fall in the (clipped) range.
    elasticity : float
        delta_norm normalised by the relative lambda range,
        (hi_cut - lo_cut) / lo_cut. np.nan under the same conditions as
        delta_norm.
    lo_cut : float
        Lambda value at the pct quantile (within the clipped range, if
        given).
    hi_cut : float
        Lambda value at the (1 - pct) quantile (within the clipped range,
        if given).
    n_samples : int
        Number of samples the bins were computed from (i.e. falling within
        [clip_lo, clip_hi] if clipping, else the full sample count).
    """
    lam = np.asarray(lam)
    beta0_k = np.asarray(beta0_k_samples)

    if clip_lo is not None and clip_hi is not None:
        in_range = (lam >= clip_lo) & (lam <= clip_hi)
        lam = lam[in_range]
        coef_c_k = coef_c_k[:, in_range]
        beta0_k = beta0_k[in_range]

    n_samples = lam.size
    if n_samples == 0:
        return np.nan, np.nan, np.nan, np.nan, n_samples

    lo_cut, hi_cut = np.quantile(lam, [pct, 1 - pct])

    def disp(mask):
        vals = coef_c_k[:, mask] - beta0_k[mask]
        return np.sqrt(np.mean(vals ** 2))

    disp_low = disp(lam <= lo_cut)
    disp_high = disp(lam >= hi_cut)

    if disp_low == 0 or hi_cut == lo_cut:
        return np.nan, np.nan, lo_cut, hi_cut, n_samples

    delta_norm = (disp_high - disp_low) / disp_low
    elasticity = delta_norm / ((hi_cut - lo_cut) / lo_cut)

    return delta_norm, elasticity, lo_cut, hi_cut, n_samples


def dispersion_change_real_data(results, pct=0.2, clipped=False):
    """For each method and each coefficient, compute the percentage change
    in across-country dispersion about beta_0 between the top and bottom
    pct of the lambda posterior, plus the range-normalised version
    (elasticity).

    Parameters
    ----------
    results : dict
        Mapping from method ("mfvi", "ssvi_i", "ssvi_c", "gibbs") to a
        sub-dict (keyed "samples" for VI methods, "results" for "gibbs")
        with keys "beta_c" (array_like, shape (n_samples, C, N*K)),
        "beta_0" (array_like, shape (n_samples, N*K)), and "lam"
        (array_like, shape (n_samples,)).
    pct : float, optional
        Fraction defining the low/high lambda tails. Default is 0.2.
    clipped : bool, optional
        If True, restrict every method's lambda values to the range of
        MFVI's own lambda posterior before binning, so all methods are
        compared over the same absolute lambda window. If False (default),
        each method uses its own full posterior range.

    Returns
    -------
    pandas.DataFrame
        One row per (method, k) with columns "delta_norm", "elasticity",
        "lam_lo", "lam_hi", "lam_range", "n_samples".
    """
    clip_lo, clip_hi = None, None
    if clipped:
        lam_mfvi = np.array(results["mfvi"]["samples"]["lam"])
        clip_lo, clip_hi = lam_mfvi.min(), lam_mfvi.max()

    rows = []
    for method in ["mfvi", "ssvi_i", "ssvi_c", "gibbs"]:
        d = results[method]["samples" if method != "gibbs" else "results"]
        beta_c = np.array(d["beta_c"])   # (n_samples, C, NK)
        beta_0 = np.array(d["beta_0"])   # (n_samples, NK)
        lam = np.array(d["lam"])

        for k in range(beta_c.shape[2]):
            coef_c_k = beta_c[:, :, k].T  # (C, n_samples)
            delta, elast, lo_cut, hi_cut, n_samples = lambda_dispersion_change(
                lam, coef_c_k, beta_0[:, k], pct=pct,
                clip_lo=clip_lo, clip_hi=clip_hi,
            )
            rows.append({
                "method": method, "k": k, "delta_norm": delta,
                "elasticity": elast, "lam_lo": lo_cut, "lam_hi": hi_cut,
                "lam_range": hi_cut - lo_cut if np.isfinite(hi_cut) else np.nan,
                "n_samples": n_samples,
            })

    return pd.DataFrame(rows)


def dispersion_change_by_seed(results_by_seed, pct=0.2, clipped=False):
    """As `dispersion_change_real_data`, applied per seed and combined into
    one DataFrame with a "seed" column.

    Parameters
    ----------
    results_by_seed : dict
        Mapping from seed to a results dict of the structure expected by
        `dispersion_change_real_data`.
    pct : float, optional
        Fraction defining the low/high lambda tails. Default is 0.2.
    clipped : bool, optional
        If True, restrict every method's lambda values to MFVI's own
        lambda range (computed separately per seed) before binning.
        Default is False.

    Returns
    -------
    pandas.DataFrame
        One row per (seed, method, k) with columns "delta_norm",
        "elasticity", "lam_lo", "lam_hi", "lam_range", "n_samples".
    """
    dfs = []
    for seed, results in results_by_seed.items():
        df = dispersion_change_real_data(results, pct=pct, clipped=clipped)
        df["seed"] = seed
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


def summarise_dispersion_change(df, group_cols=("method",), decimals=4, lam_sci_decimals=2):
    """Mean and std of delta_norm and elasticity, plus mean n_samples,
    with lambda range reported only when each group corresponds to a
    single seed (i.e. "seed" is in group_cols, or there is a single seed
    / no seed column at all). When pooling across seeds, lambda ranges
    are on different scales per seed and are omitted entirely; n_samples
    is still a genuine mean across seeds in that case.

    delta_norm/elasticity/n_samples are rounded to `decimals` places;
    lam_lo, lam_hi, and lam_range are instead formatted as scientific-
    notation strings, since lambda is on a scale (e.g. ~1e-4 or smaller)
    that fixed-point rounding reduces to indistinguishable zeros.

    Parameters
    ----------
    df : pandas.DataFrame
        Output of `dispersion_change_real_data` or `dispersion_change_by_seed`.
    group_cols : sequence of str, optional
        Columns to group by. Default is ("method",).
    decimals : int, optional
        Decimal places for delta_norm/elasticity/n_samples columns.
        Default is 4.
    lam_sci_decimals : int, optional
        Decimal places in the scientific-notation mantissa used for
        lam_lo, lam_hi, and lam_range. Default is 2.

    Returns
    -------
    pandas.DataFrame
        mean and std of delta_norm and elasticity (columns named e.g.
        "delta_norm_mean", "elasticity_std"), plus n_samples (per-seed
        case, constant within group) or n_samples_mean (pooled-across-
        seeds case, genuine mean), plus lam_lo, lam_hi, lam_range (as
        scientific-notation strings) when per-seed.
    """
    clean = df.dropna(subset=["delta_norm", "elasticity"])
    stats = clean.groupby(list(group_cols))[["delta_norm", "elasticity"]].agg(
        ["mean", "std"]
    )
    stats.columns = [f"{var}_{stat}" for var, stat in stats.columns]
    stats = stats.round(decimals)

    per_seed = "seed" in group_cols or "seed" not in clean.columns
    n_col_name = "n_samples" if per_seed else "n_samples_mean"
    n_stats = clean.groupby(list(group_cols))["n_samples"].mean().round(decimals).rename(n_col_name)
    stats = stats.join(n_stats)

    if per_seed:
        lam_stats = clean.groupby(list(group_cols))[["lam_lo", "lam_hi", "lam_range"]].mean()
        lam_stats = lam_stats.map(lambda x: f"{x:.{lam_sci_decimals}e}")
        return stats.join(lam_stats)

    return stats

"""Runtime and iteration counts"""

def runtime_iteration_table(results):
    """Wall-clock runtime and iteration count for each method. Gibbs is
    included for runtime, but has no ELBO trace or convergence-determined
    iteration count (its step count,
    `config.gibbs_kwargs["n_steps"]`, is fixed rather than a stopping
    criterion), so its iteration count is reported as NA.

    Parameters
    ----------
    results : dict
        Single-seed results dict, as returned by `pipeline.run_pipeline`,
        with keys "mfvi", "ssvi_i", "ssvi_c", "gibbs", each a sub-dict
        holding "runtime" (float, wall-clock seconds); "mfvi", "ssvi_i",
        and "ssvi_c" additionally hold "elbo" (list of float, the ELBO
        trace, one entry per coordinate-ascent iteration).

    Returns
    -------
    pandas.DataFrame
        Rows indexed by method display name ("MFVI", "SSVI-I", "SSVI-C",
        "Gibbs"), columns "runtime_s" (float, wall-clock seconds) and
        "n_iterations" (float, length of the ELBO trace, or NaN for
        Gibbs).
    """
    methods = [("MFVI", "mfvi"), ("SSVI-I", "ssvi_i"), ("SSVI-C", "ssvi_c"), ("Gibbs", "gibbs")]
    rows = [
        {
            "method": name,
            "runtime_s": results[key]["runtime"],
            "n_iterations": len(results[key]["elbo"]) if key != "gibbs" else np.nan,
        }
        for name, key in methods
    ]
    return pd.DataFrame(rows).set_index("method")

def runtime_table_seed(results_by_seed):
    """Mean wall-clock runtime per method, averaged over seeds.

    Parameters
    ----------
    results_by_seed : dict
        Mapping from seed to a single-seed results dict (as returned by
        `pipeline.run_pipeline`), each with keys "mfvi", "ssvi_i",
        "ssvi_c", "gibbs", every one a sub-dict holding "runtime" (float,
        wall-clock seconds).

    Returns
    -------
    pandas.DataFrame
        Rows indexed by method display name ("MFVI", "SSVI-I", "SSVI-C",
        "Gibbs"), column "runtime_mean_s" (float, in seconds), computed
        across seeds.
    """
    methods = [("MFVI", "mfvi"), ("SSVI-I", "ssvi_i"), ("SSVI-C", "ssvi_c"), ("Gibbs", "gibbs")]
    rows = [
        {"method": name, "runtime_s": results[key]["runtime"]}
        for results in results_by_seed.values()
        for name, key in methods
    ]
    stats = pd.DataFrame(rows).groupby("method")["runtime_s"].mean().rename("runtime_mean_s").to_frame()
    return stats.loc[[name for name, _ in methods]]

"""Comparison to true parameters"""

def coverage_table(results_by_seed, true_by_seed, param_key, methods,
                    levels=(0.5, 0.8, 0.95), axis_type="country"):
    """Compute empirical credible-interval coverage of a parameter, pooled
    across seeds (and countries, if applicable), for each method.

    Parameters
    ----------
    results_by_seed : dict
        Mapping from seed to a dict of method -> dict (keyed "results" for
        "gibbs", "samples" otherwise) containing `param_key` (array_like of
        shape (n_samples, C, ...) if `axis_type="country"`, or
        (n_samples, ...) if `axis_type="vector"`).
    true_by_seed : dict
        Mapping from seed to a dict of true parameter values (e.g.
        `simulate.simulate_data`'s `true_params`), containing `param_key`.
    param_key : str
        Which parameter to evaluate coverage for (e.g. "beta_c", "beta_0").
    methods : sequence of str
        Method names to include (e.g. ("mfvi", "ssvi_i", "ssvi_c", "gibbs")).
    levels : sequence of float, optional
        Central credible-interval levels to evaluate. Default is
        (0.5, 0.8, 0.95).
    axis_type : str, optional
        "country" for parameters with a country axis (e.g. beta_c,
        gamma_c), or "vector" for parameters without one (e.g. beta_0).
        Default is "country".

    Returns
    -------
    pandas.DataFrame
        Rows indexed by method, columns labeled e.g. "50%", "80%", "95%",
        values the empirical coverage proportion at that level.
    """
    # axis_type: "country" (e.g. beta_c, gamma_c), "vector" (e.g. beta_0)
    hits = {method: {level: [] for level in levels} for method in methods}

    for method in methods:
        result_key = "results" if method == "gibbs" else "samples"
        for seed, results in results_by_seed.items():
            samples = np.array(results[method][result_key][param_key])
            true_vals = np.array(true_by_seed[seed][param_key])

            if axis_type == "country":
                for c in range(samples.shape[1]):
                    s, t = samples[:, c, :], true_vals[c]
                    for level in levels:
                        alpha = 1 - level
                        lo, hi = np.quantile(s, [alpha / 2, 1 - alpha / 2], axis=0)
                        hits[method][level].append((lo <= t) & (t <= hi))
            else:  # vector
                s, t = samples, true_vals
                for level in levels:
                    alpha = 1 - level
                    lo, hi = np.quantile(s, [alpha / 2, 1 - alpha / 2], axis=0)
                    hits[method][level].append((lo <= t) & (t <= hi))

    table = pd.DataFrame({
        method: {level: np.concatenate(hits[method][level]).mean() for level in levels}
        for method in methods
    }).T
    table.columns = [f"{int(l*100)}%" for l in table.columns]
    return table

