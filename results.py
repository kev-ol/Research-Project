import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import invgamma, invwishart, gaussian_kde, wasserstein_distance
from scipy.linalg import eigh
from joblib import Parallel, delayed
import pandas as pd
from ssvi_i import calc_V_deltac, calc_mu_deltac
from ssvi_c import calc_V_beta02, calc_mu_beta02, calc_V_deltac2, calc_mu_deltac2


"""Posterior sample reconstruction from each method's variational/MCMC output"""

def sample_from_mfvi(results_mfvi, mfvi_pack, C, N, K, T, n_samples=10000):
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
    mu_delta, V_delta, v_bar, s_bar, S_bar_sigma = results_mfvi.values()

    idx_deltac = mfvi_pack["idx_deltac"]
    size_deltac = mfvi_pack["size_deltac"]

    # --- delta samples (beta_0, beta_c, gamma_c, delta_c all come from this) ---
    L = np.linalg.cholesky(V_delta)
    deltas = mu_delta + (L @ np.random.normal(size=(len(mu_delta), n_samples))).T  # (n_samples, size_delta)

    beta_0_samples = deltas[:, :idx_deltac[0]]  # (n_samples, size_beta0)

    beta_c_samples = [deltas[:, idx_deltac[c]:idx_deltac[c] + N*K] for c in range(C)]              # list of (n_samples, N*K)
    gamma_c_samples = [deltas[:, idx_deltac[c] + N*K:idx_deltac[c] + size_deltac] for c in range(C)]  # list of (n_samples, size_gammac)
    delta_c_samples = [deltas[:, idx_deltac[c]:idx_deltac[c] + size_deltac] for c in range(C)]     # list of (n_samples, size_deltac)

    # --- lambda samples (cheap, vectorize trivially) ---
    lam_samples = invgamma.rvs(s_bar/2, scale=v_bar/2, size=n_samples)

    # --- Sigma_c samples (vectorized per country via scipy's size argument) ---
    Sigma_c_samples = [invwishart.rvs(T, S_bar_sigma[c], size=n_samples) for c in range(C)]  # each: (n_samples, N, N)

    return {
        'beta_0': list(beta_0_samples),
        'lam': list(lam_samples),
        'beta_c': [[beta_c_samples[c][n] for c in range(C)] for n in range(n_samples)],
        'gamma_c': [[gamma_c_samples[c][n] for c in range(C)] for n in range(n_samples)],
        'delta_c': [[delta_c_samples[c][n] for c in range(C)] for n in range(n_samples)],
        'Sigma_c': [[Sigma_c_samples[c][n] for c in range(C)] for n in range(n_samples)],
    }


def sample_from_ssvi_i(results_ssvi_i, ssvi_i_pack, C, N, K, T, n_samples=10000):
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
    mu_beta0, V_beta0, q_lambda_chain, S_bar_sigma, cov_deltac = results_ssvi_i.values()
    mu_sigma_inv = [T * np.linalg.inv(S_bar_sigma[c]) for c in range(C)]
    Y, F, FF, idx_deltac, size_deltac, Pc, Lambda_inv, Lambda_inv_sum = ssvi_i_pack.values()

    # lambda: sample from a KDE fit to the converged ULA chain (log-space, since lambda > 0)
    lam_chain = np.asarray(q_lambda_chain)
    kde_lam = gaussian_kde(np.log(lam_chain))
    lam_samples = np.exp(kde_lam.resample(n_samples).flatten())

    # beta_0: independent draw, paired with lam_samples by index
    L_beta0 = np.linalg.cholesky(V_beta0)
    beta_0_samples = mu_beta0 + (L_beta0 @ np.random.normal(size=(len(mu_beta0), n_samples))).T  # (n_samples, size_beta0)

    # V_deltac, mu_deltac: both batched over the same (lam, beta0) pairs
    V_deltac = calc_V_deltac(lam_samples, mu_sigma_inv, FF, Lambda_inv, size_deltac, Pc, C, N, K)
    mu_deltac = calc_mu_deltac(lam_samples, beta_0_samples, V_deltac, mu_sigma_inv, Y, F, Lambda_inv, size_deltac, Pc, C, N, K)

    delta_c_samples_arr = np.empty((n_samples, C, size_deltac))
    for c in range(C):
        L_c = np.linalg.cholesky(V_deltac[c])                 # (n_samples, size_deltac, size_deltac)
        z = np.random.normal(size=(n_samples, size_deltac))
        delta_c_samples_arr[:, c, :] = mu_deltac[c] + np.einsum('nij,nj->ni', L_c, z)

    beta_c_samples_arr = delta_c_samples_arr[:, :, :N*K]
    gamma_c_samples_arr = delta_c_samples_arr[:, :, N*K:]

    Sigma_c_samples = np.stack(
        [invwishart.rvs(T, S_bar_sigma[c], size=n_samples) for c in range(C)], axis=1
    )

    return {
        'beta_0': list(beta_0_samples),
        'lam': list(lam_samples),
        'beta_c': list(beta_c_samples_arr),
        'gamma_c': list(gamma_c_samples_arr),
        'delta_c': list(delta_c_samples_arr),
        'Sigma_c': list(Sigma_c_samples),
    }


def sample_from_ssvi_c(results_ssvi_c, ssvi_i_pack, C, N, K, T, n_samples=10000):
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
    q_lambda_chain, S_bar_sigma, cov_deltac = results_ssvi_c.values()
    mu_sigma_inv = [T * np.linalg.inv(S_bar_sigma[c]) for c in range(C)]
    Y, F, FF, idx_deltac, size_deltac, Pc, Lambda_inv, Lambda_inv_sum = ssvi_i_pack.values()

    # lambda: sample from a KDE fit to the converged ULA chain (log-space, since lambda > 0)
    lam_chain = np.asarray(q_lambda_chain)
    kde_lam = gaussian_kde(np.log(lam_chain))
    lam_samples = np.exp(kde_lam.resample(n_samples).flatten())

    # V_deltac(lambda), computed once and reused for both beta0 and delta_c
    V_deltac = calc_V_deltac2(lam_samples, mu_sigma_inv, FF, Lambda_inv, size_deltac, Pc, C, N, K)

    # beta_0 | lambda
    V_beta0 = calc_V_beta02(lam_samples, V_deltac, Lambda_inv, Lambda_inv_sum, C, N, K)
    mu_beta0 = calc_mu_beta02(lam_samples, V_deltac, mu_sigma_inv, V_beta0, Y, F, Lambda_inv, Pc, C, N, K)

    size_beta0 = mu_beta0.shape[-1]
    L_beta0 = np.linalg.cholesky(V_beta0)
    z_beta0 = np.random.normal(size=(n_samples, size_beta0))
    beta_0_samples = mu_beta0 + np.einsum('nij,nj->ni', L_beta0, z_beta0)

    # delta_c | lambda, beta0
    mu_deltac = calc_mu_deltac2(lam_samples, beta_0_samples, V_deltac, mu_sigma_inv, Y, F, Lambda_inv, size_deltac, Pc, C, N, K)

    delta_c_samples_arr = np.empty((n_samples, C, size_deltac))
    for c in range(C):
        L_c = np.linalg.cholesky(V_deltac[c])
        z = np.random.normal(size=(n_samples, size_deltac))
        delta_c_samples_arr[:, c, :] = mu_deltac[c] + np.einsum('nij,nj->ni', L_c, z)

    beta_c_samples_arr = delta_c_samples_arr[:, :, :N*K]
    gamma_c_samples_arr = delta_c_samples_arr[:, :, N*K:]

    Sigma_c_samples = np.stack(
        [invwishart.rvs(T, S_bar_sigma[c], size=n_samples) for c in range(C)], axis=1
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
    """Per-country delta_c covariance block sliced out of MFVI's full V_delta.

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
    V_delta, without needing the full `mfvi_pack`.

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


"""Accuracy Measure (Faes et al. 2011, ter Steege eq. 20) — vectorized + parallel"""

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

    # Bounds per paper: smallest/largest value across BOTH sample sets, no padding
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
    beta_c_gibbs = np.array(gibbs_samples['beta_c'])          # (T, C, N*K)
    gamma_c_gibbs = np.array(gibbs_samples['gamma_c'])        # (T, C, N*n_zc)
    beta_0_gibbs = np.array(gibbs_samples['beta_0'])          # (T, N*K)

    # Sigma_c diagonals: extract diag once, vectorized, before parallel calls
    Sigma_c_gibbs_full = np.array(gibbs_samples['Sigma_c'])   # (T, C, N, N)
    Sigma_c_gibbs = np.diagonal(Sigma_c_gibbs_full, axis1=2, axis2=3)   # (T, C, N)

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
    beta_c_vi = np.array(vi_samples['beta_c'])          # (T, C, N*K)
    gamma_c_vi = np.array(vi_samples['gamma_c'])        # (T, C, N*n_zc)
    beta_0_vi = np.array(vi_samples['beta_0'])          # (T, N*K)

    # Sigma_c diagonals: extract diag once, vectorized, before parallel calls
    Sigma_c_vi_full = np.array(vi_samples['Sigma_c'])       # (T, C, N, N)
    Sigma_c_vi = np.diagonal(Sigma_c_vi_full, axis1=2, axis2=3)       # (T, C, N)

    scores = {}
    scores['beta_c'] = _faes_grid(beta_c_vi, gibbs_arrays['beta_c'])
    scores['gamma_c'] = _faes_grid(gamma_c_vi, gibbs_arrays['gamma_c'])
    scores['beta_0'] = _faes_vec(beta_0_vi, gibbs_arrays['beta_0'])
    scores['lam'] = faes_accuracy(vi_samples['lam'], gibbs_arrays['lam'], positive_support=True)
    scores['Sigma_c'] = _faes_grid(Sigma_c_vi, gibbs_arrays['Sigma_c'], positive_support=True)
    return scores


def plot_accuracy_boxplots(results_faes, method_name, C):
    """Boxplots of Faes accuracy per parameter block, for one VI method vs Gibbs.

    Parameters
    ----------
    results_faes : dict
        Output of `compute_faes_scores`.
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

            # --- reduced-form dynamics ---
            A_list = _build_lag_matrices(beta_c, N, L, K)
            Acomp = _build_companion(A_list, N, L)

            # --- structural impact matrix ---
            G_c, tries = _draw_admissible_G(
                Sigma_c, sign_pattern=sign_pattern, max_tries=max_tries, rng=rng
            )
            n_tries[d, c] = tries
            impact = G_c[:, shock_idx]           # period-0 response, length N

            # --- propagate through companion form ---
            Y = np.zeros(NL)
            Y[:N] = impact
            irfs[d, c, 0, :] = Y[:N]
            for h in range(1, H + 1):
                Y = Acomp @ Y
                irfs[d, c, h, :] = Y[:N]

    return irfs, n_tries


def plot_irfs_comparison(gibbs_irfs, vi_irfs, country_names, variable_names, vi_label="VI"):
    """
    Plot a grid of impulse response comparisons between Gibbs and a VI method.

    Parameters
    ----------
    gibbs_irfs : numpy.ndarray of shape (n_draws_gibbs, C, H+1, N)
        IRFs from `compute_irfs` using Gibbs posterior draws — plotted as
        blue fanchart (5-95 percentile, 5% steps) + black solid median.
    vi_irfs : numpy.ndarray of shape (n_draws_vi, C, H+1, N)
        IRFs from `compute_irfs` using a VI method's posterior draws —
        plotted as red solid median + red dashed 5/95 percentiles.
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
        (C, H+1, N) from `compute_wasserstein_curve`, e.g.
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


def plot_accuracy_boxplots_pooled(results_dict, method_name, method_key):
    """Same layout as `plot_accuracy_boxplots`, but pooled across all seeds in
    results_dict.

    Parameters
    ----------
    results_dict : dict
        Mapping from seed to a per-seed results dict, each containing key
        "C" (int) and `method_key` (dict with key "faes", the output of
        `compute_faes_scores`).
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

    # lambda: now has n_seeds values, use scatter (not boxplot, per earlier n=4 discussion)
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

def plot_corr_mae_boxplots(results_dict, N, K, Z_width):
    """Boxplots of mean absolute delta_c correlation error vs Gibbs, pooled
    across countries and seeds, for each VI method.

    Parameters
    ----------
    results_dict : dict
        Mapping from seed to a single-seed results dict (see
        `corr_mae_table` for its required structure).
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

def plot_deltac_corr_diff_heatmaps(results, N, K, Z_width, seed):
    """Plot, for each country, heatmaps of the delta_c correlation-matrix
    difference (each VI method minus Gibbs).

    Parameters
    ----------
    results : dict
        Single-seed results dict (see `corr_mae_table` for its required
        structure).
    N : int
        Number of endogenous variables.
    K : int
        Number of regressors per equation.
    Z_width : int
        Number of non-exchangeable regressors per equation.
    seed : int or str
        Seed identifier, used only in the figure title.

    Returns
    -------
    None
        Displays one matplotlib figure per country; nothing is returned.
    """
    C = results["C"]
    country_names = results["config"].country_names

    V_delta = results["mfvi"]["results"]["V_delta"]
    cov_mfvi = extract_cov_mfvi(V_delta, N, K, Z_width, C)
    cov_ssvi_i = results["ssvi_i"]["results"]["cov_deltac"]
    cov_ssvi_c = results["ssvi_c"]["results"]["cov_deltac"]
    cov_true = results["cov_true"]

    methods = [("MFVI", cov_mfvi), ("SSVI-I", cov_ssvi_i), ("SSVI-C", cov_ssvi_c)]

    for c in range(C):
            corr_true = cov2corr(cov_true[c])
            np.fill_diagonal(corr_true, np.nan)

            diffs = []
            for _, cov in methods:
                corr_method = cov2corr(cov[c])
                np.fill_diagonal(corr_method, np.nan)
                diffs.append(corr_method - corr_true)

            vmax = np.nanpercentile(np.abs(diffs), 98)

            fig, axes = plt.subplots(1, 3, figsize=(15, 5))

            for ax, (label, _), diff in zip(axes, methods, diffs):
                im = ax.imshow(diff, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
                ax.set_title(f"{label} − Gibbs")
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            fig.suptitle(f"δ_{{{country_names[c]}}} correlation error — seed {seed}", fontsize=14)
            plt.tight_layout()
            plt.show()

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
