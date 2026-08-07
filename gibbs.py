# Import libraries

import numpy as np
import arviz as az
from scipy.stats import invgamma, invwishart
from numpy.linalg import lstsq

"""Gibbs Sampling Functions"""

def beta_0_sample(lam, Sigma_c_inv, gamma_c, y, X, XX, Z, Lambda_inv, Lambda_inv_sum, C, N, rng):
    """Draw one sample of beta_0 from its conditional posterior with beta_c's marginalised out.

    Parameters
    ----------
    lam : float
        Current draw of the shrinkage parameter lambda.
    Sigma_c_inv : list of length C of numpy.ndarray of shape (N, N)
        Current per-country draw of Sigma_c^{-1}.
    gamma_c : list of length C of numpy.ndarray of shape (N*Z_width,)
        Current per-country draw of gamma_c.
    y : numpy.ndarray of shape (C, T*N)
        Column-major vectorized endogenous data, one row per country.
    X : numpy.ndarray of shape (C, T, K)
        Per-country design matrix of exchangeable regressors (endogenous and
        W lags).
    XX : numpy.ndarray of shape (C, K, K)
        Per-country X_c.T @ X_c.
    Z : numpy.ndarray of shape (T, Z_width)
        Non-exchangeable regressors, shared across countries.
    Lambda_inv : numpy.ndarray of shape (C, N*K, N*K)
        Per-country inverse Minnesota-prior scale matrices.
    Lambda_inv_sum : numpy.ndarray of shape (N*K, N*K)
        Sum of `Lambda_inv` over countries.
    C : int
        Number of countries.
    N : int
        Number of endogenous variables.
    rng : numpy.random.Generator
        Random number generator used to draw the beta_0 sample.

    Returns
    -------
    sample : numpy.ndarray of shape (N*K,)
        Draw of beta_0 from its conditional posterior (beta_c's marginalised)
    P_inv : list of length C of numpy.ndarray of shape (N*K, N*K)
        Per-country beta_c conditional-posterior covariance matrices, computed
        here for reuse by `beta_c_sample`.
    """
    # Posterior: beta_0 | rest (excluding beta_c) ~ N(mu, V)
    # V = lambda * (sum Lambda_inv_c)^{-1}
    # mu = V * (1/lambda) * sum_c Lambda_inv_c beta_c
    P_inv = [np.linalg.inv((1/lam)*Lambda_inv[c] + np.kron(Sigma_c_inv[c], XX[c])) for c in range(C)]
    precision_matrix = ((Lambda_inv_sum - (1/lam) * sum(Lambda_inv[c] @ P_inv[c] @ Lambda_inv[c] for c in range(C)))) / lam
    precision_matrix = (precision_matrix + precision_matrix.T) / 2
    V_beta_0 = np.linalg.inv(precision_matrix)
    V_beta_0 = (V_beta_0 + V_beta_0.T) / 2
    r = [y[c] - np.kron(np.eye(N), Z) @ gamma_c[c] for c in range(C)]
    mu_beta_0 = V_beta_0 @ ((1/lam) * sum(Lambda_inv[c] @ P_inv[c] @ np.kron(Sigma_c_inv[c], X[c].T) @ r[c] for c in range(C)))
    sample = rng.multivariate_normal(mu_beta_0, V_beta_0, method="cholesky")
    return sample, P_inv

def lambda_sample(beta_c, beta_0, Lambda_inv, C, N, K, rng):
    """Draw one sample of lambda from its conditional posterior,
    lambda | rest ~ InvGamma(s_bar/2, v_bar/2).

    v_bar = sum_c (beta_c - beta_0)' Lambda_inv_c (beta_c - beta_0).

    Parameters
    ----------
    beta_c : list of length C of numpy.ndarray of shape (N*K,)
        Current per-country draw of beta_c.
    beta_0 : numpy.ndarray of shape (N*K,)
        Current draw of beta_0.
    Lambda_inv : list of length C of numpy.ndarray of shape (N*K, N*K)
        Per-country inverse Minnesota-prior scale matrices.
    C : int
        Number of countries.
    N : int
        Number of endogenous variables.
    K : int
        Number of regressors per equation.
    rng : numpy.random.Generator
        Random number generator used to draw the lambda sample.

    Returns
    -------
    float
        Draw of lambda from its conditional posterior.
    """
    # Posterior: lambda | rest ~ InvGamma(s_bar/2, v_bar/2)
    # v_bar = sum_c (beta_c - beta_0)' Lambda_inv_c (beta_c - beta_0)
    s_bar = C*N*K -1
    v_bar = sum((beta_c[c]-beta_0).T @ Lambda_inv[c] @ (beta_c[c]-beta_0) for c in range(C))
    sample = invgamma.rvs(s_bar/2, scale=v_bar/2, random_state=rng)
    return sample

def beta_c_sample(lam, beta_0, Sigma_inv, gamma, V_beta_c, y_c, X_c, Z, Lambda_inv_c, N, rng):
    """Draw one sample of beta_c (for a single country) from its conditional
    posterior, beta_c | rest ~ N(mu, V), with precision
    (1/lambda) Lambda_inv_c + Sigma_inv (kron) X'X.

    Parameters
    ----------
    lam : float
        Current draw of the shrinkage parameter lambda.
    beta_0 : numpy.ndarray of shape (N*K,)
        Current draw of beta_0.
    Sigma_inv : numpy.ndarray of shape (N, N)
        Current draw of this country's Sigma_c^{-1}.
    gamma : numpy.ndarray of shape (N*Z_width,)
        Current draw of this country's gamma_c.
    V_beta_c : numpy.ndarray of shape (N*K, N*K)
        This country's beta_c conditional-posterior covariance matrix, as
        returned by `beta_0_sample`.
    y_c : numpy.ndarray of shape (T*N,)
        Column-major vectorized endogenous data for this country.
    X_c : numpy.ndarray of shape (T, K)
        This country's design matrix of exchangeable regressors.
    Z : numpy.ndarray of shape (T, Z_width)
        Non-exchangeable regressors, shared across countries.
    Lambda_inv_c : numpy.ndarray of shape (N*K, N*K)
        This country's inverse Minnesota-prior scale matrix.
    N : int
        Number of endogenous variables.
    rng : numpy.random.Generator
        Random number generator used to draw this country's beta_c sample.

    Returns
    -------
    numpy.ndarray of shape (N*K,)
        Draw of this country's beta_c from its conditional posterior.
    """
    # Posterior: beta_c | rest ~ N(mu, V)
    # Precision = (1/lambda) Lambda_inv_c + Sigma_inv ⊗ X'X
    # r_c removes the gamma contribution from y before computing mu
    r_c = y_c - np.kron(np.eye(N), Z) @ gamma
    mu_beta_c = V_beta_c@((1/lam)*Lambda_inv_c @ beta_0 + np.kron(Sigma_inv, X_c.T) @ r_c)
    sample = rng.multivariate_normal(mu_beta_c, V_beta_c, method="cholesky")
    return sample

def gamma_c_sample(Sigma_inv, beta, y_c, X_c, Z, ZZ, N, rng):
    """Draw one sample of gamma_c (for a single country) from its conditional
    posterior, gamma_c | rest ~ N(mu, V), with precision Sigma_inv (kron) Z'Z.

    Parameters
    ----------
    Sigma_inv : numpy.ndarray of shape (N, N)
        Current draw of this country's Sigma_c^{-1}.
    beta : numpy.ndarray of shape (N*K,)
        Current draw of this country's beta_c.
    y_c : numpy.ndarray of shape (T*N,)
        Column-major vectorized endogenous data for this country.
    X_c : numpy.ndarray of shape (T, K)
        This country's design matrix of exchangeable regressors.
    Z : numpy.ndarray of shape (T, Z_width)
        Non-exchangeable regressors, shared across countries.
    ZZ : numpy.ndarray of shape (Z_width, Z_width)
        Z.T @ Z.
    N : int
        Number of endogenous variables.
    rng : numpy.random.Generator
        Random number generator used to draw this country's gamma_c sample.

    Returns
    -------
    numpy.ndarray of shape (N*Z_width,)
        Draw of this country's gamma_c from its conditional posterior.
    """
    # Posterior: gamma_c | rest ~ N(mu, V)
    # Precision = Sigma_inv ⊗ Z'Z
    # r_c removes the beta contribution from y before computing mu
    V_gamma_c = np.linalg.inv(np.kron(Sigma_inv, ZZ))
    r_c = y_c - np.kron(np.eye(N), X_c) @ beta
    mu_gamma_c = V_gamma_c@(np.kron(Sigma_inv, Z.T)) @ r_c
    sample = rng.multivariate_normal(mu_gamma_c, V_gamma_c, method="cholesky")
    return sample

def Sigma_c_sample(Beta_c, gamma_c, Y_c, X_c, Z, T, rng):
    """Draw one sample of Sigma_c (for a single country) from its conditional
    posterior, Sigma_c | rest ~ InvWishart(T, S_bar), where S_bar is the
    residual sum of squares after removing fitted values.

    Parameters
    ----------
    Beta_c : numpy.ndarray of shape (K, N)
        This country's beta_c, reshaped to (K, N).
    gamma_c : numpy.ndarray of shape (Z_width, N)
        This country's gamma_c, reshaped to (Z_width, N).
    Y_c : numpy.ndarray of shape (T, N)
        This country's endogenous data.
    X_c : numpy.ndarray of shape (T, K)
        This country's design matrix of exchangeable regressors.
    Z : numpy.ndarray of shape (T, Z_width)
        Non-exchangeable regressors, shared across countries.
    N : int
        Number of endogenous variables.
    T : int
        Number of time periods.
    rng : numpy.random.Generator
        Random number generator used to draw this country's Sigma_c sample.

    Returns
    -------
    numpy.ndarray of shape (N, N)
        Draw of this country's Sigma_c from its conditional posterior.
    """
    # Posterior: Sigma_c | rest ~ InvWishart(T, S_bar)
    # S_bar is the residual sum of squares after removing fitted values
    resid = Y_c - X_c @ Beta_c - Z @ gamma_c
    S_bar = resid.T @ resid
    sample = invwishart.rvs(T, S_bar, random_state=rng)
    return sample

"""Diagnostics Computation"""

def _compute_diagnostics(all_chains_data, n_burnin):
    """Compute bulk effective sample size (ESS) and rank-normalised R-hat for
    every scalar parameter component, across chains.

    Flattens all parameters from all chains into a single
    (n_chains, n_post, D) array, where D is the total number of scalar
    components across all parameters, then hands it to arviz.

    Parameters
    ----------
    all_chains_data : list of dict
        One dict per chain, each with keys 'lam' (list of float), 'beta_0'
        (list of numpy.ndarray of shape (N*K,)), 'beta_c' (list of list of
        length C of numpy.ndarray of shape (N*K,)), 'gamma_c' (list of list
        of length C of numpy.ndarray of shape (N*Z_width,)), and 'Sigma_c'
        (list of list of length C of numpy.ndarray of shape (N, N)).
    n_burnin : int
        Number of initial post-warmup draws to discard from each chain before
        computing diagnostics.

    Returns
    -------
    ess : numpy.ndarray of shape (D,)
        Bulk effective sample size for each scalar parameter component.
    r_hat : numpy.ndarray of shape (D,)
        Rank-normalised R-hat for each scalar parameter component.
    """
    # Flatten all parameters from all chains into a single (n_chains, n_post, D)
    # array, where D is the total number of scalar components across all parameters.
    # arviz then computes bulk ESS and rank-normalised R-hat for each component.
    chain_vecs = []
    for chain_data in all_chains_data:
        n_post = len(chain_data['lam']) - n_burnin

        # Discard burn-in and flatten each parameter to (n_post, d_param)
        lam     = np.array(chain_data['lam'])[n_burnin:, np.newaxis] 
        beta_0  = np.array(chain_data['beta_0'] )[n_burnin:] 
        beta_c  = np.array(chain_data['beta_c'] )[n_burnin:].reshape(n_post, -1)
        gamma_c = np.array(chain_data['gamma_c'])[n_burnin:].reshape(n_post, -1) 
        Sigma_c = np.array(chain_data['Sigma_c'])[n_burnin:].reshape(n_post, -1) 

        chain_vecs.append(np.concatenate([lam, beta_0, beta_c, gamma_c, Sigma_c], axis=1))

    all_scalars = np.stack(chain_vecs, axis=0)

    # arviz expects (n_chains, n_draws, *param_shape); passing D as the param shape
    # gives one ESS and one R-hat value per scalar component
    data  = {'params': all_scalars}
    ess   = az.ess(data)['params'].values
    r_hat = az.rhat(data)['params'].values

    return ess, r_hat

"""Gibbs Sampling Loop"""

def run_gibbs(gibbs_pack, C, N, K, Z_width, T, n_chains=4, n_steps=10000, n_burnin=2000, rng=None):
    """Run a multi-chain Gibbs sampler over beta_0, lambda, beta_c, gamma_c and
    Sigma_c, and compute convergence diagnostics.

    Parameters
    ----------
    gibbs_pack : dict
        Data pack produced by `data_prep.prep_data`, with (in order) keys
        'Y' (numpy.ndarray, shape (C, T, N)), 'X' (numpy.ndarray, shape
        (C, T, K)), 'XX' (numpy.ndarray, shape (C, K, K)), 'Z' (numpy.ndarray,
        shape (T, Z_width)), 'ZZ' (numpy.ndarray, shape (Z_width, Z_width)),
        'Lambda_inv' (numpy.ndarray, shape (C, N*K, N*K)), and
        'Lambda_inv_sum' (numpy.ndarray, shape (N*K, N*K)).
    C : int
        Number of countries.
    N : int
        Number of endogenous variables.
    K : int
        Number of regressors per equation.
    Z_width : int
        Number of non-exchangeable regressors per equation.
    T : int
        Number of time periods.
    n_chains : int, optional
        Number of independent Gibbs chains to run. Default is 4.
    n_steps : int, optional
        Number of Gibbs sweeps per chain. Default is 10000.
    n_burnin : int, optional
        Number of initial sweeps per chain discarded before pooling samples
        and computing diagnostics. Default is 2000.
    rng : int, numpy.random.SeedSequence, numpy.random.Generator, or None, optional
        Source of randomness for chain initialization and every conditional
        draw. If None (default), a fresh, non-reproducible generator is used.

    Returns
    -------
    post_burnin_samples : dict
        Post-burn-in samples pooled across all chains, with keys 'beta_0'
        (numpy.ndarray, shape (n_chains*(n_steps-n_burnin), N*K)), 'lam'
        (numpy.ndarray, shape (n_chains*(n_steps-n_burnin),)), 'beta_c'
        (numpy.ndarray, shape (n_chains*(n_steps-n_burnin), C, N*K)),
        'gamma_c' (numpy.ndarray, shape (n_chains*(n_steps-n_burnin), C,
        N*Z_width)), 'Sigma_c' (numpy.ndarray, shape
        (n_chains*(n_steps-n_burnin), C, N, N)), and 'delta_c' (list of
        length n_chains*(n_steps-n_burnin), each a list of length C of
        numpy.ndarray of shape (N*K + N*Z_width,), the concatenation of
        beta_c and gamma_c per country and draw).
    ess : numpy.ndarray of shape (D,)
        Bulk effective sample size for each scalar parameter component (see
        `_compute_diagnostics`).
    r_hat : numpy.ndarray of shape (D,)
        Rank-normalised R-hat for each scalar parameter component (see
        `_compute_diagnostics`).
    """
    Y, X, XX, Z, ZZ, Lambda_inv, Lambda_inv_sum = gibbs_pack.values()
    rng = np.random.default_rng(rng)

    # vectorise Y column-major so y[c] = vec(Y_c), matching the Kronecker convention
    y = np.zeros((C, T*N))
    for c in range(C):
        y[c] = Y[c, :, :].flatten(order='F')

    # joint OLS per country on F_c = [X_c, Z] gives sensible beta_c/gamma_c starts
    beta_c_ols = []
    gamma_c_ols = []
    for c in range(C):
        F_c = np.hstack([X[c, :, :], Z]) 
        coef, *_ = lstsq(F_c, Y[c, :, :], rcond=None)
        beta_c_ols.append(coef[:K, :].flatten(order='F'))
        gamma_c_ols.append(coef[K:, :].flatten(order='F'))

    beta_0_ols = np.mean(beta_c_ols, axis=0)

    all_chains_data = []

    for chain_idx in range(n_chains):
        # Small, controlled per-chain perturbations around grounded starting points
        # Enough spread for a valid R-hat check, without starting chains far from the posterior's actual region.
        noise_scale = 0.05 * (chain_idx + 1)

        beta_0      = beta_0_ols + rng.standard_normal(N*K) * noise_scale
        beta_c      = [beta_c_ols[c] + rng.standard_normal(N*K) * noise_scale for c in range(C)]
        gamma_c     = [gamma_c_ols[c] + rng.standard_normal(N*Z_width) * noise_scale for c in range(C)]
        Sigma_c     = [(lambda A: A @ A.T + np.eye(N))(rng.standard_normal((N, N))) for _ in range(C)]
        Sigma_c_inv = [np.linalg.inv(S) for S in Sigma_c]
        lam         = 1e-4 + 1e-5 * rng.uniform(-1, 1) * (chain_idx + 1)

        samples = {'beta_0': [], 'lam': [], 'beta_c': [], 'gamma_c': [], 'Sigma_c': []}

        # Sweep through all conditional posteriors in turn
        for n in range(n_steps):
            beta_0, V_beta_c = beta_0_sample(lam, Sigma_c_inv, gamma_c, y, X, XX, Z, Lambda_inv, Lambda_inv_sum, C, N, rng)
            samples['beta_0'].append(beta_0.copy())

            beta_c = [beta_c_sample(lam, beta_0, Sigma_c_inv[c], gamma_c[c], V_beta_c[c], y[c], X[c,:,:], Z, Lambda_inv[c], N, rng) for c in range(C)]
            samples['beta_c'].append(beta_c.copy())

            lam = lambda_sample(beta_c, beta_0, Lambda_inv, C, N, K, rng)
            samples['lam'].append(lam)

            gamma_c = [gamma_c_sample(Sigma_c_inv[c], beta_c[c], y[c], X[c,:,:], Z, ZZ, N, rng) for c in range(C)]
            samples['gamma_c'].append(gamma_c.copy())

            Sigma_c = [Sigma_c_sample(beta_c[c].reshape(K, N, order='F'), gamma_c[c].reshape(Z_width, N, order='F'), Y[c,:,:], X[c,:,:], Z, T, rng) for c in range(C)]
            Sigma_c_inv = [np.linalg.inv(Sigma_c[c]) for c in range(C)]
            samples['Sigma_c'].append(Sigma_c.copy())

        all_chains_data.append(samples)

    ess, r_hat = _compute_diagnostics(all_chains_data, n_burnin)

    # remove burn-in
    post_burnin_samples = {k: np.concatenate([chain[k][n_burnin:] for chain in all_chains_data], axis=0)
                          for k in all_chains_data[0].keys()}
    post_burnin_samples["delta_c"] = [
        [np.concatenate([post_burnin_samples["beta_c"][t][c], post_burnin_samples["gamma_c"][t][c]])
        for c in range(C)]
        for t in range(len(post_burnin_samples["beta_c"]))
    ]
    return post_burnin_samples, ess, r_hat
