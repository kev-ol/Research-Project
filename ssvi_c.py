import numpy as np
import arviz as az

"""SSVI-C Update Functions"""

def calc_V_beta02(lam, V_deltac, Lambda_inv, Lambda_inv_sum, C, N, K):
    """Compute the batched beta_0 covariance matrix, conditional on lambda samples.

    Parameters
    ----------
    lam : float or array_like of shape (n,)
        Sample(s) of the shrinkage parameter lambda; coerced to at least 1-d.
    V_deltac : list of length C of numpy.ndarray of shape (n, size_deltac, size_deltac)
        Per-country delta_c covariance, batched over the lambda samples.
    Lambda_inv : list of length C of numpy.ndarray of shape (N*K, N*K)
        Per-country inverse Minnesota-prior scale matrices.
    Lambda_inv_sum : numpy.ndarray of shape (N*K, N*K)
        Sum of `Lambda_inv` over countries.
    C : int
        Number of countries.
    N : int
        Number of endogenous variables.
    K : int
        Number of regressors per equation.

    Returns
    -------
    numpy.ndarray of shape (n, N*K, N*K)
        Batched beta_0 covariance matrix, one per lambda sample.
    """
    lam = np.atleast_1d(lam)
    n = len(lam)
    inv_lam = (1/lam)[:, None, None]
    precision = np.tile(inv_lam * Lambda_inv_sum, (1, 1, 1)) if False else inv_lam * Lambda_inv_sum
    for c in range(C):
        precision = precision - inv_lam**2 * (Lambda_inv[c] @ V_deltac[c][:, :N*K, :N*K] @ Lambda_inv[c])
    return np.linalg.inv(precision)

def calc_mu_beta02(lam, V_deltac, mu_sigma_inv, V_beta0, Y, F, Lambda_inv, Pc, C, N, K):
    """Compute the batched beta_0 posterior mean, conditional on lambda samples.

    Parameters
    ----------
    lam : float or array_like of shape (n,)
        Sample(s) of the shrinkage parameter lambda; coerced to at least 1-d.
    V_deltac : list of length C of numpy.ndarray of shape (n, size_deltac, size_deltac)
        Per-country delta_c covariance, batched over the lambda samples.
    mu_sigma_inv : list of length C of numpy.ndarray of shape (N, N)
        Per-country expected precision of Sigma_c.
    V_beta0 : numpy.ndarray of shape (n, N*K, N*K)
        Batched beta_0 covariance, as returned by `calc_V_beta02`.
    Y : numpy.ndarray of shape (C, T, N)
        Endogenous panel data.
    F : sequence of length C of numpy.ndarray of shape (T, K+Z_width)
        Per-country design matrices (all regressors).
    Lambda_inv : list of length C of numpy.ndarray of shape (N*K, N*K)
        Per-country inverse Minnesota-prior scale matrices.
    Pc : numpy.ndarray of shape (size_deltac, size_deltac)
        Reordering matrix mapping stacked [beta_c, gamma_c] to the
        equation-interleaved delta_c ordering.
    C : int
        Number of countries.
    N : int
        Number of endogenous variables.
    K : int
        Number of regressors per equation.

    Returns
    -------
    numpy.ndarray of shape (n, N*K)
        Batched beta_0 posterior mean, one per lambda sample.
    """
    lam = np.atleast_1d(lam)
    n = len(lam)
    size_beta0 = V_beta0.shape[-1]
    total = np.zeros((n, size_beta0))
    for c in range(C):
        # use (A kron B) vec(X) = vec(A X B.T)
        term = Pc.T @ (F[c].T @ Y[c, :, :] @ mu_sigma_inv[c]).flatten(order='F')
        term_batch = np.tile(term, (n, 1))
        # V_deltac[c][:, :N*K, :] has shape (n, N*K, size_deltac); apply to term_batch
        proj = np.einsum('nij,nj->ni', V_deltac[c][:, :N*K, :], term_batch)
        total += (1/lam)[:, None] * np.einsum('ij,nj->ni', Lambda_inv[c], proj)
    return np.einsum('nij,nj->ni', V_beta0, total)

def calc_V_deltac2(lam, mu_sigma_inv, FF, Lambda_inv, size_deltac, Pc, C, N, K):
    """Compute the batched per-country delta_c covariance, conditional on lambda samples.

    Parameters
    ----------
    lam : float or array_like of shape (n,)
        Sample(s) of the shrinkage parameter lambda; coerced to at least 1-d.
    mu_sigma_inv : list of length C of numpy.ndarray of shape (N, N)
        Per-country expected precision of Sigma_c.
    FF : sequence of length C of numpy.ndarray of shape (K+Z_width, K+Z_width)
        Per-country F_c.T @ F_c matrices.
    Lambda_inv : list of length C of numpy.ndarray of shape (N*K, N*K)
        Per-country inverse Minnesota-prior scale matrices.
    size_deltac : int
        Dimension of the stacked delta_c = [beta_c, gamma_c] vector.
    Pc : numpy.ndarray of shape (size_deltac, size_deltac)
        Reordering matrix mapping stacked [beta_c, gamma_c] to the
        equation-interleaved delta_c ordering.
    C : int
        Number of countries.
    N : int
        Number of endogenous variables.
    K : int
        Number of regressors per equation.

    Returns
    -------
    list of length C of numpy.ndarray of shape (n, size_deltac, size_deltac)
        Batched delta_c covariance for each country, one matrix per lambda sample.
    """
    lam = np.atleast_1d(lam)
    n = len(lam)
    V_deltac = [np.eye(size_deltac)] * C
    for c in range(C):
        base = Pc.T @ np.kron(mu_sigma_inv[c], FF[c]) @ Pc
        precision = np.tile(base, (n, 1, 1))
        precision[:, :N*K, :N*K] += (1/lam)[:, None, None] * Lambda_inv[c]
        V_deltac[c] = np.linalg.inv(precision)
    return V_deltac

def calc_mu_deltac2(lam, beta0, V_deltac, mu_sigma_inv, Y, F, Lambda_inv, size_deltac, Pc, C, N, K):
    """Compute the batched per-country delta_c posterior mean, conditional on lambda
    (and, optionally, paired beta_0) samples.

    Parameters
    ----------
    lam : float or array_like of shape (n,)
        Sample(s) of the shrinkage parameter lambda; coerced to at least 1-d.
    beta0 : array_like of shape (N*K,) or (n, N*K)
        beta_0 mean(s). If 2-d, `beta0.shape[0]` must equal `len(lam)` and rows
        are paired one-to-one with the lambda samples; if 1-d, it is broadcast
        against all lambda samples.
    V_deltac : list of length C of numpy.ndarray of shape (n, size_deltac, size_deltac)
        Per-country delta_c covariance, batched over the lambda samples, as
        returned by `calc_V_deltac2`.
    mu_sigma_inv : list of length C of numpy.ndarray of shape (N, N)
        Per-country expected precision of Sigma_c.
    Y : numpy.ndarray of shape (C, T, N)
        Endogenous panel data.
    F : sequence of length C of numpy.ndarray of shape (T, K+Z_width)
        Per-country design matrices (all regressors).
    Lambda_inv : list of length C of numpy.ndarray of shape (N*K, N*K)
        Per-country inverse Minnesota-prior scale matrices.
    size_deltac : int
        Dimension of the stacked delta_c = [beta_c, gamma_c] vector.
    Pc : numpy.ndarray of shape (size_deltac, size_deltac)
        Reordering matrix mapping stacked [beta_c, gamma_c] to the
        equation-interleaved delta_c ordering.
    C : int
        Number of countries.
    N : int
        Number of endogenous variables.
    K : int
        Number of regressors per equation.

    Returns
    -------
    list of length C of numpy.ndarray of shape (n, size_deltac)
        Batched delta_c posterior mean for each country, one vector per lambda
        (and paired beta_0) sample.
    """
    lam = np.atleast_1d(lam)
    n = len(lam)
    beta0 = np.asarray(beta0)
    batched_beta0 = beta0.ndim == 2

    if batched_beta0:
        assert beta0.shape[0] == n, "lam and beta0 batch sizes must match for pairing"

    mu_deltac = [np.zeros(shape=size_deltac)] * C
    for c in range(C):
        # use (A kron B) vec(X) = vec(A X B.T)
        term = Pc.T @ (F[c].T @ Y[c, :, :] @ mu_sigma_inv[c]).flatten(order='F')
        term_batch = np.tile(term, (n, 1))                         

        if batched_beta0:
            beta_transformed = np.einsum('ij,nj->ni', Lambda_inv[c], beta0) 
        else:
            beta_transformed = (Lambda_inv[c] @ beta0)[None, :]

        beta_term = (1.0/lam)[:, None] * beta_transformed
        term_batch[:, :N*K] += beta_term
        mu_deltac[c] = np.matmul(V_deltac[c], term_batch[..., None]).squeeze(-1)
    return mu_deltac

def calc_D2(lam, mu_sigma_inv, Y, F, FF, Lambda_inv, Lambda_inv_sum, size_deltac, Pc, C, N, K):
    """Compute the per-country D statistic (expected quadratic form used in the
    score-function update of q(lambda)) for a single scalar lambda value.

    Parameters
    ----------
    lam : float
        A single lambda value (internally broadcast to a length-1 batch, then
        the batch dimension is squeezed out of the result).
    mu_sigma_inv : list of length C of numpy.ndarray of shape (N, N)
        Per-country expected precision of Sigma_c.
    Y : numpy.ndarray of shape (C, T, N)
        Endogenous panel data.
    F : sequence of length C of numpy.ndarray of shape (T, K+Z_width)
        Per-country design matrices (all regressors).
    FF : sequence of length C of numpy.ndarray of shape (K+Z_width, K+Z_width)
        Per-country F_c.T @ F_c matrices.
    Lambda_inv : list of length C of numpy.ndarray of shape (N*K, N*K)
        Per-country inverse Minnesota-prior scale matrices.
    Lambda_inv_sum : numpy.ndarray of shape (N*K, N*K)
        Sum of `Lambda_inv` over countries.
    size_deltac : int
        Dimension of the stacked delta_c = [beta_c, gamma_c] vector.
    Pc : numpy.ndarray of shape (size_deltac, size_deltac)
        Reordering matrix mapping stacked [beta_c, gamma_c] to the
        equation-interleaved delta_c ordering.
    C : int
        Number of countries.
    N : int
        Number of endogenous variables.
    K : int
        Number of regressors per equation.

    Returns
    -------
    list of length C of float
        D statistic for each country, evaluated at `lam`.
    """
    V_deltac = calc_V_deltac2(lam, mu_sigma_inv, FF, Lambda_inv, size_deltac, Pc, C, N, K)

    V_beta0 = calc_V_beta02(lam, V_deltac, Lambda_inv, Lambda_inv_sum, C, N, K)
    mu_beta0 = calc_mu_beta02(lam, V_deltac, mu_sigma_inv, V_beta0, Y, F, Lambda_inv, Pc, C, N, K)

    mu_bar_deltac = calc_mu_deltac2(lam, mu_beta0, V_deltac, mu_sigma_inv, Y, F, Lambda_inv, size_deltac, Pc, C, N, K)

    # squeeze everything once, together, at the end
    V_deltac = [V[0] for V in V_deltac]
    V_beta0 = V_beta0[0]
    mu_beta0 = mu_beta0[0]
    mu_bar_deltac = [m[0] for m in mu_bar_deltac]
    mu_bar_betac = [mu_bar_deltac[c][:N*K] for c in range(C)]

    G = [lam**-1 * V_deltac[c][:N*K,:N*K] @ Lambda_inv[c] - np.eye(N*K) for c in range(C)]

    D = [np.trace(Lambda_inv[c] @ (V_deltac[c][:N*K,:N*K]
                               + np.outer(mu_bar_betac[c]-mu_beta0, mu_bar_betac[c]-mu_beta0)
                               + G[c] @ V_beta0 @ G[c].T))
                               for c in range(C)]

    return D

def calc_q_lambda2(n_steps, s, lam_init, mu_sigma_inv, Y, F, FF, Lambda_inv, Lambda_inv_sum, size_deltac, Pc, C, N, K, rng):
    """Draw a chain of lambda samples via an unadjusted Langevin algorithm (ULA)
    with an RMSProp-style adaptive step size, sampling in log(lambda) space.

    Parameters
    ----------
    n_steps : int
        Number of ULA steps to run.
    s : float
        Base step-size scale.
    lam_init : float
        Initial value of lambda.
    mu_sigma_inv : list of length C of numpy.ndarray of shape (N, N)
        Per-country expected precision of Sigma_c.
    Y : numpy.ndarray of shape (C, T, N)
        Endogenous panel data.
    F : sequence of length C of numpy.ndarray of shape (T, K+Z_width)
        Per-country design matrices (all regressors).
    FF : sequence of length C of numpy.ndarray of shape (K+Z_width, K+Z_width)
        Per-country F_c.T @ F_c matrices.
    Lambda_inv : list of length C of numpy.ndarray of shape (N*K, N*K)
        Per-country inverse Minnesota-prior scale matrices.
    Lambda_inv_sum : numpy.ndarray of shape (N*K, N*K)
        Sum of `Lambda_inv` over countries.
    size_deltac : int
        Dimension of the stacked delta_c = [beta_c, gamma_c] vector.
    Pc : numpy.ndarray of shape (size_deltac, size_deltac)
        Reordering matrix mapping stacked [beta_c, gamma_c] to the
        equation-interleaved delta_c ordering.
    C : int
        Number of countries.
    N : int
        Number of endogenous variables.
    K : int
        Number of regressors per equation.
    rng : numpy.random.Generator
        Random number generator used to draw the ULA innovation at each step.

    Returns
    -------
    lams : numpy.ndarray of shape (n_steps,)
        Sampled lambda chain (exponentiated log-lambda samples).
    Ds : numpy.ndarray of shape (n_steps, C)
        D statistic (per country) recorded at each step of the chain.
    """
    log_lams = np.zeros(n_steps)
    Ds = np.zeros((n_steps, C))
    # initialise log-lambda and v
    l = np.log(lam_init)
    v = 0
    # set decay=0.9
    beta = 0.9
    for n in range(n_steps):
        lam = np.exp(l)
        D = calc_D2(lam, mu_sigma_inv, Y, F, FF, Lambda_inv, Lambda_inv_sum, size_deltac, Pc, C, N, K)
        Ds[n] = D
        log_lams[n] = l
        # score function after transforming density to log space
        score = np.sum(D)/(2*lam) - (C*N*K - 1)/2
        v = beta * v + (1 - beta) * score**2
        step_size = s / (np.sqrt(v) + 1e-6)
        max_tries = 20
        for _ in range(max_tries):
            l_new = l + step_size*score + np.sqrt(2*step_size)*rng.normal()
            # clip log-lambda for numerical stability
            if -50 < l_new < 50:
                break
        else:
            # fallback if it never lands in range
            l_new = np.clip(l_new, -50, 50)  

        l = l_new
    lams = np.exp(log_lams)
    return lams, Ds

def calc_exp_lambda2(lams, mu_sigma_inv, Ds, Y, F, FF, Lambda_inv, Lambda_inv_sum, size_deltac, Pc, C, N, K):
    """Compute Monte Carlo expectations (over sampled lambda values) of the
    quantities needed to update the CAVI blocks and the ELBO.

    Parameters
    ----------
    lams : array_like of shape (n,)
        Chain samples of lambda (post burn-in).
    mu_sigma_inv : list of length C of numpy.ndarray of shape (N, N)
        Per-country expected precision of Sigma_c.
    Ds : numpy.ndarray of shape (n, C)
        D statistic per step per country, recorded alongside `lams`
        (see `calc_q_lambda2`).
    Y : numpy.ndarray of shape (C, T, N)
        Endogenous panel data.
    F : sequence of length C of numpy.ndarray of shape (T, K+Z_width)
        Per-country design matrices (all regressors).
    FF : sequence of length C of numpy.ndarray of shape (K+Z_width, K+Z_width)
        Per-country F_c.T @ F_c matrices.
    Lambda_inv : list of length C of numpy.ndarray of shape (N*K, N*K)
        Per-country inverse Minnesota-prior scale matrices.
    Lambda_inv_sum : numpy.ndarray of shape (N*K, N*K)
        Sum of `Lambda_inv` over countries.
    size_deltac : int
        Dimension of the stacked delta_c = [beta_c, gamma_c] vector.
    Pc : numpy.ndarray of shape (size_deltac, size_deltac)
        Reordering matrix mapping stacked [beta_c, gamma_c] to the
        equation-interleaved delta_c ordering.
    C : int
        Number of countries.
    N : int
        Number of endogenous variables.
    K : int
        Number of regressors per equation.

    Returns
    -------
    exp_mu_deltac : list of length C of numpy.ndarray of shape (size_deltac,)
        Monte Carlo mean of delta_c over the lambda samples, per country.
    cov_deltac : list of length C of numpy.ndarray of shape (size_deltac, size_deltac)
        Total covariance of delta_c (law-of-total-variance decomposition,
        accounting for lambda uncertainty), per country.
    mu_log_lambda : float
        Mean of log(lambda) over the samples.
    mu_log_q_lambda : float
        Monte Carlo (Vasicek spacing) estimate of E[log q(lambda)].
    exp_logdet_V_beta0 : float
        Expected log-determinant of the beta_0 covariance over lambda samples.
    exp_logdet_V_deltac : list of length C of float
        Expected log-determinant of the delta_c covariance, per country.
    mu_lambda_inv_D : float
        Mean, over the samples, of sum_c D_c / lambda.
    """
    # expectations for other updates
    lams = np.atleast_1d(lams)
    inv_lams = 1/lams

    V_deltac = calc_V_deltac2(lams, mu_sigma_inv, FF, Lambda_inv, size_deltac, Pc, C, N, K)

    V_beta0 = calc_V_beta02(lams, V_deltac, Lambda_inv, Lambda_inv_sum, C, N, K)
    mu_beta0 = calc_mu_beta02(lams, V_deltac, mu_sigma_inv, V_beta0, Y, F, Lambda_inv, Pc, C, N, K)

    mu_bar_deltac = calc_mu_deltac2(lams, mu_beta0, V_deltac, mu_sigma_inv, Y, F, Lambda_inv, size_deltac, Pc, C, N, K)
    exp_mu_deltac = [mu_bar_deltac[c].mean(axis=0) for c in range(C)]

    # calculating unconditional cov_deltac
    cov_term1 = [V_deltac[c].mean(axis=0) for c in range(C)]
    core = [Lambda_inv[c] @ V_beta0 @ Lambda_inv[c] for c in range(C)]
    cov_term2 = [(inv_lams[:, None, None]**2 * (V_deltac[c][:, :, :N*K] @ core[c] @ V_deltac[c][:, :N*K, :])).mean(axis=0)
        for c in range(C)]
    cov_term3 = [np.cov(mu_bar_deltac[c], rowvar=False) for c in range(C)]
    cov_deltac = [cov_term1[c] + cov_term2[c] + cov_term3[c] for c in range(C)]

    # term for ELBO
    log_lams = np.log(lams)
    mu_log_lambda = np.mean(log_lams)

    # entropy for ELBO using Vasicek method
    sorted_log_lams = np.sort(log_lams)
    n = len(sorted_log_lams)
    # window size, standard sqrt(n) choice for Vasicek's estimator
    m = int(np.sqrt(n))
    # spacing between order statistics 2m apart
    diffs = sorted_log_lams[2*m:] - sorted_log_lams[:-2*m]
    # guard against zero/negative spacing from ties or numerical error
    diffs = np.maximum(diffs, 1e-12)
    # entropy estimate, adjusted for the u=log(lambda) transform (Jacobian)
    mu_log_q_lambda = -np.mean(np.log(n * diffs / (2*m))) - mu_log_lambda

    # other terms for ELBO
    logdet_V_beta0 = np.linalg.slogdet(V_beta0)[1]
    exp_logdet_V_beta0 = logdet_V_beta0.mean(axis=0)
    logdet_V_deltac = [np.linalg.slogdet(V_deltac[c])[1] for c in range(C)]
    exp_logdet_V_deltac = [logdet_V_deltac[c].mean(axis=0) for c in range(C)]

    mu_lambda_inv_D = np.mean(np.sum(Ds, axis=1) / lams)

    return exp_mu_deltac, cov_deltac, mu_log_lambda, mu_log_q_lambda, exp_logdet_V_beta0, exp_logdet_V_deltac, mu_lambda_inv_D

def calc_S_bar_sigma2(exp_mu_deltac, cov_deltac, Y, F, FF, Z_width, Pc, C, N, K):
    """Compute the per-country expected residual-sum-of-squares-plus-uncertainty
    matrix used to update the expected precision of Sigma_c.

    Parameters
    ----------
    exp_mu_deltac : list of length C of numpy.ndarray of shape (size_deltac,)
        Expected delta_c vector per country.
    cov_deltac : list of length C of numpy.ndarray of shape (size_deltac, size_deltac)
        Covariance of delta_c per country.
    Y : numpy.ndarray of shape (C, T, N)
        Endogenous panel data.
    F : sequence of length C of numpy.ndarray of shape (T, K+Z_width)
        Per-country design matrices (all regressors).
    FF : sequence of length C of numpy.ndarray of shape (K+Z_width, K+Z_width)
        Per-country F_c.T @ F_c matrices.
    Z_width : int
        Number of non-exchangeable regressors per equation.
    Pc : numpy.ndarray of shape (size_deltac, size_deltac)
        Reordering matrix mapping stacked [beta_c, gamma_c] to the
        equation-interleaved delta_c ordering.
    C : int
        Number of countries.
    N : int
        Number of endogenous variables.
    K : int
        Number of regressors per equation.

    Returns
    -------
    list of length C of numpy.ndarray of shape (N, N)
        Expected scale matrix S_bar_sigma_c for each country's Sigma_c update.
    """
    width = K+Z_width
    S_bar_sigma = [np.eye(N)] * C
    for c in range(C):
        vec_Gc = Pc @ exp_mu_deltac[c]
        mu_Gc = vec_Gc.reshape(width, N, order='F')

        Omega_Gc = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                Pc_i = Pc[i*width:(i+1)*width, :]
                Pc_j = Pc[j*width:(j+1)*width, :]
                Omega_Gc[i, j] = np.trace(FF[c] @ Pc_i @ cov_deltac[c] @ Pc_j.T)

        S_bar_sigma[c] = (Y[c, :, :] - F[c] @ mu_Gc).T @ (Y[c, :, :] - F[c] @ mu_Gc) + Omega_Gc
    return S_bar_sigma

def calc_ELBO2(exp_logdet_V_beta0, exp_logdet_V_deltac, S_bar_sigma, mu_log_lambda, mu_lambda_inv_D, mu_log_q_lambda, C, N, K, T):
    """Compute the evidence lower bound (ELBO) for the current variational
    approximation.

    Parameters
    ----------
    exp_logdet_V_beta0 : float
        Expected log-determinant of the beta_0 covariance over lambda samples.
    exp_logdet_V_deltac : list of length C of float
        Expected log-determinant of the delta_c covariance, per country.
    S_bar_sigma : list of length C of numpy.ndarray of shape (N, N)
        Expected scale matrix for each country's Sigma_c.
    mu_log_lambda : float
        Mean of log(lambda) over the samples.
    mu_lambda_inv_D : float
        Mean, over the samples, of sum_c D_c / lambda.
    mu_log_q_lambda : float
        Monte Carlo estimate of E[log q(lambda)].
    C : int
        Number of countries.
    N : int
        Number of endogenous variables.
    K : int
        Number of regressors per equation.
    T : int
        Number of time periods.

    Returns
    -------
    float
        The ELBO value.
    """
    elbo = (exp_logdet_V_beta0 + np.sum((exp_logdet_V_deltac)) - (C*N*K + 1)* mu_log_lambda - mu_lambda_inv_D)/2 - mu_log_q_lambda
    for c in range(C):
        _, logdet_S = np.linalg.slogdet(S_bar_sigma[c])
        elbo -= T * logdet_S / 2
    return elbo


"""SSVI-C Loop"""

def run_ssvi_c(ssvi_i_pack, Z_width, C, N, K, T, n_steps=1000, s = 0.01, n_burnin = 100, rng=None):
    """Run the SSVI-C (semi-structured variational inference, correlated-lambda
    variant) coordinate-ascent loop until the ELBO converges.

    Parameters
    ----------
    ssvi_i_pack : dict
        Data pack produced by `data_prep.prep_data`, with (in order) keys
        'Y' (numpy.ndarray, shape (C, T, N)), 'F' (sequence of length C of
        numpy.ndarray, shape (T, K+Z_width)), 'FF' (sequence of length C of
        numpy.ndarray, shape (K+Z_width, K+Z_width)), 'idx_deltac' (list of
        int), 'size_deltac' (int), 'Pc' (numpy.ndarray, shape
        (size_deltac, size_deltac)), 'Lambda_inv' (list of length C of
        numpy.ndarray, shape (N*K, N*K)), and 'Lambda_inv_sum' (numpy.ndarray,
        shape (N*K, N*K)).
    Z_width : int
        Number of non-exchangeable regressors per equation.
    C : int
        Number of countries.
    N : int
        Number of endogenous variables.
    K : int
        Number of regressors per equation.
    T : int
        Number of time periods.
    n_steps : int, optional
        Number of post-burn-in ULA steps drawn per outer iteration. Default is 1000.
    s : float, optional
        Base ULA step-size scale. Default is 0.01.
    n_burnin : int, optional
        Number of initial ULA steps discarded per outer iteration. Default is 100.
    rng : int, numpy.random.SeedSequence, numpy.random.Generator, or None, optional
        Source of randomness for the Langevin (ULA) chain. If None (default),
        a fresh, non-reproducible generator is used.

    Returns
    -------
    params : dict
        Dictionary with keys 'q_lambda' (numpy.ndarray, shape (n_steps,), the
        converged lambda chain), 'S_bar_sigma' (list of length C of
        numpy.ndarray, shape (N, N)), and 'cov_deltac' (list of length C of
        numpy.ndarray, shape (size_deltac, size_deltac)).
    ELBO : list of float
        ELBO value at each outer coordinate-ascent iteration.
    ess_list : list of float
        Effective sample size of the log-lambda chain at each outer iteration.
    log_lams_history : list of numpy.ndarray
        log(lambda) chain samples recorded at each outer iteration.
    """
    Y, F, FF, idx_deltac, size_deltac, Pc, Lambda_inv, Lambda_inv_sum = ssvi_i_pack.values()
    rng = np.random.default_rng(rng)

    # chosen initialisations
    lam_init = 1e-4
    mu_sigma_inv = [T * np.eye(N) for c in range(C)]

    epsilon = 0.05
    ELBO = []
    ess_list = []
    log_lams_history = []

    while len(ELBO) < 10 or np.mean([abs(ELBO[-i] - ELBO[-i-1]) for i in range(1, 4)]) > epsilon:
        q_lambda, Ds = calc_q_lambda2(n_steps+n_burnin, s, lam_init, mu_sigma_inv, Y, F, FF, Lambda_inv, Lambda_inv_sum, size_deltac, Pc, C, N, K, rng)
        q_lambda = q_lambda[n_burnin:]
        Ds = Ds[n_burnin:]
        log_lams = np.log(q_lambda)
        log_lams_history.append(log_lams.copy())
        ess_val = az.ess(log_lams[None, :]).item()
        ess_list.append(ess_val)
        lam_init = q_lambda[-1]
        step_size = s * np.var(log_lams)
        exp_mu_deltac, cov_deltac, mu_log_lambda, mu_log_q_lambda, exp_logdet_V_beta0, exp_logdet_V_deltac, mu_lambda_inv_D = calc_exp_lambda2(
            q_lambda, mu_sigma_inv, Ds, Y, F, FF, Lambda_inv, Lambda_inv_sum, size_deltac, Pc, C, N, K)

        S_bar_sigma = calc_S_bar_sigma2(exp_mu_deltac, cov_deltac, Y, F, FF, Z_width, Pc, C, N, K)
        mu_sigma_inv = [T * np.linalg.inv(S_bar_sigma[c]) for c in range(C)]
        elbo = calc_ELBO2(exp_logdet_V_beta0, exp_logdet_V_deltac, S_bar_sigma, mu_log_lambda, mu_lambda_inv_D, mu_log_q_lambda, C, N, K, T)
        ELBO.append(elbo)

    params = {
        'q_lambda': q_lambda,
        'S_bar_sigma': S_bar_sigma,
        'cov_deltac': cov_deltac
    }

    return params, ELBO, ess_list, log_lams_history
