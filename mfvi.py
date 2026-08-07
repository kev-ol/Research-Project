# Import libraries

import numpy as np

"""MFVI Update Functions"""

def calc_V_delta(mu_lambda_inv, mu_sigma_inv, FF, Big_S, idx_deltac, size_deltac, Pc, C):
    """Compute the full joint covariance matrix V_delta over [beta_0, delta_1,
    ..., delta_C] by direct inversion of the joint precision matrix.

    Parameters
    ----------
    mu_lambda_inv : float
        Current expectation of 1/lambda under q(lambda).
    mu_sigma_inv : list of length C of numpy.ndarray of shape (N, N)
        Per-country expected precision of Sigma_c.
    FF : sequence of length C of numpy.ndarray of shape (K+Z_width, K+Z_width)
        Per-country F_c.T @ F_c matrices.
    Big_S : numpy.ndarray of shape (size_delta, size_delta)
        Combined prior precision structure over the full delta vector
        [beta_0, delta_1, ..., delta_C], as built in `data_prep.prep_data`.
    idx_deltac : list of length C of int
        Starting index of each country's delta_c block within delta.
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
    numpy.ndarray of shape (size_delta, size_delta)
        Full joint covariance matrix V_delta over [beta_0, delta_1, ..., delta_C].
    """
    precision = mu_lambda_inv * Big_S.copy()

    for c in range(C):
        start = idx_deltac[c]
        likelihood_precision = np.kron(mu_sigma_inv[c], FF[c])

        # S_deltac places this into the delta_c block, Pc reorders
        PtLP = Pc.T @ likelihood_precision @ Pc  # (size_deltac, size_deltac)
        precision[start:start+size_deltac, start:start+size_deltac] += PtLP

    return np.linalg.inv(precision)


def calc_mu_delta(V_delta, mu_sigma_inv, Y, F, idx_deltac, size_deltac, Pc, C):
    """Compute the full joint mean vector mu_delta over [beta_0, delta_1, ...,
    delta_C].

    Parameters
    ----------
    V_delta : numpy.ndarray of shape (size_delta, size_delta)
        Full joint covariance matrix, as returned by `calc_V_delta`
        (or `calc_V_delta`).
    mu_sigma_inv : list of length C of numpy.ndarray of shape (N, N)
        Per-country expected precision of Sigma_c.
    Y : numpy.ndarray of shape (C, T, N)
        Endogenous panel data.
    F : sequence of length C of numpy.ndarray of shape (T, K+Z_width)
        Per-country design matrices (all regressors).
    idx_deltac : list of length C of int
        Starting index of each country's delta_c block within delta.
    size_deltac : int
        Dimension of the stacked delta_c = [beta_c, gamma_c] vector.
    Pc : numpy.ndarray of shape (size_deltac, size_deltac)
        Reordering matrix mapping stacked [beta_c, gamma_c] to the
        equation-interleaved delta_c ordering.
    C : int
        Number of countries.

    Returns
    -------
    numpy.ndarray of shape (size_delta,)
        Full joint mean vector mu_delta over [beta_0, delta_1, ..., delta_C].
    """
    sum = np.zeros(V_delta.shape[0])
    for c in range(C):
        start = idx_deltac[c]
        # use (A kron B) vec(X) = vec(A X B.T)
        sum[start : start + size_deltac] += Pc.T @ (F[c].T @ Y[c, :, :] @ mu_sigma_inv[c]).flatten(order='F')
    return V_delta @ sum

def calc_S_bar_sigma(mu_delta, V_delta, Y, F, FF, idx_deltac, size_deltac, Z_width, Pc, C, N, K):
    """Compute the per-country expected residual-sum-of-squares-plus-uncertainty
    matrix used to update the expected precision of Sigma_c.

    Parameters
    ----------
    mu_delta : numpy.ndarray of shape (size_delta,)
        Full joint mean vector, as returned by `calc_mu_delta`.
    V_delta : numpy.ndarray of shape (size_delta, size_delta)
        Full joint covariance matrix, as returned by `calc_V_delta`.
    Y : numpy.ndarray of shape (C, T, N)
        Endogenous panel data.
    F : sequence of length C of numpy.ndarray of shape (T, K+Z_width)
        Per-country design matrices (all regressors).
    FF : sequence of length C of numpy.ndarray of shape (K+Z_width, K+Z_width)
        Per-country F_c.T @ F_c matrices.
    idx_deltac : list of length C of int
        Starting index of each country's delta_c block within delta.
    size_deltac : int
        Dimension of the stacked delta_c = [beta_c, gamma_c] vector.
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
    width = K + Z_width
    S_bar_sigma = [np.eye(N)] * C
    for c in range(C):
        start = idx_deltac[c]
        mu_deltac = mu_delta[start : start + size_deltac]
        vec_Gc = Pc @ mu_deltac
        mu_Gc = vec_Gc.reshape(width, N, order='F')

        V_deltac = V_delta[start : start + size_deltac, start : start + size_deltac]
        Omega_Gc = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                Pc_i = Pc[i*width:(i+1)*width, :]
                Pc_j = Pc[j*width:(j+1)*width, :]
                Omega_Gc[i, j] = np.trace(FF[c] @ Pc_i @ V_deltac @ Pc_j.T)

        S_bar_sigma[c] = (Y[c, :, :] - F[c] @ mu_Gc).T @ (Y[c, :, :] - F[c] @ mu_Gc) + Omega_Gc
    return S_bar_sigma

# Use the corrected derivation of ELBO
def calc_ELBO(V_delta, s_bar, v_bar, S_bar_sigma, T, C):
    """Compute the evidence lower bound (ELBO) for the current MFVI
    variational approximation.

    Parameters
    ----------
    V_delta : numpy.ndarray of shape (size_delta, size_delta)
        Full joint covariance matrix, as returned by `calc_V_delta`.
    s_bar : float
        Shape parameter of lambda's Inverse-Gamma conditional posterior
        (C*N*K - 1).
    v_bar : float
        Scale parameter of lambda's Inverse-Gamma conditional posterior.
    S_bar_sigma : list of length C of numpy.ndarray of shape (N, N)
        Expected scale matrix for each country's Sigma_c.
    T : int
        Number of time periods.
    C : int
        Number of countries.

    Returns
    -------
    float
        The ELBO value.
    """
    _, logdet_V = np.linalg.slogdet(V_delta)
    elbo = logdet_V / 2 - s_bar * np.log(v_bar) / 2
    for c in range(C):
        _, logdet_S = np.linalg.slogdet(S_bar_sigma[c])
        elbo -= T * logdet_S / 2
    return elbo

"""MFVI Loop"""

def run_mfvi(mfvi_pack, Z_width, C, N, K, T):
    """Run the mean-field variational inference (MFVI) coordinate-ascent loop
    until the ELBO converges.

    Parameters
    ----------
    mfvi_pack : dict
        Data pack produced by `data_prep.prep_data`, with (in order) keys
        'Y' (numpy.ndarray, shape (C, T, N)), 'F' (sequence of length C of
        numpy.ndarray, shape (T, K+Z_width)), 'FF' (sequence of length C of
        numpy.ndarray, shape (K+Z_width, K+Z_width)), 'XX' (numpy.ndarray,
        shape (C, K, K)), 'XZ' (numpy.ndarray, shape (C, K, Z_width)), 'ZZ'
        (numpy.ndarray, shape (Z_width, Z_width)), 'idx_deltac' (list of
        int), 'size_gammmac' (int), 'size_deltac' (int), 'Pc' (numpy.ndarray,
        shape (size_deltac, size_deltac)), 'Big_S' (numpy.ndarray, shape
        (size_delta, size_delta)), 'Lambda_inv' (list of length C of
        numpy.ndarray, shape (N*K, N*K)), and 'Lambda_inv_sum'
        (numpy.ndarray, shape (N*K, N*K)).
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

    Returns
    -------
    params : dict
        Dictionary with keys 'mu_delta' (numpy.ndarray, shape (size_delta,)),
        'V_delta' (numpy.ndarray, shape (size_delta, size_delta)), 'v_bar'
        (float), 's_bar' (float), and 'S_bar_sigma' (list of length C of
        numpy.ndarray, shape (N, N)).
    ELBO : list of float
        ELBO value at each coordinate-ascent iteration.
    """
    Y, F, FF, idx_deltac, size_deltac, Pc, Big_S = mfvi_pack.values()

    # chosen initialisations
    mu_lambda_inv = 1e4
    mu_sigma_inv = [T * np.eye(N) for c in range(C)]

    epsilon = 1e-4
    ELBO = []
    s_bar = C*N*K - 1
    while len(ELBO) < 10 or ELBO[-1] - ELBO[-2] > epsilon:
        V_delta = calc_V_delta(mu_lambda_inv, mu_sigma_inv, FF, Big_S, idx_deltac, size_deltac, Pc, C)
        mu_delta = calc_mu_delta(V_delta, mu_sigma_inv, Y, F, idx_deltac, size_deltac, Pc, C)
        v_bar = mu_delta.T @ Big_S @ mu_delta + np.trace(Big_S @ V_delta)
        mu_lambda_inv = s_bar/v_bar
        S_bar_sigma = calc_S_bar_sigma(mu_delta, V_delta, Y, F, FF, idx_deltac, size_deltac, Z_width, Pc, C, N, K)
        mu_sigma_inv = [T * np.linalg.inv(S_bar_sigma[c]) for c in range(C)]
        elbo = calc_ELBO(V_delta, s_bar, v_bar, S_bar_sigma, T, C)
        ELBO.append(elbo)

    params = {
        'mu_delta': mu_delta,
        'V_delta': V_delta,
        'v_bar': v_bar,
        's_bar': s_bar,
        'S_bar_sigma': S_bar_sigma
    }

    return params, ELBO

