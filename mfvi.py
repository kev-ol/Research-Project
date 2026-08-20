"""Mean-field variational inference (MFVI) baseline for the hierarchical Panel VAR."""

# Import libraries

import numpy as np

### MFVI Update Functions ###

def calc_V_delta(mu_lambda_inv, mu_sigma_inv, FF, Big_S, idx_deltac, size_deltac, Pc, C):
    """Compute the full joint covariance matrix V_delta over [beta_0, delta_1,
    ..., delta_C]."""
    precision = mu_lambda_inv * Big_S.copy()

    for c in range(C):
        start = idx_deltac[c]
        likelihood_precision = np.kron(mu_sigma_inv[c], FF[c])
        PtLP = Pc.T @ likelihood_precision @ Pc 
        precision[start:start+size_deltac, start:start+size_deltac] += PtLP

    return np.linalg.inv(precision)


def calc_mu_delta(V_delta, mu_sigma_inv, Y, F, idx_deltac, size_deltac, Pc, C):
    """Compute the full joint mean vector mu_delta over [beta_0, delta_1, ...,
    delta_C]."""
    sum = np.zeros(V_delta.shape[0])
    for c in range(C):
        start = idx_deltac[c]
        # use (A kron B) vec(X) = vec(A X B.T)
        sum[start : start + size_deltac] += Pc.T @ (F[c].T @ Y[c, :, :] @ mu_sigma_inv[c]).flatten(order='F')
    return V_delta @ sum

def calc_S_bar_sigma(mu_delta, V_delta, Y, F, FF, idx_deltac, size_deltac, Z_width, Pc, C, N, K):
    """Compute the per-country expected residual-sum-of-squares-plus-uncertainty
    matrix used to update the expected precision of Sigma_c."""
    width = K + Z_width
    S_bar_sigma = [np.eye(N)] * C
    for c in range(C):
        start = idx_deltac[c]
        mu_deltac = mu_delta[start : start + size_deltac]
        vec_Gc = Pc @ mu_deltac
        # get G_c term
        mu_Gc = vec_Gc.reshape(width, N, order='F')

        V_deltac = V_delta[start : start + size_deltac, start : start + size_deltac]
        # create Omega matrix
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
    variational approximation."""
    _, logdet_V = np.linalg.slogdet(V_delta)
    elbo = logdet_V / 2 - s_bar * np.log(v_bar) / 2
    for c in range(C):
        _, logdet_S = np.linalg.slogdet(S_bar_sigma[c])
        elbo -= T * logdet_S / 2
    return elbo

### MFVI Loop ###

def run_mfvi(mfvi_pack, Z_width, C, N, K, T):
    """Run the MFVI coordinate-ascent loop until the ELBO converges.

    Parameters
    ----------
    mfvi_pack : dict
        Data pack keys: 'Y', 'F', 'FF', 'idx_deltac',
        'size_deltac', 'Pc', 'Big_S', 'Lambda_inv', 'Lambda_inv_sum'.

    Returns
    -------
    params : dict
        Keys: 'mu_delta', 'V_delta', 'v_bar', 's_bar', 'S_bar_sigma'.
    ELBO : list of float
        ELBO at each iteration.
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

        # update for lambda
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

