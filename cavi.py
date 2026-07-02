# Import libraries

import numpy as np

"""CAVI Update Functions"""
# Lee-Wand streamlined version (still to be fixd)
def calc_V_delta(mu_lambda_inv, mu_sigma_inv, XX, XZ, ZZ, size_deltac, Lambda_inv, Lambda_inv_sum, C, N, K):
    NK = N * K
    N_z = size_deltac - NK
    size_c = NK + N_z
    total_size = NK + C * size_c
    
    total = mu_lambda_inv * Lambda_inv_sum
    H = []

    for c in range(C):
        top_left = mu_lambda_inv * Lambda_inv[c] + np.kron(mu_sigma_inv[c], XX[c])
        top_right = np.kron(mu_sigma_inv[c], XZ[c])
        bottom_left = top_right.T
        bottom_right = np.kron(mu_sigma_inv[c], ZZ)

        H_c_inv = np.block([[top_left, top_right],[bottom_left, bottom_right]])
        H_c = np.linalg.inv(H_c_inv)
        H.append(H_c)

        total -= mu_lambda_inv**2 * (Lambda_inv[c] @ H_c[:N*K, :N*K] @ Lambda_inv[c])

    V_beta0 = np.linalg.inv(total)

    V_delta = np.zeros((total_size, total_size))
    V_delta[:NK, :NK] = V_beta0

    for c in range(C):
        row = NK + c * size_c
        
        V_dc = H[c] + H[c][:, :NK] @ (mu_lambda_inv**2 * Lambda_inv[c] @ V_beta0 @ Lambda_inv[c]) @ H[c][:NK, :]
        V_cross = -V_beta0 @ (mu_lambda_inv * Lambda_inv[c] @ H[c][:NK, :])

        V_delta[row:row+size_c, row:row+size_c] = V_dc
        V_delta[:NK, row:row+size_c] = V_cross
        V_delta[row:row+size_c, :NK] = V_cross.T

    return V_delta

# basic version currently used
def calc_V_delta_naive(mu_lambda_inv, mu_sigma_inv, FF, Big_S, idx_deltac, size_deltac, Pc, C, N, K):
    
    precision = mu_lambda_inv * Big_S.copy()
    
    for c in range(C):
        start = idx_deltac[c]
        likelihood_precision = np.kron(mu_sigma_inv[c], FF[c])  # (N*(K+1), N*(K+1))
        
        # S_deltac places this into the delta_c block, Pc reorders
        PtLP = Pc.T @ likelihood_precision @ Pc  # (size_deltac, size_deltac)
        precision[start:start+size_deltac, start:start+size_deltac] += PtLP
    
    return np.linalg.inv(precision)


def calc_mu_delta(V_delta, mu_sigma_inv, Y, F, idx_deltac, size_deltac, Pc, C):
    sum = np.zeros(V_delta.shape[0])
    for c in range(C):
        start = idx_deltac[c]
        # (A kron B) vec(X) = vec(A X B.T)
        sum[start : start + size_deltac] += Pc.T @ (F[c].T @ Y[c, :, :] @ mu_sigma_inv[c]).flatten(order='F')
    return V_delta @ sum

def calc_S_bar_sigma(mu_delta, V_delta, Y, F, FF, idx_deltac, size_deltac, Pc, C, N, K):
    S_bar_sigma = [np.eye(N)] * C
    for c in range(C):
        start = idx_deltac[c]
        mu_deltac = mu_delta[start : start + size_deltac]
        vec_Gc = Pc @ mu_deltac
        mu_Gc = vec_Gc.reshape(K+1, N, order='F')

        V_deltac = V_delta[start : start + size_deltac, start : start + size_deltac]
        Omega_Gc = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                Pc_i = Pc[i*(K+1):(i+1)*(K+1), :]
                Pc_j = Pc[j*(K+1):(j+1)*(K+1), :]
                Omega_Gc[i, j] = np.trace(FF[c] @ Pc_i @ V_deltac @ Pc_j.T)

        S_bar_sigma[c] = (Y[c, :, :] - F[c] @ mu_Gc).T @ (Y[c, :, :] - F[c] @ mu_Gc) + Omega_Gc
    return S_bar_sigma

# Use the corrected derivation of ELBO
def calc_ELBO(V_delta, s_bar, v_bar, S_bar_sigma, T, C):
    _, logdet_V = np.linalg.slogdet(V_delta)
    elbo = logdet_V / 2 - s_bar * np.log(v_bar) / 2
    for c in range(C):
        _, logdet_S = np.linalg.slogdet(S_bar_sigma[c])
        elbo -= T * logdet_S / 2
    return elbo

"""CAVI Loop"""

def run_cavi(cavi_pack, C, N, K, T):
    Y, F, FF, XX, XZ, ZZ, idx_deltac, size_gammmac, size_deltac, Pc, Big_S, Lambda_inv, Lambda_inv_sum = cavi_pack.values()

    # chosen initialisations
    mu_lambda_inv = 1e4
    mu_sigma_inv = [T * np.eye(N) for c in range(C)]

    epsilon = 1e-4
    ELBO = []
    s_bar = C*N*K - 1
    while len(ELBO) < 10 or ELBO[-1] - ELBO[-2] > epsilon:
        V_delta = calc_V_delta_naive(mu_lambda_inv, mu_sigma_inv, F, Big_S, idx_deltac, size_deltac, Pc, C, N, K)
        mu_delta = calc_mu_delta(V_delta, mu_sigma_inv, Y, F, idx_deltac, size_deltac, Pc, C)
        v_bar = mu_delta.T @ Big_S @ mu_delta + np.trace(Big_S @ V_delta)
        mu_lambda_inv = s_bar/v_bar
        S_bar_sigma = calc_S_bar_sigma(mu_delta, V_delta, Y, F, FF, idx_deltac, size_deltac, Pc, C, N, K)
        mu_sigma_inv = [T * np.linalg.inv(S_bar_sigma[c]) for c in range(C)]
        ELBO.append(calc_ELBO(V_delta, s_bar, v_bar, S_bar_sigma, T, C))

    params = {
        'mu_delta': mu_delta,
        'V_delta': V_delta,
        'v_bar': v_bar,
        's_bar': s_bar,
        'S_bar_sigma': S_bar_sigma
    }
    
    return params, ELBO


