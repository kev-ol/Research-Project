import numpy as np

def calc_V_beta0(mu_lambda_inv, mu_lambda2_V, Lambda_inv, Lambda_inv_sum, C):
    sum = mu_lambda_inv * Lambda_inv_sum
    for c in range(C):
        sum -= Lambda_inv[c] @ mu_lambda2_V[c][BETAC SELECTION] @ Lambda_inv[c]
    return np.linalg.inv(sum)
    
def calc_mu_beta0(mu_lambda1_V, mu_sigma_inv, V_beta0, Y, F, Lambda_inv, Pc, C):
    sum = np.zeros(V_beta0.shape[0])
    for c in range(C):
        sum += [SELECTION] mu_lambda1_V[c] * Lambda_inv[c] @ Pc.T @ (F[c].T @ Y[c, :, :] @ mu_sigma_inv[c]).flatten(order='F')
    return V_beta0 @ sum

def calc_q_lambda():
    pass
def calc_V_deltac(lam, mu_sigma_inv_c, Lambda_inv_c, Pc, FFc):
    precision = lam**-1 * [SELECTION] Lambda_inv_c + Pc.T @ np.kron(mu_sigma_inv_c, FFc) @ Pc
    return np.linalg.inv(precision)

def calc_mu_deltac(lam, beta0, V_deltac, mu_sigma_inv_c, Yc, Fc, Lambda_inv_c, Pc):
    term = lam**-1 * [SELECTION] Lambda_inv_c @ beta0 + Pc.T @ (Fc.T @ Yc @ mu_sigma_inv_c).flatten(order='F')
    return V_deltac @ term

def calc_S_bar_sigma(mu_delta, V_delta, Y, F, idx_deltac, size_deltac, Pc, C, N, K):
    S_bar_sigma = [np.eye(N)] * C
    for c in range(C):
        start = idx_deltac[c]
        mu_deltac = mu_delta[start : start + size_deltac]
        vec_Gc = Pc @ mu_deltac
        mu_Gc = vec_Gc.reshape(K+1, N, order='F')

        FtF = F[c].T @ F[c]
        V_deltac = V_delta[start : start + size_deltac, start : start + size_deltac]
        Omega_Gc = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                Pc_i = Pc[i*(K+1):(i+1)*(K+1), :]
                Pc_j = Pc[j*(K+1):(j+1)*(K+1), :]
                Omega_Gc[i, j] = np.trace(FtF @ Pc_i @ V_deltac @ Pc_j.T)

        S_bar_sigma[c] = (Y[c, :, :] - F[c] @ mu_Gc).T @ (Y[c, :, :] - F[c] @ mu_Gc) + Omega_Gc
    return S_bar_sigma



def run_ssvi(ssvi_pack, C, N, K, T):
    # Y, F, XX, XZ, ZZ, idx_deltac, size_gammmac, size_deltac, Pc, Big_S, Lambda_inv, Lambda_inv_sum = cavi_pack.values()

    # chosen initialisations
    mu_lambda_inv = 1e4
    mu_lambda_V = 4
    mu_sigma_inv = [T * np.eye(N) for c in range(C)]

    epsilon = 1e-4
    ELBO = []
    
    while len(ELBO) < 10 or ELBO[-1] - ELBO[-2] > epsilon:
        V_delta = calc_V_delta_naive(mu_lambda_inv, mu_sigma_inv, F, Big_S, idx_deltac, size_deltac, Pc, C, N, K)
        mu_delta = calc_mu_delta(V_delta, mu_sigma_inv, Y, F, idx_deltac, size_deltac, Pc, C)
        v_bar = mu_delta.T @ Big_S @ mu_delta + np.trace(Big_S @ V_delta)
        mu_lambda_inv = s_bar/v_bar
        S_bar_sigma = calc_S_bar_sigma(mu_delta, V_delta, Y, F, idx_deltac, size_deltac, Pc, C, N, K)
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