import numpy as np

def calc_V_beta0(mu_lambda_inv, mu_lambda2_V, Lambda_inv, Lambda_inv_sum, C, N, K):
    sum = mu_lambda_inv * Lambda_inv_sum
    for c in range(C):
        sum -= Lambda_inv[c] @ mu_lambda2_V[c][:N*K,:N*K] @ Lambda_inv[c]
    return np.linalg.inv(sum)
    
def calc_mu_beta0(mu_lambda1_V, mu_sigma_inv, V_beta0, Y, F, Lambda_inv, Pc, C, N, K):
    sum = np.zeros(V_beta0.shape[0])
    for c in range(C):
        sum += Lambda_inv[c] @ mu_lambda1_V[c][:N*K,:] @ Pc.T @ (F[c].T @ Y[c, :, :] @ mu_sigma_inv[c]).flatten(order='F')
    return V_beta0 @ sum

def calc_q_lambda():
    pass
def calc_V_deltac(lam, mu_sigma_inv, Lambda_inv, size_deltac, Pc, FF, C, N, K):
    V_deltac = [np.eye(size_deltac)] * C
    for c in range(C):
        precision = Pc.T @ np.kron(mu_sigma_inv[c], FF[c]) @ Pc
        precision[:NK, :NK] += lam**-1 * Lambda_inv[c]
        V_deltac[c] = np.linalg.inv(precision)
    return V_deltac

def calc_mu_deltac(lam, beta0, V_deltac, mu_sigma_inv, Y, F, Lambda_inv, size_deltac Pc, C, N, K):
    mu_deltac = [np.zeros(shape=size_deltac)] * C
    for c in range(C):
        term = Pc.T @ (F[c].T @ Y[c, :, :] @ mu_sigma_inv[c]).flatten(order='F')
        term[:NK] += lam**-1 * Lambda_inv[c] @ beta0
        mu_deltac[c] = V_deltac[c] @ term
    return mu_deltac

def calc_S_bar_sigma(mu_deltac, V_deltac, Y, F, FF, Pc, C, N, K):
    S_bar_sigma = [np.eye(N)] * C
    for c in range(C):
        vec_Gc = Pc @ mu_deltac[c]
        mu_Gc = vec_Gc.reshape(K+1, N, order='F')

        Omega_Gc = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                Pc_i = Pc[i*(K+1):(i+1)*(K+1), :]
                Pc_j = Pc[j*(K+1):(j+1)*(K+1), :]
                Omega_Gc[i, j] = np.trace(FF[c] @ Pc_i @ V_deltac[c] @ Pc_j.T)

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
        V_beta0 = calc_V_beta0(mu_lambda_inv, mu_lambda2_V, Lambda_inv, Lambda_inv_sum, C)
        mu_beta0 = calc_mu_beta0(mu_lambda1_V, mu_sigma_inv, V_beta0, Y, F, Lambda_inv, Pc, C)
        q_lambda = calc_q_lambda(y)
        V_deltac = calc_V_deltac
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