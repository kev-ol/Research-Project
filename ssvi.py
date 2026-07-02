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

def calc_V_deltac(lam, mu_sigma_inv, FF, Lambda_inv, size_deltac, Pc, C, N, K):
    lam = np.atleast_1d(lam)
    n = len(lam)
    V_deltac = [np.eye(size_deltac)] * C
    for c in range(C):
        base = Pc.T @ np.kron(mu_sigma_inv[c], FF[c]) @ Pc
        precision = np.tile(base, (n, 1, 1))
        precision[:, :N*K, :N*K] += (1/lam)[:, None, None] * Lambda_inv[c]
        V_deltac[c] = np.linalg.inv(precision)
    return V_deltac

def calc_mu_deltac(lam, beta0, V_deltac, mu_sigma_inv, Y, F, Lambda_inv, size_deltac, Pc, C, N, K):
    lam = np.atleast_1d(lam)
    n = len(lam)
    mu_deltac = [np.zeros(shape=size_deltac)] * C
    for c in range(C):
        term = Pc.T @ (F[c].T @ Y[c, :, :] @ mu_sigma_inv[c]).flatten(order='F')
        beta_term = (1.0/lam)[:, None] * (Lambda_inv[c] @ beta0)[None, :] 
        term_batch = np.tile(term, (n, 1))                                 # (n, size_deltac)
        term_batch[:, :N*K] += beta_term
        mu_deltac[c] = np.matmul(V_deltac[c], term_batch[..., None]).squeeze(-1)
    return mu_deltac

def calc_D(lam, V_beta0, mu_beta0, mu_sigma_inv, Y, F, FF, Lambda_inv, size_deltac, Pc, C, N, K):
    V_deltac = calc_V_deltac(lam, mu_sigma_inv, FF, Lambda_inv, size_deltac, Pc, C, N, K)
    mu_deltac = calc_mu_deltac(lam, mu_beta0, V_deltac, mu_sigma_inv, Y, F, Lambda_inv, size_deltac, Pc, C, N, K)
    mu_bar_deltac = [mu_deltac[c][:N*K] for c in range(C)]

    G = [lam**-1 * V_deltac[c][:N*K,:N*K] @ Lambda_inv[c] - np.eye[N*K] for c in range(C)]

    D = [np.trace(Lambda_inv[c] * (V_deltac[:N*K,:N*K]
                                   + (mu_bar_deltac[c]-mu_beta0) @ (mu_bar_deltac[c]-mu_beta0).T
                                   + G[c] @ V_beta0 @ G[c].T) 
                                   for c in range(C)]
    return D

def calc_q_lambda(n_steps, step_size, lam_init, V_beta0, mu_beta0, mu_sigma_inv, Y, F, FF, Lambda_inv, size_deltac, Pc, C, N, K):
    log_lams = np.zeros(n_steps)
    l = np.log(lam_init)
    for n in range(n_steps):
        lam = np.exp(l)
        D = calc_D(lam, V_beta0, mu_beta0, mu_sigma_inv, Y, F, FF, Lambda_inv, size_deltac, Pc, C, N, K)
        # score function after transforming density to log space
        score = np.sum(D)/(2*lam) - (C*N*K - 1)/2
        l_new = l + step_size * score + np.sqrt(2*step_size) * np.random.normal()
        log_lams[n] = l_new
        l = l_new
    lams = np.exp(log_lams)
    return lams

def calc_exp_lambda(lams, mu_sigma_inv, mu_beta0, Y, F, FF, Lambda_inv, size_deltac, Pc, C, N, K):
    lams = np.atleast_1d(lams)
    inv_lams = 1/lams
    mu_lambda_inv = np.mean(inv_lams)
    V_deltac =  calc_V_deltac(lams, mu_sigma_inv, FF, Lambda_inv, size_deltac, Pc, C, N, K)
    mu_deltac = calc_mu_deltac(lams, mu_beta0, V_deltac, mu_sigma_inv, Y, F, Lambda_inv, size_deltac, Pc, C, N, K)
    mu_lambda1_V = [(inv_lams[:, None, None] * V_deltac[c]).mean(axis=0) for c in range(C)]
    mu_lambda2_V = [(inv_lams[:, None, None]**2 * V_deltac[c]).mean(axis=0) for c in range(C)]
    exp_V_deltac = [V_deltac[c].mean(axis=0) for c in range(C)]
    exp_mu_deltac = [mu_deltac[c].mean(axis=0) for c in range(C)]

    return mu_lambda_inv, mu_lambda1_V, mu_lambda2_V, exp_mu_deltac, exp_V_deltac


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
        V_beta0 = calc_V_beta0(mu_lambda_inv, mu_lambda2_V, Lambda_inv, Lambda_inv_sum, C, N, K)
        mu_beta0 = calc_mu_beta0(mu_lambda1_V, mu_sigma_inv, V_beta0, Y, F, Lambda_inv, Pc, C, N, K)

        q_lambda = calc_q_lambda(n_steps, step_size, lam_init, V_beta0, mu_beta0, mu_sigma_inv, Y, F, FF, Lambda_inv, size_deltac, Pc, C, N, K)
        mu_lambda_inv, mu_lambda1_V, mu_lambda2_V, exp_mu_deltac, exp_V_deltac = calc_exp_lambda(
            q_lambda, mu_sigma_inv, mu_beta0, Y, F, FF, Lambda_inv, size_deltac, Pc, C, N, K)
        V_deltac = calc_V_deltac(lam, mu_sigma_inv, FF, Lambda_inv, size_deltac, Pc, C, N, K)
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