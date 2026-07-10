import numpy as np
def calc_V_beta02(lam, V_deltac, Lambda_inv, Lambda_inv_sum, C, N, K):
    lam = np.atleast_1d(lam)
    n = len(lam)
    inv_lam = (1/lam)[:, None, None]          # (n, 1, 1)
    precision = np.tile(inv_lam * Lambda_inv_sum, (1, 1, 1)) if False else inv_lam * Lambda_inv_sum  # (n, size_beta0, size_beta0)
    for c in range(C):
        precision = precision - inv_lam**2 * (Lambda_inv[c] @ V_deltac[c][:, :N*K, :N*K] @ Lambda_inv[c])
    return np.linalg.inv(precision)            # (n, size_beta0, size_beta0)

def calc_mu_beta02(lam, V_deltac, mu_sigma_inv, V_beta0, Y, F, Lambda_inv, Pc, C, N, K):
    lam = np.atleast_1d(lam)
    n = len(lam)
    size_beta0 = V_beta0.shape[-1]
    total = np.zeros((n, size_beta0))
    for c in range(C):
        term = Pc.T @ (F[c].T @ Y[c, :, :] @ mu_sigma_inv[c]).flatten(order='F')   # (size_deltac,)
        term_batch = np.tile(term, (n, 1))                                          # (n, size_deltac)
        # V_deltac[c][:, :N*K, :] has shape (n, N*K, size_deltac); apply to term_batch
        proj = np.einsum('nij,nj->ni', V_deltac[c][:, :N*K, :], term_batch)         # (n, N*K)
        total += (1/lam)[:, None] * np.einsum('ij,nj->ni', Lambda_inv[c], proj)
    return np.einsum('nij,nj->ni', V_beta0, total)                                  # (n, size_beta0)

def calc_V_deltac2(lam, mu_sigma_inv, FF, Lambda_inv, size_deltac, Pc, C, N, K):
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
    lam = np.atleast_1d(lam)
    n = len(lam)
    beta0 = np.asarray(beta0)
    batched_beta0 = beta0.ndim == 2

    if batched_beta0:
        assert beta0.shape[0] == n, "lam and beta0 batch sizes must match for pairing"

    mu_deltac = [np.zeros(shape=size_deltac)] * C
    for c in range(C):
        term = Pc.T @ (F[c].T @ Y[c, :, :] @ mu_sigma_inv[c]).flatten(order='F')
        term_batch = np.tile(term, (n, 1))                                 # (n, size_deltac)

        if batched_beta0:
            beta_transformed = np.einsum('ij,nj->ni', Lambda_inv[c], beta0)  # (n, N*K), paired per-sample
        else:
            beta_transformed = (Lambda_inv[c] @ beta0)[None, :]              # (1, N*K), broadcasts as before

        beta_term = (1.0/lam)[:, None] * beta_transformed
        term_batch[:, :N*K] += beta_term
        mu_deltac[c] = np.matmul(V_deltac[c], term_batch[..., None]).squeeze(-1)
    return mu_deltac

def calc_D2(lam, mu_sigma_inv, Y, F, FF, Lambda_inv, Lambda_inv_sum, size_deltac, Pc, C, N, K):
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

def calc_q_lambda2(n_steps, step_size, lam_init, mu_sigma_inv, Y, F, FF, Lambda_inv, Lambda_inv_sum, size_deltac, Pc, C, N, K):
    log_lams = np.zeros(n_steps)
    Ds = np.zeros((n_steps, C))
    l = np.log(lam_init)
    for n in range(n_steps):
        lam = np.exp(l)
        D = calc_D2(lam, mu_sigma_inv, Y, F, FF, Lambda_inv, Lambda_inv_sum, size_deltac, Pc, C, N, K)
        Ds[n] = D
        log_lams[n] = l
        # score function after transforming density to log space
        score = np.sum(D)/(2*lam) - (C*N*K - 1)/2
        l = l + step_size * score + np.sqrt(2*step_size) * np.random.normal()
    lams = np.exp(log_lams)
    return lams, Ds

def calc_exp_lambda2(lams, mu_sigma_inv, Ds, Y, F, FF, Lambda_inv, Lambda_inv_sum, size_deltac, Pc, C, N, K):
    lams = np.atleast_1d(lams)
    inv_lams = 1/lams

    V_deltac = calc_V_deltac2(lams, mu_sigma_inv, FF, Lambda_inv, size_deltac, Pc, C, N, K)

    V_beta0 = calc_V_beta02(lams, V_deltac, Lambda_inv, Lambda_inv_sum, C, N, K)
    mu_beta0 = calc_mu_beta02(lams, V_deltac, mu_sigma_inv, V_beta0, Y, F, Lambda_inv, Pc, C, N, K)

    mu_bar_deltac = calc_mu_deltac2(lams, mu_beta0, V_deltac, mu_sigma_inv, Y, F, Lambda_inv, size_deltac, Pc, C, N, K)
    exp_mu_deltac = [mu_bar_deltac[c].mean(axis=0) for c in range(C)]

    cov_term1 = [V_deltac[c].mean(axis=0) for c in range(C)]
    core = [Lambda_inv[c] @ V_beta0 @ Lambda_inv[c] for c in range(C)]
    cov_term2 = [(inv_lams[:, None, None]**2 * (V_deltac[c][:, :, :N*K] @ core[c] @ V_deltac[c][:, :N*K, :])).mean(axis=0)
        for c in range(C)]
    cov_term3 = [np.cov(mu_bar_deltac[c], rowvar=False) for c in range(C)]
    cov_deltac = [cov_term1[c] + cov_term2[c] + cov_term3[c] for c in range(C)]

    log_lams = np.log(lams)
    mu_log_lambda = np.mean(log_lams)

    sorted_log_lams = np.sort(log_lams)
    n = len(sorted_log_lams)
    m = int(np.sqrt(n))
    padded = np.concatenate([np.full(m, sorted_log_lams[0]), sorted_log_lams, np.full(m, sorted_log_lams[-1])])
    diffs = padded[2*m:] - padded[:-2*m]
    mu_log_q_lambda = -np.mean(np.log(n * diffs / (2*m))) - mu_log_lambda

    logdet_V_beta0 = np.linalg.slogdet(V_beta0)[1]
    exp_logdet_V_beta0 = logdet_V_beta0.mean(axis=0)
    logdet_V_deltac = [np.linalg.slogdet(V_deltac[c])[1] for c in range(C)]
    exp_logdet_V_deltac = [logdet_V_deltac[c].mean(axis=0) for c in range(C)]

    mu_lambda_inv_D = np.mean(np.sum(Ds, axis=1) / lams)

    return exp_mu_deltac, cov_deltac, mu_log_lambda, mu_log_q_lambda, exp_logdet_V_beta0, exp_logdet_V_deltac, mu_lambda_inv_D

def calc_S_bar_sigma2(exp_mu_deltac, cov_deltac, Y, F, FF, Z_width, Pc, C, N, K):
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
    elbo = (exp_logdet_V_beta0 + np.sum((exp_logdet_V_deltac)) - (C*N*K + 1)* mu_log_lambda - mu_lambda_inv_D)/2 - mu_log_q_lambda
    for c in range(C):
        _, logdet_S = np.linalg.slogdet(S_bar_sigma[c])
        elbo -= T * logdet_S / 2
    return elbo


def run_ssvi2(ssvi_pack, Z_width, C, N, K, T, n_steps=1000, step_size = 0.01, n_burnin = 100):
    Y, F, FF, idx_deltac, size_deltac, Pc, Lambda_inv, Lambda_inv_sum = ssvi_pack.values()

    # chosen initialisations
    lam_init = 1e-4
    mu_sigma_inv = [T * np.eye(N) for c in range(C)]

    epsilon = 1e-4
    ELBO = []

    while len(ELBO) < 10 or ELBO[-1] - ELBO[-2] > epsilon:
        q_lambda, Ds = calc_q_lambda2(n_steps, step_size, lam_init, mu_sigma_inv, Y, F, FF, Lambda_inv, Lambda_inv_sum, size_deltac, Pc, C, N, K)
        q_lambda = q_lambda[n_burnin:]
        Ds = Ds[n_burnin:]
        lam_init = q_lambda[-1]
        exp_mu_deltac, cov_deltac, mu_log_lambda, mu_log_q_lambda, exp_logdet_V_beta0, exp_logdet_V_deltac, mu_lambda_inv_D = calc_exp_lambda2(
            q_lambda, mu_sigma_inv, Ds, Y, F, FF, Lambda_inv, Lambda_inv_sum, size_deltac, Pc, C, N, K)
        #if len(ELBO)>0:
        #    elbo_after_lambda = calc_ELBO(V_beta0, exp_logdet_V_deltac, S_bar_sigma, mu_log_lambda, mu_lambda_inv_D, mu_log_q_lambda, C, N, K, T)

        S_bar_sigma = calc_S_bar_sigma2(exp_mu_deltac, cov_deltac, Y, F, FF, Z_width, Pc, C, N, K)
        mu_sigma_inv = [T * np.linalg.inv(S_bar_sigma[c]) for c in range(C)]  
        elbo_after_sigma = calc_ELBO2(exp_logdet_V_beta0, exp_logdet_V_deltac, S_bar_sigma, mu_log_lambda, mu_lambda_inv_D, mu_log_q_lambda, C, N, K, T)
        #if len(ELBO)>0:
        #    print(elbo_after_beta0, elbo_after_lambda, elbo_after_sigma)
        ELBO.append(elbo_after_sigma)
    
    params = {
        'q_lambda': q_lambda,
        'S_bar_sigma': S_bar_sigma
    }
    
    return params, ELBO