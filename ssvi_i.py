import numpy as np
import arviz as az

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

def calc_D(lam, V_beta0, mu_beta0, mu_sigma_inv, Y, F, FF, Lambda_inv, size_deltac, Pc, C, N, K):
    V_deltac = calc_V_deltac(lam, mu_sigma_inv, FF, Lambda_inv, size_deltac, Pc, C, N, K)
    mu_bar_deltac = calc_mu_deltac(lam, mu_beta0, V_deltac, mu_sigma_inv, Y, F, Lambda_inv, size_deltac, Pc, C, N, K)

    # calc_D is called with a single scalar lam, so squeeze the batch dim
    V_deltac = [V[0] for V in V_deltac]      # (size_deltac, size_deltac)
    mu_bar_deltac = [m[0] for m in mu_bar_deltac]    # (size_deltac,)
    mu_bar_betac = [mu_bar_deltac[c][:N*K] for c in range(C)]

    G = [lam**-1 * V_deltac[c][:N*K,:N*K] @ Lambda_inv[c] - np.eye(N*K) for c in range(C)]

    D = [np.trace(Lambda_inv[c] @ (V_deltac[c][:N*K,:N*K]
                               + np.outer(mu_bar_betac[c]-mu_beta0, mu_bar_betac[c]-mu_beta0)
                               + G[c] @ V_beta0 @ G[c].T)) 
                               for c in range(C)]
    
    return D

def calc_q_lambda(n_steps, step_size, lam_init, V_beta0, mu_beta0, mu_sigma_inv, Y, F, FF, Lambda_inv, size_deltac, Pc, C, N, K):
    log_lams = np.zeros(n_steps)
    Ds = np.zeros((n_steps, C))
    l = np.log(lam_init)
    for n in range(n_steps):
        lam = np.exp(l)
        D = calc_D(lam, V_beta0, mu_beta0, mu_sigma_inv, Y, F, FF, Lambda_inv, size_deltac, Pc, C, N, K)
        Ds[n] = D
        log_lams[n] = l
        # score function after transforming density to log space
        score = np.sum(D)/(2*lam) - (C*N*K - 1)/2
        max_tries = 20
        for _ in range(max_tries):
            l_new = l + step_size*score + np.sqrt(2*step_size)*np.random.normal()
            if -50 < l_new < 50:
                break
        else:
            l_new = np.clip(l_new, -50, 50)  # fallback if it never lands in range

        l = l_new
    lams = np.exp(log_lams)
    return lams, Ds

def calc_exp_lambda(lams, mu_sigma_inv, mu_beta0, V_beta0, Ds, Y, F, FF, Lambda_inv, size_deltac, Pc, C, N, K):
    lams = np.atleast_1d(lams)
    inv_lams = 1/lams
    mu_lambda_inv = np.mean(inv_lams)
    V_deltac =  calc_V_deltac(lams, mu_sigma_inv, FF, Lambda_inv, size_deltac, Pc, C, N, K)
    mu_bar_deltac = calc_mu_deltac(lams, mu_beta0, V_deltac, mu_sigma_inv, Y, F, Lambda_inv, size_deltac, Pc, C, N, K)
    mu_lambda1_V = [(inv_lams[:, None, None] * V_deltac[c]).mean(axis=0) for c in range(C)]
    mu_lambda2_V = [(inv_lams[:, None, None]**2 * V_deltac[c]).mean(axis=0) for c in range(C)]
    exp_mu_deltac = [mu_bar_deltac[c].mean(axis=0) for c in range(C)]

    cov_term1 = [V_deltac[c].mean(axis=0) for c in range(C)]
    core = [Lambda_inv[c] @ V_beta0 @ Lambda_inv[c] for c in range(C)]   # (N*K, N*K) per country
    cov_term2 = [(inv_lams[:, None, None]**2 * (V_deltac[c][:, :, :N*K] @ core[c] @ V_deltac[c][:, :N*K, :])).mean(axis=0)
        for c in range(C)]
    cov_term3 = [np.cov(mu_bar_deltac[c], rowvar=False) for c in range(C)]
    cov_deltac = [cov_term1[c] + cov_term2[c] + cov_term3[c] for c in range(C)]

    log_lams = np.log(lams)
    mu_log_lambda = np.mean(log_lams)

    sorted_log_lams = np.sort(log_lams)
    n = len(sorted_log_lams)
    m = int(np.sqrt(n))
    diffs = sorted_log_lams[2*m:] - sorted_log_lams[:-2*m]
    mu_log_q_lambda = -np.mean(np.log(n * diffs / (2*m))) - mu_log_lambda  

    """"
    counts, edges = np.histogram(log_lams, bins=50, density=True)
    widths = np.diff(edges)
    mask = counts > 0
    mu_log_q_lambda = np.sum(counts[mask] * np.log(counts[mask]) * widths[mask]) - mu_log_lambda
    """
    logdet_V_deltac = [np.linalg.slogdet(V_deltac[c])[1] for c in range(C)]
    exp_logdet_V_deltac = [logdet_V_deltac[c].mean(axis=0) for c in range(C)]

    mu_lambda_inv_D = np.mean(np.sum(Ds, axis=1) / lams)

    return mu_lambda_inv, mu_lambda1_V, mu_lambda2_V, exp_mu_deltac, cov_deltac, mu_log_lambda, mu_log_q_lambda, exp_logdet_V_deltac, mu_lambda_inv_D


def calc_S_bar_sigma(exp_mu_deltac, cov_deltac, Y, F, FF, Z_width, Pc, C, N, K):
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

def calc_ELBO(V_beta0, exp_logdet_V_deltac, S_bar_sigma, mu_log_lambda, mu_lambda_inv_D, mu_log_q_lambda, C, N, K, T):
    _, logdet_V_beta0 = np.linalg.slogdet(V_beta0)
    elbo = (logdet_V_beta0 + np.sum((exp_logdet_V_deltac)) - (C*N*K + 1)* mu_log_lambda - mu_lambda_inv_D)/2 - mu_log_q_lambda
    for c in range(C):
        _, logdet_S = np.linalg.slogdet(S_bar_sigma[c])
        elbo -= T * logdet_S / 2
    return elbo



def run_ssvi_i(ssvi_i_pack, Z_width, C, N, K, T, n_steps=1000, step_size_init =  0.01, s = 0.01, n_burnin = 100):
    Y, F, FF, idx_deltac, size_deltac, Pc, Lambda_inv, Lambda_inv_sum = ssvi_i_pack.values()

    # chosen initialisations
    lam_init = 1e-4
    mu_lambda_inv = 1e4
    mu_lambda1_V = [mu_lambda_inv * np.eye(size_deltac) for _ in range(C)]
    mu_lambda2_V = [mu_lambda_inv**2 * np.eye(size_deltac) for _ in range(C)]
    mu_sigma_inv = [T * np.eye(N) for c in range(C)]
    step_size = step_size_init

    epsilon = 0.05
    ELBO = []
    ess_list =  []
    log_lams_history = []
    
    while len(ELBO) < 10 or np.mean([abs(ELBO[-i] - ELBO[-i-1]) for i in range(1, 4)]) > epsilon:
        V_beta0 = calc_V_beta0(mu_lambda_inv, mu_lambda2_V, Lambda_inv, Lambda_inv_sum, C, N, K)
        mu_beta0 = calc_mu_beta0(mu_lambda1_V, mu_sigma_inv, V_beta0, Y, F, Lambda_inv, Pc, C, N, K)

        q_lambda, Ds = calc_q_lambda(n_steps+n_burnin, step_size, lam_init, V_beta0, mu_beta0, mu_sigma_inv, Y, F, FF, Lambda_inv, size_deltac, Pc, C, N, K)
        q_lambda = q_lambda[n_burnin:]
        Ds = Ds[n_burnin:]
        log_lams = np.log(q_lambda)    
        log_lams_history.append(log_lams.copy())            
        ess_val = az.ess(log_lams[None, :]).item()
        ess_list.append(ess_val)
        lam_init = q_lambda[-1]
        step_size = s * np.var(log_lams)
        mu_lambda_inv, mu_lambda1_V, mu_lambda2_V, exp_mu_deltac, cov_deltac, mu_log_lambda, mu_log_q_lambda, exp_logdet_V_deltac, mu_lambda_inv_D = calc_exp_lambda(
            q_lambda, mu_sigma_inv, mu_beta0, V_beta0, Ds, Y, F, FF, Lambda_inv, size_deltac, Pc, C, N, K)

        S_bar_sigma = calc_S_bar_sigma(exp_mu_deltac, cov_deltac, Y, F, FF, Z_width, Pc, C, N, K)
        mu_sigma_inv = [T * np.linalg.inv(S_bar_sigma[c]) for c in range(C)]  
        elbo = calc_ELBO(V_beta0, exp_logdet_V_deltac, S_bar_sigma, mu_log_lambda, mu_lambda_inv_D, mu_log_q_lambda, C, N, K, T)
        ELBO.append(elbo)
        print(elbo)
    
    params = {
        'mu_beta0': mu_beta0,
        'V_beta0': V_beta0,
        'q_lambda': q_lambda,
        'S_bar_sigma': S_bar_sigma,
        'cov_deltac': cov_deltac
    }
    
    return params, ELBO, ess_list, log_lams_history