# Import libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import arviz as az
from scipy.stats import invgamma, invwishart

# ── Gibbs sampling functions ─────────────────────────────────────────────────
# Each function draws one sample from the conditional posterior of a single
# parameter block, given the current values of all other parameters.

def beta_0_sample(lam, beta_c, Lambda_inv, Lambda_inv_sum_inv, C):
    # Posterior: beta_0 | rest ~ N(mu, V)
    # V = lambda * (sum Lambda_inv_c)^{-1}
    # mu = V * (1/lambda) * sum_c Lambda_inv_c beta_c
    V_beta_0 = lam * Lambda_inv_sum_inv
    mu_beta_0 = V_beta_0 @ ((1/lam) * sum(Lambda_inv[c] @ beta_c[c] for c in range(C)))
    sample = np.random.multivariate_normal(mu_beta_0, V_beta_0)
    return sample

def lambda_sample(beta_c, beta_0, Lambda_inv, C, N, K):
    # Posterior: lambda | rest ~ InvGamma(s_bar/2, v_bar/2)
    # v_bar = sum_c (beta_c - beta_0)' Lambda_inv_c (beta_c - beta_0)
    s_bar = C*N*K -1
    v_bar = sum((beta_c[c]-beta_0).T @ Lambda_inv[c] @ (beta_c[c]-beta_0) for c in range(C))
    sample = invgamma.rvs(s_bar/2, scale=v_bar/2)
    return sample

def beta_c_sample(lam, beta_0, Sigma_inv, gamma, y_c, X_c, Z, Lambda_inv_c, N):
    # Posterior: beta_c | rest ~ N(mu, V)
    # Precision = (1/lambda) Lambda_inv_c + Sigma_inv ⊗ X'X
    # r_c removes the gamma contribution from y before computing mu
    V_beta_c = np.linalg.inv((1/lam)*Lambda_inv_c + np.kron(Sigma_inv, (X_c.T @ X_c)))
    r_c = y_c - np.kron(np.eye(N), Z) @ gamma
    mu_beta_c = V_beta_c@((1/lam)*Lambda_inv_c @ beta_0 + np.kron(Sigma_inv, X_c.T) @ r_c)
    sample = np.random.multivariate_normal(mu_beta_c, V_beta_c)
    return sample

def gamma_c_sample(Sigma_inv, beta, y_c, X_c, Z, N):
    # Posterior: gamma_c | rest ~ N(mu, V)
    # Precision = Sigma_inv ⊗ Z'Z
    # r_c removes the beta contribution from y before computing mu
    V_gamma_c = np.linalg.inv(np.kron(Sigma_inv, (Z.T @ Z)))
    r_c = y_c - np.kron(np.eye(N), X_c) @ beta
    mu_gamma_c = V_gamma_c@(np.kron(Sigma_inv, Z.T)) @ r_c
    sample = np.random.multivariate_normal(mu_gamma_c, V_gamma_c)
    return sample

def Sigma_c_sample(Beta_c, gamma_c, Y_c, X_c, Z, N, T):
    # Posterior: Sigma_c | rest ~ InvWishart(T, S_bar)
    # S_bar is the residual sum of squares after removing fitted values
    resid = Y_c - X_c @ Beta_c - Z @ gamma_c
    S_bar = resid.T @ resid
    sample = invwishart.rvs(T, S_bar)
    return sample

# ── Diagnostics ───────────────────────────────────────────────────────────────

def _compute_diagnostics(all_chains_data, n_burnin):
    # Flatten all parameters from all chains into a single (n_chains, n_post, D)
    # array, where D is the total number of scalar components across all parameters.
    # arviz then computes bulk ESS and rank-normalised R-hat for each component.
    chain_vecs = []
    for chain_data in all_chains_data:
        n_post = len(chain_data['lam']) - n_burnin

        # Discard burn-in and flatten each parameter to (n_post, d_param)
        lam     = np.array(chain_data['lam']    )[n_burnin:, np.newaxis]          # (n_post, 1)
        beta_0  = np.array(chain_data['beta_0'] )[n_burnin:]                       # (n_post, N*K)
        beta_c  = np.array(chain_data['beta_c'] )[n_burnin:].reshape(n_post, -1)  # (n_post, C*N*K)
        gamma_c = np.array(chain_data['gamma_c'])[n_burnin:].reshape(n_post, -1)  # (n_post, C*N)
        Sigma_c = np.array(chain_data['Sigma_c'])[n_burnin:].reshape(n_post, -1)  # (n_post, C*N*N)

        chain_vecs.append(np.concatenate([lam, beta_0, beta_c, gamma_c, Sigma_c], axis=1))

    all_scalars = np.stack(chain_vecs, axis=0)  # (n_chains, n_post, D)

    # arviz expects (n_chains, n_draws, *param_shape); passing D as the param shape
    # gives one ESS and one R-hat value per scalar component
    data  = {'params': all_scalars}
    ess   = az.ess(data)['params'].values
    r_hat = az.rhat(data)['params'].values

    return ess, r_hat

# ── Gibbs sampler ────────────────────────────────────────────────────────────

def run_gibbs(gibbs_pack, C, N, K, Z_width, T, n_chains=4, n_steps=10000, n_burnin=2000):
    Y, X, Z, Lambda_inv, Lambda_inv_sum_inv = gibbs_pack.values()

    # Vectorise Y column-major so y[c] = vec(Y_c), matching the Kronecker convention
    y = np.zeros((C, T*N))
    for c in range(C):
        y[c] = Y[c, :, :].flatten(order='F')

    all_chains_data = []

    for chain_idx in range(n_chains):
        # Each chain starts at a different scale to ensure overdispersed initialisation,
        # which is required for R-hat to detect non-convergence
        scale       = chain_idx + 1
        beta_0      = np.random.randn(N*K) * scale
        beta_c      = [np.random.randn(N*K) * scale for _ in range(C)]
        gamma_c     = [np.random.randn(N*Z_width) * scale for _ in range(C)]
        Sigma_c     = [(lambda A: A @ A.T + np.eye(N))(np.random.randn(N, N)) for _ in range(C)]
        Sigma_c_inv = [np.linalg.inv(S) for S in Sigma_c]
        lam         = np.random.exponential(scale)

        samples = {'beta_0': [], 'lam': [], 'beta_c': [], 'gamma_c': [], 'Sigma_c': []}

        # Sweep through all conditional posteriors in turn
        for n in range(n_steps):
            beta_0 = beta_0_sample(lam, beta_c, Lambda_inv, Lambda_inv_sum_inv, C)
            samples['beta_0'].append(beta_0.copy())

            lam = lambda_sample(beta_c, beta_0, Lambda_inv, C, N, K)
            samples['lam'].append(lam)

            beta_c = [beta_c_sample(lam, beta_0, Sigma_c_inv[c], gamma_c[c], y[c], X[c,:,:], Z, Lambda_inv[c], N) for c in range(C)]
            samples['beta_c'].append(beta_c.copy())

            gamma_c = [gamma_c_sample(Sigma_c_inv[c], beta_c[c], y[c], X[c,:,:], Z, N) for c in range(C)]
            samples['gamma_c'].append(gamma_c.copy())

            Sigma_c = [Sigma_c_sample(beta_c[c].reshape(K, N, order='F'), gamma_c[c].reshape(Z_width, N, order='F'), Y[c,:,:], X[c,:,:], Z, N, T) for c in range(C)]
            Sigma_c_inv = [np.linalg.inv(Sigma_c[c]) for c in range(C)]
            samples['Sigma_c'].append(Sigma_c.copy())

        all_chains_data.append(samples)

    ess, r_hat = _compute_diagnostics(all_chains_data, n_burnin)

    # Return post-burnin samples pooled across all chains alongside the diagnostics
    post_burnin_samples = {k: np.concatenate([chain[k][n_burnin:] for chain in all_chains_data], axis=0) 
                          for k in all_chains_data[0].keys()}
    post_burnin_samples["delta_c"] = [
        [np.concatenate([post_burnin_samples["beta_c"][t][c], post_burnin_samples["gamma_c"][t][c]])
        for c in range(C)]
        for t in range(len(post_burnin_samples["beta_c"]))
    ]
    return post_burnin_samples, ess, r_hat
