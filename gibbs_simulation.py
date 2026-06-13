# Import libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import invgamma, invwishart

# ── Gibbs sampling functions ─────────────────────────────────────────────────

def beta_0_sample(lam, beta_c):
    V_beta_0 = lam * Lambda_inv_sum_inv
    mu_beta_0 = V_beta_0 @ ((1/lam) * sum(Lambda_inv[c] @ beta_c[c] for c in range(C)))
    sample = np.random.multivariate_normal(mu_beta_0, V_beta_0)
    return sample

def lambda_sample(beta_c, beta_0):
    s_bar = C*N*K -1
    v_bar = sum((beta_c[c]-beta_0).T @ Lambda_inv[c] @ (beta_c[c]-beta_0) for c in range(C))
    sample = invgamma.rvs(s_bar/2, scale=v_bar/2)
    return sample

def beta_c_sample(lam, beta_0, Sigma_inv, X_c, y_c, z, gamma):
    V_beta_c = np.linalg.inv((1/lam)*Lambda_inv[c] + np.kron(Sigma_inv, (X_c.T @ X_c)))
    r_c = y_c - np.kron(np.eye(N), z) @ gamma
    mu_beta_c = V_beta_c@((1/lam)*Lambda_inv[c]@beta_0 + np.kron(Sigma_inv, X_c.T)@r_c)
    sample = np.random.multivariate_normal(mu_beta_c, V_beta_c)
    return sample

def gamma_c_sample(Sigma_inv, z, beta, X_c, y_c):
    V_gamma_c = np.linalg.inv(np.kron(Sigma_inv, (z.T @ z)))
    r_c = y_c - np.kron(np.eye(N), X_c) @ beta
    mu_gamma_c = V_gamma_c@(np.kron(Sigma_inv, z.T))@r_c
    sample = np.random.multivariate_normal(mu_gamma_c, V_gamma_c)
    return sample

def Sigma_c_sample(Y_c, X_c, Beta_c, z, gamma_c):
    resid = Y_c - X_c @ Beta_c - z @ gamma_c.reshape(1, N)
    S_bar = resid.T @ resid
    sample = invwishart.rvs(T, S_bar)
    return sample

# ── Gibbs sampler ────────────────────────────────────────────────────────────

n_steps = 10000

# Initialise parameters

beta_0 = np.zeros(N*K)
beta_c = [np.zeros(N*K) for c in range(C)]
gamma_c = [np.zeros(N) for c in range(C)]
Sigma_c = [np.eye(N) for c in range(C)]
Sigma_c_inv = [np.eye(N) for c in range(C)]
lam = 0.5

samples = {
    'beta_0': [],
    'lam': [],
    'beta_c': [],
    'gamma_c': [],
    'Sigma_c': []
}

for n in range(n_steps):
    beta_0 = beta_0_sample(lam, Lambda, beta_c)
    samples['beta_0'].append(beta_0.copy())

    lam = lambda_sample(beta_c, beta_0, Lambda)
    samples['lam'].append(lam)

    beta_c = [beta_c_sample(lam, beta_0, Lambda[c], Sigma_c_inv[c], X[c,:,:], y[c], z, gamma_c[c]) for c in range(C)]
    samples['beta_c'].append(beta_c.copy())

    gamma_c = [gamma_c_sample(Sigma_c_inv[c], z, beta_c[c], X[c,:,:], y[c]) for c in range(C)]
    samples['gamma_c'].append(gamma_c.copy())

    Sigma_c = [Sigma_c_sample(Y[:,c,:], X[c,:,:], beta_c[c].reshape(K, N, order='F'), z, gamma_c[c]) for c in range(C)]
    Sigma_c_inv = [np.linalg.inv(Sigma_c[c]) for c in range(C)]
    samples['Sigma_c'].append(Sigma_c.copy())

plt.plot(samples['lam'])
