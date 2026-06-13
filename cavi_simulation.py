# Import libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# ── Initialise variational parameters ────────────────────────────────────────
def initialise_params(T, N):
    params = {
        'mu_lambda_inv': 1e4,
        'mu_sigma_inv': [T * np.eye(N) for c in range(C)]
    }
    return params

# ── CAVI update functions ─────────────────────────────────────────────────────

def calc_V_delta(mu_lambda_inv, mu_sigma_inv, Lambda_inv, Lambda_inv_sum, XX, XZ, ZZ, ):
    total = mu_lambda_inv * Lambda_inv_sum
    for c in range(C):
        top_left = mu_lambda_inv * Lambda_inv[c] + np.kron(mu_sigma_inv[c], XX[c])
        top_right = np.kron(mu_sigma_inv[c], XZ[c])
        bottom_left = top_right.T
        bottom_right = np.kron(mu_sigma_inv[c], ZZ)

        H_c_inv = np.block([[top_left, top_right],[bottom_left, bottom_right]])
        H_c = np.linalg.inv(H_c_inv)

        total -= mu_lambda_inv**2 * (Lambda_inv[c] @ H_c[:N*K, :N*K] @ Lambda_inv[c])
    return np.linalg.inv(total)

def calc_mu_delta(V_delta, mu_sigma_inv):
    sum = 0
    for c in range(C):
        sum += S_delta[c].T @ Pc.T @ np.kron(mu_sigma_inv[c], F[c].T) @ y[c]
    return V_delta @ sum

def calc_S_bar_sigma(mu_delta, V_delta):
    S_bar_sigma = [np.eye(N)] * C
    for c in range(C):
        mu_deltac = S_delta[c] @ mu_delta
        vec_Gc = Pc @ mu_deltac
        mu_Gc = vec_Gc.reshape(K+1, N, order='F')

        M_c = Pc @ S_delta[c]
        FtF = F[c].T @ F[c]
        Omega_Gc = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                M_ci = M_c[i*(K+1):(i+1)*(K+1), :]
                M_cj = M_c[j*(K+1):(j+1)*(K+1), :]
                Omega_Gc[i, j] = np.trace(FtF @ M_ci @ V_delta @ M_cj.T)

        S_bar_sigma[c] = (Y[:, c, :] - F[c] @ mu_Gc).T @ (Y[:, c, :] - F[c] @ mu_Gc) + Omega_Gc
    return S_bar_sigma

def calc_ELBO(V_delta, s_bar, v_bar, S_bar_sigma):
    _, logdet_V = np.linalg.slogdet(V_delta)
    elbo = logdet_V - s_bar * np.log(v_bar) / 2
    for c in range(C):
        _, logdet_S = np.linalg.slogdet(S_bar_sigma[c])
        elbo -= T * logdet_S
    return elbo

# ── CAVI loop ─────────────────────────────────────────────────────────────────

epsilon = 1e-4
ELBO = []
s_bar = C*N*K -1
while len(ELBO) < 10 or ELBO[-1] - ELBO[-2] > epsilon:
    V_delta = calc_V_delta(mu_lambda_inv, mu_sigma_inv)
    mu_delta = calc_mu_delta(V_delta, mu_sigma_inv)
    v_bar = mu_delta.T @ Big_S @ mu_delta + np.trace(Big_S @ V_delta)
    mu_lambda_inv = s_bar/v_bar
    S_bar_sigma = calc_S_bar_sigma(mu_delta, V_delta)
    mu_sigma_inv = [T * np.linalg.inv(S_bar_sigma[c]) for c in range(C)]
    ELBO.append(calc_ELBO(V_delta, s_bar, v_bar, S_bar_sigma))

