import numpy as np

def calc_V_beta0():

def calc_mu_beta0():

def calc_q_lambda():

def calc_V_deltac():

def calc_mu_deltac():

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