import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# ── Define relevant matrices ───────────────────────────────────────────────

def prep_data(Y, W, Z1, Z2, C, N, N_w, N_z, T, K, L, L_w, L_z1, L_z2):
    n_lag_z = len(L_z1)  # == len(L_z2)
    Z_width = N_z * n_lag_z
    
    F = np.zeros((C, T, K+Z_width))
    X = np.zeros((C, T, K))
    Z = np.zeros((T, Z_width))

    for t in range(L, T+L):
        z_lags = np.concatenate(
            [Z1[t-l] for l in L_z1] +
            [Z2[t-l] for l in L_z2]
        )
        Z[t-L, :] = z_lags
    for c in range(C):
        for t in range(L, T+L):
            y_lags = np.concatenate([Y[t-l, c, :] for l in range(1, L+1)])
            w_lags = np.concatenate([W[t-l] for l in L_w])
            lags = np.concatenate([y_lags, w_lags])
            X[c, t-L, :] = lags
            F[c, t-L, :] = np.concatenate([lags, Z[t-L, :]])

    Y = Y[:, L:, :]  # (T, C, N)
    W = W[L:, :]
    ZZ = Z.T @ Z                                      # (N_z, N_z)
    XX = np.array([X[c].T @ X[c] for c in range(C)])  # (C, K, K)
    XZ = np.array([X[c].T @ Z for c in range(C)])     # (C, K, N_z)

    # block sizes
    size_beta0 = N * K
    size_betac = N * K
    size_gammac = N * Z_width
    size_deltac = size_betac + size_gammac
    size_delta = size_beta0 + C * size_deltac

    # starting index of each block in delta
    idx_deltac = [size_beta0 + c * size_deltac for c in range(C)]

    Pc = np.zeros((size_deltac, size_deltac))

    for n in range(N):
        for k in range(K):
            col_major_pos = k * N + n
            row_major_pos = n * K + k
            Pc[row_major_pos, col_major_pos] = 1

    for n in range(N):
        for k in range(Z_width):
            col_major_pos = k * N + n
            row_major_pos = n * Z_width + k
            Pc[N*K + row_major_pos, N*K + col_major_pos] = 1

    Lambda = np.zeros((C, N*K, N*K))
    for c in range(C):
        var_y = np.var(Y[c, :, :], axis=0)  # (N,)
        var_w = np.var
        var_all = np.append(var_y, np.var(w))
        var_index = [n for l in range(L) for n in range(N)] + [N]

        diag = np.array([var_y[n] / var_all[var_index[k]]
                        for n in range(N) for k in range(K)])
        Lambda[c] = np.diag(diag)

    Lambda_inv = np.array([np.diag(1.0 / np.diag(Lambda[c])) for c in range(C)])
    Lambda_inv_sum = np.sum(Lambda_inv, axis=0)
    Lambda_inv_sum_inv = np.diag(1.0 / np.diag(Lambda_inv_sum))

    Big_S = np.zeros((size_delta, size_delta))
    for c in range(C):
        b0 = slice(0, size_beta0)
        bc = slice(idx_deltac[c], idx_deltac[c] + size_betac)
        Big_S[b0, b0] += Lambda_inv[c]
        Big_S[bc, bc] += Lambda_inv[c]
        Big_S[b0, bc] -= Lambda_inv[c]
        Big_S[bc, b0] -= Lambda_inv[c]

    cavi_pack = {'Y': Y, 'F': F, 'XX': XX, 'XZ': XZ, 'ZZ': ZZ,
                 'idx_deltac': idx_deltac, 'size_gammac': size_gammac, 'size_deltac': size_deltac,
                 'Pc': Pc, 'Big_S': Big_S,
                 'Lambda_inv': Lambda_inv, 'Lambda_inv_sum': Lambda_inv_sum}
    gibbs_pack =  {'Y': Y, 'X': X, 'Z': Z,
                 'Lambda_inv': Lambda_inv, 'Lambda_inv_sum_inv': Lambda_inv_sum_inv}
    
    return cavi_pack, gibbs_pack