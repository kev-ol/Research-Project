# ── Define relevant matrices ───────────────────────────────────────────────

def prep_data(Y, w, z, C, N, T, K, L):
    F = np.zeros((C, T, K+1))
    X = np.zeros((C, T, K))
    for c in range(C):
        for t in range(L, T+L):
            lags = np.concatenate([Y[t-1, c, :], Y[t-2, c, :], [w[t-1]]])
            X[c, t-L, :] = lags
            F[c, t-L, :] = np.concatenate([lags, [z[t-1]]])

    Y = Y[L:, :, :]  # (T, C, N)

    y = np.zeros((C, T*N))
    for c in range(C):
        y[c] = Y[:, c, :].flatten(order='F')

    z = z[L:]
    w = w[L:]
    z = z.reshape(-1, 1)  # (T, 1)
    ZZ = z.T @ z
    XX = np.array([X[c].T @ X[c] for c in range(C)])  # (C, NK, NK)
    XZ = np.array([X[c].T @ z for c in range(C)])      # (C, NK, N_z)


    # block sizes
    size_beta0 = N * K
    size_betac = N * K
    size_gammac = N
    size_deltac = size_betac + size_gammac
    size_delta = size_beta0 + C * size_deltac

    # starting index of each block in delta
    idx_beta0 = 0
    idx_deltac = [size_beta0 + c * size_deltac for c in range(C)]  # start of delta_c
    idx_betac  = [idx_deltac[c] for c in range(C)]                 # start of beta_c (same)
    # selection matrices — S @ delta extracts the relevant block
    def selection_matrix(size_delta, start, block_size):
        S = np.zeros((block_size, size_delta))
        S[:, start:start+block_size] = np.eye(block_size)
        return S

    S0      = selection_matrix(size_delta, idx_beta0, size_beta0)
    S_beta  = [selection_matrix(size_delta, idx_betac[c],  size_betac)  for c in range(C)]
    S_delta = [selection_matrix(size_delta, idx_deltac[c], size_deltac) for c in range(C)]

    Pc = np.zeros((size_deltac, size_deltac))

    for n in range(N):
        for k in range(K):
            col_major_pos = k * N + n
            row_major_pos = n * K + k
            Pc[row_major_pos, col_major_pos] = 1

    for i in range(N):
        Pc[N*K + i, N*K + i] = 1

    Lambda = np.zeros((C, N*K, N*K))
    for c in range(C):
        var_y = np.var(Y[:, c, :], axis=0)  # (N,)
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
        diff = S_beta[c] - S0
        Big_S += diff.T @ Lambda_inv[c] @ diff

    cavi_pack = {'Y': Y, 'F': F, 'XX': X, 'XZ': XZ, 'ZZ': ZZ, 
                 'S_delta': S_delta, 'Pc': Pc, 'Big_S': Big_S, 
                 'Lambda_inv': Lambda_inv, 'Lambda_inv_sum': Lambda_inv_sum}
    gibbs_pack =  {'Y': Y, 'X': X, 'z': z,
                 'Lambda_inv': Lambda_inv, 'Lambda_inv_sum_inv': Lambda_inv_sum_inv}
    
    return cavi_pack, gibbs_pack