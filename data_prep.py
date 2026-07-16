import numpy as np

"""Preprocessing Data"""

def prep_data(Y, W, Z1, Z2, C, N, N_w, T, K, Z_width, L, L_w, L_z1, L_z2, Lambda = None):
    
    F = np.zeros((C, T, K+Z_width))
    X = np.zeros((C, T, K))
    Z = np.zeros((T, Z_width))
    
    # concatenate non-exchangeable prior data lags
    for t in range(L, T+L):
        z_lags = np.concatenate(
            [Z1[t-l] for l in L_z1] +
            [Z2[t-l] for l in L_z2]
        )
        Z[t-L, :] = z_lags
    
    # concatenate exchangeable prior data lags
    for c in range(C):
        for t in range(L, T+L):
            y_lags = np.concatenate([Y[c, t-l, :] for l in range(1, L+1)])
            w_lags = np.concatenate([W[t-l] for l in L_w])
            lags = np.concatenate([y_lags, w_lags])
            X[c, t-L, :] = lags
            F[c, t-L, :] = np.concatenate([lags, Z[t-L, :]])

    if Lambda is None:
        # make Lambda for Minnesota prior
        def ar_resid_var(x, L):
            # x: (T,) univariate series; returns residual variance of AR(L) with constant
            T = len(x)
            Y_ = x[L:]
            X_ = np.column_stack([x[L-l:T-l] for l in range(1, L+1)] + [np.ones(T-L)])
            coef, _, _, _ = np.linalg.lstsq(X_, Y_, rcond=None)
            resid = Y_ - X_ @ coef
            return np.var(resid)

        Lambda = np.zeros((C, N*K, N*K))
        for c in range(C):
            var_y = np.array([ar_resid_var(Y[c, :, n], L) for n in range(N)])
            var_w = np.array([ar_resid_var(W[:, j], L) for j in range(W.shape[1])])
            var_all = np.append(var_y, var_w)
            var_index = ([n for l in range(L) for n in range(N)] +
                    [N + j for l in range(len(L_w)) for j in range(N_w)])

            diag = np.array([var_y[n] / var_all[var_index[k]]
                            for n in range(N) for k in range(K)])
            Lambda[c] = np.diag(diag)

    # get rid of extra data used for lags
    Y = Y[:, L:, :]
    W = W[L:, :]
    # define relevant matrices for later use
    ZZ = Z.T @ Z                  
    FF = np.array([F[c].T @ F[c] for c in range(C)])
    XX = np.array([X[c].T @ X[c] for c in range(C)])
    XZ = np.array([X[c].T @ Z for c in range(C)])

    # block sizes
    size_beta0 = N * K
    size_betac = N * K
    size_gammac = N * Z_width
    size_deltac = size_betac + size_gammac
    size_delta = size_beta0 + C * size_deltac

    # starting index of each block in delta
    idx_deltac = [size_beta0 + c * size_deltac for c in range(C)]

    # create reordering matrix
    Pc = np.zeros((size_deltac, size_deltac))

    for n in range(N):
        # beta_c's n-th block of K terms -> goes to positions n*(K+Z_width) .. n*(K+Z_width)+K-1
        for k in range(K):
            col_pos = n*K + k                      # position in delta_c's beta_c segment
            row_pos = n*(K + Z_width) + k          # position in the interleaved output
            Pc[row_pos, col_pos] = 1

        # gamma_c's n-th block of Z_width terms -> goes to positions n*(K+Z_width)+K .. n*(K+Z_width)+K+Z_width-1
        for z in range(Z_width):
            col_pos = N*K + n*Z_width + z          # position in delta_c's gamma_c segment (offset by N*K)
            row_pos = n*(K + Z_width) + K + z      # position in the interleaved output
            Pc[row_pos, col_pos] = 1

    # perform inverses now
    Lambda_inv = np.array([np.diag(1.0 / np.diag(Lambda[c])) for c in range(C)])
    Lambda_inv_sum = np.sum(Lambda_inv, axis=0)
    Lambda_inv_sum_inv = np.diag(1.0 / np.diag(Lambda_inv_sum))

    # define Big_S term to save later calculations
    Big_S = np.zeros((size_delta, size_delta))
    for c in range(C):
        b0 = slice(0, size_beta0)
        bc = slice(idx_deltac[c], idx_deltac[c] + size_betac)
        Big_S[b0, b0] += Lambda_inv[c]
        Big_S[bc, bc] += Lambda_inv[c]
        Big_S[b0, bc] -= Lambda_inv[c]
        Big_S[bc, b0] -= Lambda_inv[c]

    # export packs of what is relevant for each model
    mfvi_pack = {'Y': Y, 'F': F, 'FF': FF, 'XX': XX, 'XZ': XZ, 'ZZ': ZZ,
                 'idx_deltac': idx_deltac, 'size_gammac': size_gammac, 'size_deltac': size_deltac,
                 'Pc': Pc, 'Big_S': Big_S,
                 'Lambda_inv': Lambda_inv, 'Lambda_inv_sum': Lambda_inv_sum}
    ssvi_i_pack = {'Y': Y, 'F': F, 'FF': FF, 'idx_deltac': idx_deltac, 'size_deltac': size_deltac,
                 'Pc': Pc, 'Lambda_inv': Lambda_inv, 'Lambda_inv_sum': Lambda_inv_sum}
    gibbs_pack =  {'Y': Y, 'X': X, 'XX': XX, 'Z': Z, 'ZZ': ZZ,
                 'Lambda_inv': Lambda_inv, 'Lambda_inv_sum': Lambda_inv_sum}
    gibbs_pack_og =  {'Y': Y, 'X': X, 'Z': Z,
                 'Lambda_inv': Lambda_inv, 'Lambda_inv_sum_inv': Lambda_inv_sum_inv}

    return mfvi_pack, ssvi_i_pack, gibbs_pack