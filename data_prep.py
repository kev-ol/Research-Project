import numpy as np

"""Preprocessing Data"""

def prep_data(Y, W, Z1, Z2, C, N, N_w, T, K, Z_width, L, L_w, L_z1, L_z2, Lambda = None):
    """Build lagged design matrices, Minnesota-prior scale matrices, and the
    per-model data packs consumed by `mfvi.run_mfvi`, `ssvi_i.run_ssvi_i`,
    `ssvi_c.run_ssvi_c` and `gibbs.run_gibbs`.

    Parameters
    ----------
    Y : numpy.ndarray of shape (C, T+L, N)
        Raw endogenous panel data, including the L extra leading periods
        needed to form lags.
    W : numpy.ndarray of shape (T+L, N_w)
        Raw exchangeable exogenous series, including the leading periods
        needed to form lags.
    Z1 : numpy.ndarray of shape (T+L, ...)
        Raw first non-exchangeable exogenous series, including the leading
        periods needed to form the lags in `L_z1`.
    Z2 : numpy.ndarray of shape (T+L, ...)
        Raw second non-exchangeable exogenous series, including the leading
        periods needed to form the lags in `L_z2`.
    C : int
        Number of countries.
    N : int
        Number of endogenous variables.
    N_w : int
        Number of exogenous W variables.
    T : int
        Number of usable (post-lag) time periods.
    K : int
        Total number of regressors per equation.
    Z_width : int
        Total number of non-exchangeable regressors per equation.
    L : int
        Number of endogenous lags.
    L_w : sequence of int
        Lags of W included as regressors.
    L_z1 : sequence of int
        Lags of Z1 included as regressors.
    L_z2 : sequence of int
        Lags of Z2 included as regressors.
    Lambda : numpy.ndarray of shape (C, N*K, N*K) or None, optional
        Pre-specified Minnesota-prior diagonal scale matrices. If None
        (default), Lambda is built internally from AR(L) residual variances
        of the real series.

    Returns
    -------
    mfvi_pack : dict
        Data pack for `mfvi.run_mfvi`, with keys 'Y' (numpy.ndarray, shape
        (C, T, N)), 'F' (numpy.ndarray, shape (C, T, K+Z_width)), 'FF'
        (numpy.ndarray, shape (C, K+Z_width, K+Z_width)), 'XX'
        (numpy.ndarray, shape (C, K, K)), 'XZ' (numpy.ndarray, shape
        (C, K, Z_width)), 'ZZ' (numpy.ndarray, shape (Z_width, Z_width)),
        'idx_deltac' (list of length C of int), 'size_gammac' (int),
        'size_deltac' (int), 'Pc' (numpy.ndarray, shape
        (size_deltac, size_deltac)), 'Big_S' (numpy.ndarray, shape
        (size_delta, size_delta)), 'Lambda_inv' (numpy.ndarray, shape
        (C, N*K, N*K)), and 'Lambda_inv_sum' (numpy.ndarray, shape
        (N*K, N*K)).
    ssvi_i_pack : dict
        Data pack for `ssvi_i.run_ssvi_i` and `ssvi_c.run_ssvi_c`, with keys
        'Y' (numpy.ndarray, shape (C, T, N)), 'F' (numpy.ndarray, shape
        (C, T, K+Z_width)), 'FF' (numpy.ndarray, shape
        (C, K+Z_width, K+Z_width)), 'idx_deltac' (list of length C of int),
        'size_deltac' (int), 'Pc' (numpy.ndarray, shape
        (size_deltac, size_deltac)), 'Lambda_inv' (numpy.ndarray, shape
        (C, N*K, N*K)), and 'Lambda_inv_sum' (numpy.ndarray, shape
        (N*K, N*K)).
    gibbs_pack : dict
        Data pack for `gibbs.run_gibbs`, with keys 'Y' (numpy.ndarray, shape
        (C, T, N)), 'X' (numpy.ndarray, shape (C, T, K)), 'XX'
        (numpy.ndarray, shape (C, K, K)), 'Z' (numpy.ndarray, shape
        (T, Z_width)), 'ZZ' (numpy.ndarray, shape (Z_width, Z_width)),
        'Lambda_inv' (numpy.ndarray, shape (C, N*K, N*K)), and
        'Lambda_inv_sum' (numpy.ndarray, shape (N*K, N*K)).
    """
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
            """Fit an AR(L) model with constant to a univariate series and
            return the residual variance.

            Parameters
            ----------
            x : numpy.ndarray of shape (T,)
                Univariate time series.
            L : int
                Number of autoregressive lags.

            Returns
            -------
            float
                Variance of the residuals from the fitted AR(L) model.
            """
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
    mfvi_pack = {'Y': Y, 'F': F, 'FF': FF, 'idx_deltac': idx_deltac,
                 'size_deltac': size_deltac, 'Pc': Pc, 'Big_S': Big_S}
    ssvi_i_pack = {'Y': Y, 'F': F, 'FF': FF, 'idx_deltac': idx_deltac, 'size_deltac': size_deltac,
                 'Pc': Pc, 'Lambda_inv': Lambda_inv, 'Lambda_inv_sum': Lambda_inv_sum}
    gibbs_pack =  {'Y': Y, 'X': X, 'XX': XX, 'Z': Z, 'ZZ': ZZ,
                 'Lambda_inv': Lambda_inv, 'Lambda_inv_sum': Lambda_inv_sum}
    
    return mfvi_pack, ssvi_i_pack, gibbs_pack
