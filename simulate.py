import numpy as np


def _sample_var_y(var_y_real, C, rng):
    """Draw (C, N) variances for simulated countries, matched in scale/spread
    to the real data's per-country variable variances (var_y_real: (C_real, N))."""
    log_var = np.log(var_y_real)
    mean_log, std_log = log_var.mean(axis=0), log_var.std(axis=0)
    return np.exp(rng.normal(mean_log, std_log, size=(C, len(mean_log))))


def _build_lambda_c(var_y, target_var_w, N, N_w, L, L_w, K):
    """var_y: (C, N) -- variances per simulated country.
    target_var_w: (N_w,) -- variance of the (simulated) W series."""
    C = var_y.shape[0]
    Lambda = np.zeros((C, N*K, N*K))
    var_index = ([n for l in range(L) for n in range(N)] +
                 [N + j for l in range(len(L_w)) for j in range(N_w)])
    for c in range(C):
        var_all = np.append(var_y[c], target_var_w)
        diag = np.array([var_y[c][n] / var_all[var_index[k]]
                          for n in range(N) for k in range(K)])
        Lambda[c] = np.diag(diag)
    return Lambda


def _fit_ar1(x):
    x = np.asarray(x)
    x_t, x_lag = x[1:], x[:-1]
    phi = np.sum(x_lag * x_t) / np.sum(x_lag**2)
    resid = x_t - phi * x_lag
    sigma = np.std(resid)
    return phi, sigma


def _simulate_ar1(phi, sigma, T_total, rng, x0=0.0):
    x = np.zeros(T_total)
    x[0] = x0
    eps = rng.normal(0, sigma, size=T_total)
    for t in range(1, T_total):
        x[t] = phi * x[t-1] + eps[t]
    return x


def _simulate_exog(real_series, T_total, rng):
    # real_series: (T_real, n_cols)
    n_cols = real_series.shape[1]
    sim = np.zeros((T_total, n_cols))
    for j in range(n_cols):
        phi, sigma = _fit_ar1(real_series[:, j])
        sim[:, j] = _simulate_ar1(phi, sigma, T_total, rng)
    return sim


def simulate_data(Y_real, W_real, Z1_real, Z2_real, results_gibbs,
                   C, T, N, N_w, L, L_w, L_z1, L_z2, K, Z_width,
                   burn=50, seed=None):
    """Simulate a (C, T, N) panel calibrated to the real data and a real Gibbs run.

    Exogenous series (W, Z1, Z2) are re-simulated as AR(1) processes fit to the
    real series, rather than reused verbatim, so datasets with C/T different from
    the real panel still get exogenous paths of the right length. beta_0 and lambda
    are taken directly from results_gibbs; per-country betas are then drawn from
    the Minnesota prior implied by beta_0/lambda/Lambda and stabilized. Sigma_c
    reuses the real posterior's average correlation structure with resampled
    variances. gamma_c is drawn fresh (no equivalent real target).

    Returns
    -------
    true_params : dict with beta_c (C,N,K), gamma_c (C,N,Z_width), Sigma_c (C,N,N),
        beta_0 (N*K,), lam (float)
    Y : (C, T, N) array
    """
    rng = np.random.default_rng(seed)
    C_real = Y_real.shape[0]
    T_total = T + L + burn

    # exogenous series, calibrated to real AR(1) dynamics
    W_sim = _simulate_exog(W_real, T_total, rng)
    Z1_sim = _simulate_exog(Z1_real, T_total, rng)
    Z2_sim = _simulate_exog(Z2_real, T_total, rng)

    # Minnesota-prior variances, calibrated to real data scale
    var_y_real = np.array([np.var(Y_real[c], axis=0) for c in range(C_real)])
    var_y = _sample_var_y(var_y_real, C, rng)
    target_var_w = np.var(W_sim, axis=0)
    Lambda_sim = _build_lambda_c(var_y, target_var_w, N, N_w, L, L_w, K)

    # true beta_0 / lambda, taken directly from the real posterior
    beta0_sim = np.mean(results_gibbs['beta_0'], axis=0)
    lambda_sim = np.mean(results_gibbs['lam'])

    # true Sigma_c: real posterior's average correlation structure, resampled variances
    Sigma_c_real = np.array(results_gibbs["Sigma_c"])[:, :C_real].mean(axis=0)  # (C_real, N, N)
    corr_real = np.array([
        Sigma_c_real[c] / np.outer(np.sqrt(np.diag(Sigma_c_real[c])), np.sqrt(np.diag(Sigma_c_real[c])))
        for c in range(C_real)
    ])
    target_corr = corr_real.mean(axis=0)
    sigma_diag_real = np.array([np.diag(Sigma_c_real[c]) for c in range(C_real)])
    sigma_diag = _sample_var_y(sigma_diag_real, C, rng)
    Sigma_sim = np.zeros((C, N, N))
    for c in range(C):
        D = np.diag(np.sqrt(sigma_diag[c]))
        Sigma_sim[c] = D @ target_corr @ D

    # true betas: Minnesota-shrunk around beta_0, stabilized per country
    betas = np.zeros((C, N, K))
    for c in range(C):
        beta_vec = beta0_sim + np.sqrt(lambda_sim * np.diag(Lambda_sim[c])) * rng.standard_normal(N*K)
        betas[c] = beta_vec.reshape(N, K)

        Comp = np.zeros((N*L, N*L))
        Comp[:N, :] = betas[c, :, :N*L]
        Comp[N:, :-N] = np.eye(N*(L-1))
        radius = np.max(np.abs(np.linalg.eigvals(Comp)))
        if radius >= 1:
            betas[c, :, :N*L] *= 0.95 / radius

    # true gamma_c: single combined block over [Z1 lags, Z2 lags], matching data_prep.py's Z
    gamma_c = rng.normal(0, 0.2, size=(C, N, Z_width))

    # innovations
    innovations = np.zeros((T_total, C, N))
    for c in range(C):
        innovations[:, c, :] = rng.multivariate_normal(np.zeros(N), Sigma_sim[c], size=T_total)

    # simulate Y recursively
    Y = np.zeros((C, T_total, N))
    for t in range(L, T_total):
        exog_w = np.concatenate([W_sim[t-l] for l in L_w])
        exog_z = np.concatenate([Z1_sim[t-l] for l in L_z1] + [Z2_sim[t-l] for l in L_z2])
        for c in range(C):
            lags_y = np.concatenate([Y[c, t-l, :] for l in range(1, L+1)])
            regressors = np.concatenate([lags_y, exog_w])
            Y[c, t] = betas[c] @ regressors + gamma_c[c] @ exog_z + innovations[t, c]

    Y = Y[:, burn:, :]

    true_params = {
        "beta_0": beta0_sim, "lam": lambda_sim,
        "beta_c": betas, "gamma_c": gamma_c, "Sigma_c": Sigma_sim, 'Lambda': Lambda_sim
    }
    return true_params, Y
