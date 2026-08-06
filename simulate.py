import numpy as np


def _sample_var_y(var_y_real, C, rng):
    """Draw (C, N) variances for simulated countries, matched in scale/spread
    to the real data's per-country variable variances (var_y_real: (C_real, N)).

    Parameters
    ----------
    var_y_real : numpy.ndarray of shape (C_real, N)
        Per-country, per-variable residual variances estimated from the real data.
    C : int
        Number of simulated countries to draw variances for.
    rng : numpy.random.Generator
        Random number generator used to draw the simulated variances.

    Returns
    -------
    numpy.ndarray of shape (C, N)
        Simulated per-country, per-variable variances, drawn log-normally with
        mean/std matched to `var_y_real`.
    """
    log_var = np.log(var_y_real)
    mean_log, std_log = log_var.mean(axis=0), log_var.std(axis=0)
    return np.exp(rng.normal(mean_log, std_log, size=(C, len(mean_log))))

def ar_resid_var(x, L):
    """Fit an AR(L) model with constant to a univariate series and return the
    residual variance.

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


def _build_lambda_c(var_y, target_var_w, N, N_w, L, L_w, K):
    """Build the per-country Minnesota-prior diagonal precision-scaling matrices
    Lambda_c from simulated variances.

    Parameters
    ----------
    var_y : numpy.ndarray of shape (C, N)
        Variances per simulated country, one per endogenous variable.
    target_var_w : numpy.ndarray of shape (N_w,)
        Variance of the (simulated) W series, one per exogenous variable.
    N : int
        Number of endogenous variables.
    N_w : int
        Number of exogenous W variables.
    L : int
        Number of endogenous lags.
    L_w : sequence of int
        Lags of W included as regressors.
    K : int
        Total number of regressors per equation (N*L + N_w*len(L_w)).

    Returns
    -------
    numpy.ndarray of shape (C, N*K, N*K)
        Stack of diagonal Lambda_c matrices, one per simulated country.
    """
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
    """Fit a demeaned AR(1) model to a univariate series by least squares.

    Parameters
    ----------
    x : array_like of shape (T,)
        Univariate time series.

    Returns
    -------
    phi : float
        Estimated AR(1) coefficient.
    sigma : float
        Standard deviation of the AR(1) residuals.
    mu : float
        Sample mean of `x`, used as the series' level.
    """
    x = np.asarray(x)
    mu = x.mean()
    x_c = x - mu
    x_t, x_lag = x_c[1:], x_c[:-1]
    phi = np.sum(x_lag * x_t) / np.sum(x_lag**2)
    resid = x_t - phi * x_lag
    sigma = np.std(resid)
    return phi, sigma, mu


def _simulate_ar1(phi, sigma, T_total, rng, mu=0.0, x0=0.0):
    """Simulate a univariate AR(1) process.

    Parameters
    ----------
    phi : float
        AR(1) coefficient.
    sigma : float
        Standard deviation of the innovation noise.
    T_total : int
        Number of time steps to simulate.
    rng : numpy.random.Generator
        Random number generator used to draw the innovations.
    mu : float, optional
        Level added to the (zero-mean) simulated series. Default is 0.0.
    x0 : float, optional
        Initial (demeaned) value of the series at t=0. Default is 0.0.

    Returns
    -------
    numpy.ndarray of shape (T_total,)
        Simulated AR(1) series, including the added level `mu`.
    """
    x = np.zeros(T_total)
    x[0] = x0
    eps = rng.normal(0, sigma, size=T_total)
    for t in range(1, T_total):
        x[t] = phi * x[t-1] + eps[t]
    return x + mu


def _simulate_exog(real_series, T_total, rng):
    """Simulate exogenous columns as independent AR(1) processes fit to the
    corresponding real series.

    Parameters
    ----------
    real_series : numpy.ndarray of shape (T_real, n_cols)
        Real exogenous series, one column per variable.
    T_total : int
        Number of time steps to simulate.
    rng : numpy.random.Generator
        Random number generator used to draw the simulated series.

    Returns
    -------
    numpy.ndarray of shape (T_total, n_cols)
        Simulated exogenous series, one column per variable, each an AR(1)
        process fit to the matching column of `real_series`.
    """
    # real_series: (T_real, n_cols)
    n_cols = real_series.shape[1]
    sim = np.zeros((T_total, n_cols))
    for j in range(n_cols):
        phi, sigma, mu = _fit_ar1(real_series[:, j])
        sim[:, j] = _simulate_ar1(phi, sigma, T_total, rng, mu=mu)
    return sim


def simulate_data(Y_real, W_real, Z1_real, Z2_real, results_gibbs,
                   C, T, N, N_w, L, L_w, L_z1, L_z2, K, Z_width,
                   burn=50, seed=None):
    """Simulate a (C, T, N) panel calibrated to the real data and a real Gibbs run.

    Exogenous series (W, Z1, Z2) are re-simulated as AR(1) processes fit to the
    real series, rather than reused verbatim, so datasets with C/T different from
    the real panel still get exogenous paths of the right length. beta_0 and lambda
    are taken directly from the means of results_gibbs; per-country betas are then drawn from
    the Minnesota prior implied by beta_0/lambda/Lambda and stabilized. Sigma_c
    reuses the real posterior's average correlation structure with resampled
    variances. gamma_c is drawn from the mean and std of the results_gibbs output.

    Parameters
    ----------
    Y_real : numpy.ndarray of shape (C_real, T_real, N)
        Real endogenous panel data.
    W_real : numpy.ndarray of shape (T_real, N_w)
        Real exchangeable exogenous series.
    Z1_real : numpy.ndarray of shape (T_real, ...)
        Real first non-exchangeable exogenous series.
    Z2_real : numpy.ndarray of shape (T_real, ...)
        Real second non-exchangeable exogenous series.
    results_gibbs : dict
        Posterior draws from a real-data Gibbs run; must contain keys
        'beta_0', 'lam', 'Sigma_c', and 'gamma_c'.
    C : int
        Number of countries to simulate.
    T : int
        Number of usable (post-lag) time periods to simulate.
    N : int
        Number of endogenous variables.
    N_w : int
        Number of exogenous W variables.
    L : int
        Number of endogenous lags.
    L_w : sequence of int
        Lags of W included as regressors.
    L_z1 : sequence of int
        Lags of Z1 included as regressors.
    L_z2 : sequence of int
        Lags of Z2 included as regressors.
    K : int
        Total number of regressors per equation (N*L + N_w*len(L_w)).
    Z_width : int
        Total number of non-exchangeable regressors per equation
        (len(L_z1)*Z1 width + len(L_z2)*Z2 width).
    burn : int, optional
        Number of extra initial time periods simulated then discarded to
        reduce initial-condition dependence. Default is 50.
    seed : int or None, optional
        Seed for the random number generator. Default is None.

    Returns
    -------
    true_params : dict
        Dictionary with keys ``beta_c`` (numpy.ndarray, shape (C, N*K)),
        ``gamma_c`` (numpy.ndarray, shape (C, N*Z_width)), ``Sigma_c``
        (numpy.ndarray, shape (C, N, N)), ``beta_0`` (numpy.ndarray, shape
        (N*K,)), ``lam`` (float), and ``Lambda`` (numpy.ndarray, shape
        (C, N*K, N*K)) — the true generating parameters.
    Y : numpy.ndarray of shape (C, T, N)
        Simulated endogenous panel.
    W_sim : numpy.ndarray of shape (T, N_w)
        Simulated exogenous W series.
    Z1_sim : numpy.ndarray of shape (T, ...)
        Simulated exogenous Z1 series.
    Z2_sim : numpy.ndarray of shape (T, ...)
        Simulated exogenous Z2 series.
    """
    rng = np.random.default_rng(seed)
    C_real = Y_real.shape[0]
    T_total = T + L + burn

    # exogenous series, calibrated to real AR(1) dynamics
    W_sim = _simulate_exog(W_real, T_total, rng)
    Z1_sim = _simulate_exog(Z1_real, T_total, rng)
    Z2_sim = _simulate_exog(Z2_real, T_total, rng)

    # Minnesota-prior variances, calibrated to real data scale
    var_y_real = np.array([[ar_resid_var(Y_real[c, :, n], L) for n in range(N)] for c in range(C_real)])
    var_y = _sample_var_y(var_y_real, C, rng)
    target_var_w = np.array([ar_resid_var(W_sim[:, j], L) for j in range(W_sim.shape[1])])
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
    betas = np.zeros((C, N*K))
    for c in range(C):
        beta_vec = beta0_sim + np.sqrt(lambda_sim * np.diag(Lambda_sim[c])) * rng.standard_normal(N*K)
        B_c = beta_vec.reshape(N, K)

        Comp = np.zeros((N*L, N*L))
        Comp[:N, :] = B_c[ :, :N*L]
        Comp[N:, :-N] = np.eye(N*(L-1))
        radius = np.max(np.abs(np.linalg.eigvals(Comp)))
        # shrink if needed to ensure stationarity
        if radius >= 1:
            B_c[ :, :N*L] *= 0.95 / radius
        betas[c] = B_c.flatten()

    # true gamma_c: single combined block over [Z1 lags, Z2 lags], matching data_prep.py's Z
    gamma_c_real = np.mean(results_gibbs["gamma_c"], axis=0)  # (C_real, N*Z_width)
    mean_g, std_g = gamma_c_real.mean(axis=0), gamma_c_real.std(axis=0)
    gamma_c = rng.normal(mean_g, std_g, size=(C, N*Z_width))

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
            Y[c, t] = betas[c].reshape(N, K) @ regressors + gamma_c[c].reshape(N, Z_width) @ exog_z + innovations[t, c]

    Y = Y[:, burn:, :]
    W_sim = W_sim[burn:]
    Z1_sim = Z1_sim[burn:]
    Z2_sim = Z2_sim[burn:]

    true_params = {
        "beta_0": beta0_sim, "lam": lambda_sim,
        "beta_c": betas, "gamma_c": gamma_c, "Sigma_c": Sigma_sim, 'Lambda': Lambda_sim
    }
    return true_params, Y, W_sim, Z1_sim, Z2_sim
