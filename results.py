import numpy as np
from scipy.stats import invgamma, invwishart, gaussian_kde, wasserstein_distance
from scipy.linalg import eigh
from joblib import Parallel, delayed
import pandas as pd
from ssvi_i import calc_V_deltac, calc_mu_deltac
from ssvi_c import calc_V_beta02, calc_mu_beta02, calc_V_deltac2, calc_mu_deltac2


### Posterior sample reconstruction from each method's VI output ###

def sample_from_mfvi(results_mfvi, mfvi_pack, C, N, K, T, n_samples=10000, rng=None):
    """Reconstruct posterior samples from the MFVI variational approximation.

    Parameters
    ----------
    results_mfvi : dict
        MFVI parameters: 'mu_delta', 'V_delta', 'v_bar', 's_bar', 'S_bar_sigma'.
    mfvi_pack : dict
        Data pack: 'idx_deltac', 'size_deltac'.
    n_samples : int, optional
        Number of posterior samples. Default is 10000.
    rng : int, Generator, or None, optional
        Random seed. Default is None.

    Returns
    -------
    dict
        Keys: 'beta_0', 'lam', 'beta_c', 'gamma_c', 'delta_c', 'Sigma_c'.
    """
    rng = np.random.default_rng(rng)
    mu_delta, V_delta, v_bar, s_bar, S_bar_sigma = results_mfvi.values()

    idx_deltac = mfvi_pack["idx_deltac"]
    size_deltac = mfvi_pack["size_deltac"]

    # delta samples (beta_0, beta_c, gamma_c, delta_c all come from this)
    L = np.linalg.cholesky(V_delta)
    deltas = mu_delta + (L @ rng.normal(size=(len(mu_delta), n_samples))).T

    beta_0_samples = deltas[:, :idx_deltac[0]] 

    beta_c_samples = [deltas[:, idx_deltac[c]:idx_deltac[c] + N*K] for c in range(C)]  
    gamma_c_samples = [deltas[:, idx_deltac[c] + N*K:idx_deltac[c] + size_deltac] for c in range(C)] 
    delta_c_samples = [deltas[:, idx_deltac[c]:idx_deltac[c] + size_deltac] for c in range(C)]    

    # lambda samples (cheap, vectorize trivially)
    lam_samples = invgamma.rvs(s_bar/2, scale=v_bar/2, size=n_samples, random_state=rng)

    # Sigma_c samples (vectorized per country via scipy's size argument)
    Sigma_c_samples = [invwishart.rvs(T, S_bar_sigma[c], size=n_samples, random_state=rng) for c in range(C)]  

    return {
        'beta_0': list(beta_0_samples),
        'lam': list(lam_samples),
        'beta_c': [[beta_c_samples[c][n] for c in range(C)] for n in range(n_samples)],
        'gamma_c': [[gamma_c_samples[c][n] for c in range(C)] for n in range(n_samples)],
        'delta_c': [[delta_c_samples[c][n] for c in range(C)] for n in range(n_samples)],
        'Sigma_c': [[Sigma_c_samples[c][n] for c in range(C)] for n in range(n_samples)],
    }


def sample_from_ssvi_i(results_ssvi_i, ssvi_i_pack, C, N, K, T, n_samples=10000, rng=None):
    """Reconstruct posterior samples from the SSVI-I variational approximation.

    Parameters
    ----------
    results_ssvi_i : dict
        SSVI-I parameters: 'mu_beta0', 'V_beta0', 'q_lambda', 'S_bar_sigma',
        'cov_deltac'.
    ssvi_i_pack : dict
        Data pack: 'Y', 'F', 'FF', 'idx_deltac', 'size_deltac', 'Pc',
        'Lambda_inv', 'Lambda_inv_sum'.
    n_samples : int, optional
        Number of posterior samples. Default is 10000.
    rng : int, Generator, or None, optional
        Random seed. Default is None.

    Returns
    -------
    dict
        Keys: 'beta_0', 'lam', 'beta_c', 'gamma_c', 'delta_c', 'Sigma_c'.
    """
    rng = np.random.default_rng(rng)
    mu_beta0, V_beta0, q_lambda_chain, S_bar_sigma, cov_deltac = results_ssvi_i.values()
    mu_sigma_inv = [T * np.linalg.inv(S_bar_sigma[c]) for c in range(C)]
    Y, F, FF, idx_deltac, size_deltac, Pc, Lambda_inv, Lambda_inv_sum = ssvi_i_pack.values()

    # lambda: sample from a KDE fit to the converged ULA chain (log-space, since lambda > 0)
    lam_chain = np.asarray(q_lambda_chain)
    kde_lam = gaussian_kde(np.log(lam_chain))
    lam_samples = np.exp(kde_lam.resample(n_samples, seed=rng).flatten())

    # beta_0: independent draw, paired with lam_samples by index
    L_beta0 = np.linalg.cholesky(V_beta0)
    beta_0_samples = mu_beta0 + (L_beta0 @ rng.normal(size=(len(mu_beta0), n_samples))).T 

    # V_deltac, mu_deltac: both batched over the same (lam, beta0) pairs
    V_deltac = calc_V_deltac(lam_samples, mu_sigma_inv, FF, Lambda_inv, size_deltac, Pc, C, N, K)
    mu_deltac = calc_mu_deltac(lam_samples, beta_0_samples, V_deltac, mu_sigma_inv, Y, F, Lambda_inv, size_deltac, Pc, C, N, K)

    delta_c_samples_arr = np.empty((n_samples, C, size_deltac))
    for c in range(C):
        L_c = np.linalg.cholesky(V_deltac[c])          
        z = rng.normal(size=(n_samples, size_deltac))
        delta_c_samples_arr[:, c, :] = mu_deltac[c] + np.einsum('nij,nj->ni', L_c, z)

    beta_c_samples_arr = delta_c_samples_arr[:, :, :N*K]
    gamma_c_samples_arr = delta_c_samples_arr[:, :, N*K:]

    Sigma_c_samples = np.stack(
        [invwishart.rvs(T, S_bar_sigma[c], size=n_samples, random_state=rng) for c in range(C)], axis=1
    )

    return {
        'beta_0': list(beta_0_samples),
        'lam': list(lam_samples),
        'beta_c': list(beta_c_samples_arr),
        'gamma_c': list(gamma_c_samples_arr),
        'delta_c': list(delta_c_samples_arr),
        'Sigma_c': list(Sigma_c_samples),
    }


def sample_from_ssvi_c(results_ssvi_c, ssvi_i_pack, C, N, K, T, n_samples=10000, rng=None):
    """Reconstruct posterior samples from the SSVI-C variational approximation.

    Parameters
    ----------
    results_ssvi_c : dict
        SSVI-C parameters: 'q_lambda', 'S_bar_sigma', 'cov_deltac'.
    ssvi_c_pack : dict
        Data pack: 'Y', 'F', 'FF', 'idx_deltac', 'size_deltac', 'Pc',
        'Lambda_inv', 'Lambda_inv_sum'.
    n_samples : int, optional
        Number of posterior samples. Default is 10000.
    rng : int, Generator, or None, optional
        Random seed. Default is None.

    Returns
    -------
    dict
        Keys: 'beta_0', 'lam', 'beta_c', 'gamma_c', 'delta_c', 'Sigma_c'.
    """
    rng = np.random.default_rng(rng)
    q_lambda_chain, S_bar_sigma, cov_deltac = results_ssvi_c.values()
    mu_sigma_inv = [T * np.linalg.inv(S_bar_sigma[c]) for c in range(C)]
    Y, F, FF, idx_deltac, size_deltac, Pc, Lambda_inv, Lambda_inv_sum = ssvi_i_pack.values()

    # lambda: sample from a KDE fit to the converged ULA chain (log-space, since lambda > 0)
    lam_chain = np.asarray(q_lambda_chain)
    kde_lam = gaussian_kde(np.log(lam_chain))
    lam_samples = np.exp(kde_lam.resample(n_samples, seed=rng).flatten())

    # V_deltac(lambda), computed once and reused for both beta0 and delta_c
    V_deltac = calc_V_deltac2(lam_samples, mu_sigma_inv, FF, Lambda_inv, size_deltac, Pc, C, N, K)

    # beta_0 | lambda
    V_beta0 = calc_V_beta02(lam_samples, V_deltac, Lambda_inv, Lambda_inv_sum, C, N, K)
    mu_beta0 = calc_mu_beta02(lam_samples, V_deltac, mu_sigma_inv, V_beta0, Y, F, Lambda_inv, Pc, C, N, K)

    size_beta0 = mu_beta0.shape[-1]
    L_beta0 = np.linalg.cholesky(V_beta0)
    z_beta0 = rng.normal(size=(n_samples, size_beta0))
    beta_0_samples = mu_beta0 + np.einsum('nij,nj->ni', L_beta0, z_beta0)

    # delta_c | lambda, beta0
    mu_deltac = calc_mu_deltac2(lam_samples, beta_0_samples, V_deltac, mu_sigma_inv, Y, F, Lambda_inv, size_deltac, Pc, C, N, K)

    delta_c_samples_arr = np.empty((n_samples, C, size_deltac))
    for c in range(C):
        L_c = np.linalg.cholesky(V_deltac[c])
        z = rng.normal(size=(n_samples, size_deltac))
        delta_c_samples_arr[:, c, :] = mu_deltac[c] + np.einsum('nij,nj->ni', L_c, z)

    beta_c_samples_arr = delta_c_samples_arr[:, :, :N*K]
    gamma_c_samples_arr = delta_c_samples_arr[:, :, N*K:]

    Sigma_c_samples = np.stack(
        [invwishart.rvs(T, S_bar_sigma[c], size=n_samples, random_state=rng) for c in range(C)], axis=1
    )

    return {
        'beta_0': list(beta_0_samples),
        'lam': list(lam_samples),
        'beta_c': list(beta_c_samples_arr),
        'gamma_c': list(gamma_c_samples_arr),
        'delta_c': list(delta_c_samples_arr),
        'Sigma_c': list(Sigma_c_samples),
    }


### Uncertainty Quantification Factor ###

def compute_cov_true(results_gibbs, C):
    """Empirical delta_c covariance from Gibbs draws, treated as ground truth."""
    cov_true = []
    for c in range(C):
        delta_c_draws = np.array([results_gibbs["delta_c"][t][c] for t in range(len(results_gibbs["delta_c"]))])
        cov_true.append(np.cov(delta_c_draws.T))
    return cov_true


def extract_cov_mfvi_pipeline(results_mfvi, mfvi_pack, C):
    """Per-country delta_c covariance block sliced out of MFVI's full V_delta (used within pipeline)."""
    idx_deltac = mfvi_pack["idx_deltac"]
    size_deltac = mfvi_pack["size_deltac"]
    V_delta = results_mfvi["V_delta"]

    cov_mfvi = []
    for c in range(C):
        start = idx_deltac[c]
        cov_mfvi.append(V_delta[start:start + size_deltac, start:start + size_deltac])
    return cov_mfvi

def cov2corr(cov):
    """Convert a covariance matrix to a correlation matrix."""
    d = np.sqrt(np.diag(cov))
    d[d == 0] = 1e-12
    return cov / np.outer(d, d)

def extract_cov_mfvi(V_delta, N, K, Z_width, C):
    """Slice per-country delta_c covariance blocks directly out of MFVI's full
    V_delta, without needing the full mfvi_pack (used outside of pipeline)."""
    size_beta0 = N * K
    size_deltac = N * K + N * Z_width
    idx_deltac = [size_beta0 + c * size_deltac for c in range(C)]
    return [V_delta[idx_deltac[c]:idx_deltac[c]+size_deltac,
                     idx_deltac[c]:idx_deltac[c]+size_deltac] for c in range(C)]


def UQF(cov_true, cov_est):
    """Compute the Uncertainty Quantification Factor (UQF) between a reference
    covariance matrix and an estimated covariance matrix."""
    eigenvalues = eigh(cov_true, cov_est, eigvals_only=True)
    return 1 / np.max(eigenvalues)


### Mean Absolute Error of Off-Diagonal Correlations ###

def corr_mae_table(results, N, K, Z_width):
    """Compute, per country, the mean absolute correlation error of each VI
    method's delta_c correlation structure (excluding diagonals) against Gibbs."""
    C = results["C"]
    method_names = ["MFVI", "SSVI-I", "SSVI-C"]
    V_delta = results["mfvi"]["results"]["V_delta"]
    cov_mfvi = extract_cov_mfvi(V_delta, N, K, Z_width, C)
    cov_ssvi_i = results["ssvi_i"]["results"]["cov_deltac"]
    cov_ssvi_c = results["ssvi_c"]["results"]["cov_deltac"]
    cov_true = results["cov_true"]
    methods = {"MFVI": cov_mfvi, "SSVI-I": cov_ssvi_i, "SSVI-C": cov_ssvi_c}

    country_names = results["config"].country_names
    table = {name: [] for name in method_names}

    for c in range(C):
        corr_true = cov2corr(cov_true[c])
        np.fill_diagonal(corr_true, np.nan)
        for name, cov in methods.items():
            corr_method = cov2corr(cov[c])
            np.fill_diagonal(corr_method, np.nan)
            diff = corr_method - corr_true
            table[name].append(np.nanmean(np.abs(diff)))

    return pd.DataFrame(table, index=country_names).round(4)

### Accuracy Measure ###

def faes_accuracy(vi_samples, gibbs_samples, positive_support=False, grid_size=500):
    """Compute the accuracy score between a VI method's
    approximate posterior samples and Gibbs posterior samples for
    a single scalar parameter."""
    vi_samples = np.asarray(vi_samples)
    gibbs_samples = np.asarray(gibbs_samples)

    if positive_support:
        vi_samples = np.log(vi_samples)
        gibbs_samples = np.log(gibbs_samples)

    kde_q = gaussian_kde(vi_samples)
    kde_p = gaussian_kde(gibbs_samples)

    # Bounds per paper: smallest/largest value across sample sets, no padding
    lo = min(vi_samples.min(), gibbs_samples.min())
    hi = max(vi_samples.max(), gibbs_samples.max())
    grid = np.linspace(lo, hi, grid_size)

    iae = np.trapezoid(np.abs(kde_q(grid) - kde_p(grid)), grid)
    return 100 * (1 - 0.5 * iae)


def _faes_grid(vi_arr, gibbs_arr, positive_support=False, n_jobs=-1):
    """Compute accuracy scores for a (country, dimension) grid of scalar
    parameters, in parallel."""
    T, C, D = vi_arr.shape
    pairs = [(c, d) for c in range(C) for d in range(D)]

    results = Parallel(n_jobs=n_jobs)(
        delayed(faes_accuracy)(vi_arr[:, c, d], gibbs_arr[:, c, d], positive_support)
        for c, d in pairs
    )
    out = np.empty((C, D))
    for (c, d), val in zip(pairs, results):
        out[c, d] = val
    return out


def _faes_vec(vi_arr, gibbs_arr, positive_support=False, n_jobs=-1):
    """Compute Faes accuracy scores for a vector of scalar parameters (no
    country axis), in parallel."""
    T, D = vi_arr.shape
    results = Parallel(n_jobs=n_jobs)(
        delayed(faes_accuracy)(vi_arr[:, d], gibbs_arr[:, d], positive_support)
        for d in range(D)
    )
    return np.array(results)


def prepare_gibbs_faes_arrays(gibbs_samples):
    """Convert Gibbs sample lists to arrays once, for reuse across every VI
    method's Faes scoring."""
    beta_c_gibbs = np.array(gibbs_samples['beta_c'])    
    gamma_c_gibbs = np.array(gibbs_samples['gamma_c']) 
    beta_0_gibbs = np.array(gibbs_samples['beta_0'])        

    # Sigma_c diagonals: extract diag once, vectorized, before parallel calls
    Sigma_c_gibbs_full = np.array(gibbs_samples['Sigma_c'])  
    Sigma_c_gibbs = np.diagonal(Sigma_c_gibbs_full, axis1=2, axis2=3)

    return {
        'beta_c': beta_c_gibbs,
        'gamma_c': gamma_c_gibbs,
        'beta_0': beta_0_gibbs,
        'Sigma_c': Sigma_c_gibbs,
        'lam': gibbs_samples['lam'],
    }


def compute_faes_scores(vi_samples, gibbs_arrays):
    """Accuracy of a VI method's samples against Gibbs, for every
    parameter block."""
    beta_c_vi = np.array(vi_samples['beta_c'])         
    gamma_c_vi = np.array(vi_samples['gamma_c'])    
    beta_0_vi = np.array(vi_samples['beta_0']) 

    # Sigma_c diagonals: extract diag once, vectorized, before parallel calls
    Sigma_c_vi_full = np.array(vi_samples['Sigma_c'])      
    Sigma_c_vi = np.diagonal(Sigma_c_vi_full, axis1=2, axis2=3)    

    scores = {}
    scores['beta_c'] = _faes_grid(beta_c_vi, gibbs_arrays['beta_c'])
    scores['gamma_c'] = _faes_grid(gamma_c_vi, gibbs_arrays['gamma_c'])
    scores['beta_0'] = _faes_vec(beta_0_vi, gibbs_arrays['beta_0'])
    # positive support therefore log-space
    scores['lam'] = faes_accuracy(vi_samples['lam'], gibbs_arrays['lam'], positive_support=True)
    scores['Sigma_c'] = _faes_grid(Sigma_c_vi, gibbs_arrays['Sigma_c'], positive_support=True)
    return scores


### Impulse Response Functions ###

def _build_lag_matrices(beta_c, N, L, K):
    """Reshape a country's stacked beta_c coefficient vector into per-lag
    coefficient matrices."""
    A_list = [np.zeros((N, N)) for _ in range(L)]
    for i in range(N):
        eq_block = beta_c[i * K: i * K + N * L]          # drop w_lags tail
        eq_lags = eq_block.reshape(L, N)                  # [lag, variable]
        for l in range(L):
            A_list[l][i, :] = eq_lags[l, :]
    return A_list


def _build_companion(A_list, N, L):
    """Stack A_1..A_L into the top block row of the NL x NL companion matrix. """
    NL = N * L
    Acomp = np.zeros((NL, NL))
    Acomp[:N, :] = np.hstack(A_list)          # [A_1 A_2 ... A_L]
    if L > 1:
        Acomp[N:, :NL - N] = np.eye(NL - N)   # shift-down identity blocks
    return Acomp


def _draw_admissible_G(Sigma_c, sign_pattern, max_tries=1000, rng=None):
    """Draw a structural impact matrix G_c whose 2x2 rotated block satisfies a
    given sign pattern, by rejection sampling over random rotations.
    Raises RuntimeError if no admissible rotation is found within max_tries attempts."""
    if rng is None:
        rng = np.random.default_rng()

    N = Sigma_c.shape[0]
    P_c = np.linalg.cholesky(Sigma_c)

    for attempt in range(1, max_tries + 1):
        theta = rng.uniform(0, 2 * np.pi)
        c, s = np.cos(theta), np.sin(theta)
        V = np.array([[c, -s],
                      [s,  c]])

        Q = np.eye(N)
        Q[N - 2:, N - 2:] = V

        G_c = P_c @ Q

        ok = all(G_c[row, col] * sgn > 0 for row, col, sgn in sign_pattern)
        if ok:
            return G_c, attempt

    raise RuntimeError(f"No admissible rotation found in {max_tries} tries.")


def compute_irfs(beta_samples, sigma_samples, N, L, K, C, H=36,
                  shock_idx=2,
                  sign_pattern=((2, 2, 1.0), (3, 2, -1.0), (2, 3, 1.0), (3, 3, 1.0)),
                  max_tries=1000, seed=None):
    """Compute impulse responses to a monetary policy shock for each posterior
    draw and country, stopping at the first admissible rotation per draw."""
    rng = np.random.default_rng(seed)
    n_draws = beta_samples.shape[0]
    NL = N * L

    irfs = np.zeros((n_draws, C, H + 1, N))
    n_tries = np.zeros((n_draws, C), dtype=int)

    for d in range(n_draws):
        for c in range(C):
            beta_c = beta_samples[d, c]
            Sigma_c = sigma_samples[d, c]

            # reduced-form dynamics
            A_list = _build_lag_matrices(beta_c, N, L, K)
            Acomp = _build_companion(A_list, N, L)

            # structural impact matrix
            G_c, tries = _draw_admissible_G(
                Sigma_c, sign_pattern=sign_pattern, max_tries=max_tries, rng=rng
            )
            n_tries[d, c] = tries
            impact = G_c[:, shock_idx]           # period-0 response, length N

            # propagate through companion form
            Y = np.zeros(NL)
            Y[:N] = impact
            irfs[d, c, 0, :] = Y[:N]
            for h in range(1, H + 1):
                Y = Acomp @ Y
                irfs[d, c, h, :] = Y[:N]

    return irfs, n_tries


def compute_wasserstein_curve(gibbs_irfs, vi_irfs):
    """Compute the Wasserstein distance between Gibbs and VI IRF draw
    distributions, at every country/horizon/variable."""
    _, C, H_plus_1, N = gibbs_irfs.shape
    distances = np.zeros((C, H_plus_1, N))

    for c in range(C):
        for h in range(H_plus_1):
            for n in range(N):
                distances[c, h, n] = wasserstein_distance(
                    gibbs_irfs[:, c, h, n], vi_irfs[:, c, h, n]
                )
    return distances


### Effect of lambda on coefficient means ###

def lambda_dispersion_change(lam, coef_c_k, beta0_k_samples, pct=0.25,
                              clip_lo=None, clip_hi=None):
    """Proportional change in across-country dispersion about beta_0 between
    the top and bottom pct of the lambda posterior, for one coefficient,
    plus the same change normalised by the relative lambda range spanned
    (elasticity). Allows for clipping of lambda ranges if required.
    Returns values alongside the inner range between upper and lower parts. """
    lam = np.asarray(lam)
    beta0_k = np.asarray(beta0_k_samples)

    if clip_lo is not None and clip_hi is not None:
        in_range = (lam >= clip_lo) & (lam <= clip_hi)
        lam = lam[in_range]
        coef_c_k = coef_c_k[:, in_range]
        beta0_k = beta0_k[in_range]

    n_samples = lam.size
    if n_samples == 0:
        return np.nan, np.nan, np.nan, np.nan, n_samples

    lo_cut, hi_cut = np.quantile(lam, [pct, 1 - pct])

    def disp(mask):
        vals = coef_c_k[:, mask] - beta0_k[mask]
        return np.sqrt(np.mean(vals ** 2))

    disp_low = disp(lam <= lo_cut)
    disp_high = disp(lam >= hi_cut)

    if disp_low == 0 or hi_cut == lo_cut:
        return np.nan, np.nan, lo_cut, hi_cut, n_samples

    delta_norm = (disp_high - disp_low) / disp_low
    elasticity = delta_norm / ((hi_cut - lo_cut) / lo_cut)

    return delta_norm, elasticity, lo_cut, hi_cut, n_samples


def dispersion_change_real_data(results, pct=0.25, clipped=False):
    """For each method and each coefficient, compute the percentage change
    in across-country dispersion about beta_0 between the top and bottom
    pct of the lambda posterior, plus the range-normalised version
    (elasticity). Clipped=True means values clipped to MFVI posterior range."""
    clip_lo, clip_hi = None, None
    if clipped:
        lam_mfvi = np.array(results["mfvi"]["samples"]["lam"])
        clip_lo, clip_hi = lam_mfvi.min(), lam_mfvi.max()

    rows = []
    for method in ["mfvi", "ssvi_i", "ssvi_c", "gibbs"]:
        d = results[method]["samples" if method != "gibbs" else "results"]
        beta_c = np.array(d["beta_c"])   # (n_samples, C, NK)
        beta_0 = np.array(d["beta_0"])   # (n_samples, NK)
        lam = np.array(d["lam"])

        for k in range(beta_c.shape[2]):
            coef_c_k = beta_c[:, :, k].T  # (C, n_samples)
            delta, elast, lo_cut, hi_cut, n_samples = lambda_dispersion_change(
                lam, coef_c_k, beta_0[:, k], pct=pct,
                clip_lo=clip_lo, clip_hi=clip_hi,
            )
            rows.append({
                "method": method, "k": k, "delta_norm": delta,
                "elasticity": elast, "lam_lo": lo_cut, "lam_hi": hi_cut,
                "lam_range": hi_cut - lo_cut if np.isfinite(hi_cut) else np.nan,
                "n_samples": n_samples,
            })

    return pd.DataFrame(rows)


def dispersion_change_by_seed(results_by_seed, pct=0.2, clipped=False):
    """"For each method and each coefficient, compute the percentage change
    in across-country dispersion about beta_0 between the top and bottom
    pct of the lambda posterior, plus the range-normalised version
    (elasticity), applied per seed and combined into
    one DataFrame with a "seed" column. Clipped=True means values clipped to MFVI posterior range."""
    dfs = []
    for seed, results in results_by_seed.items():
        df = dispersion_change_real_data(results, pct=pct, clipped=clipped)
        df["seed"] = seed
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


def summarise_dispersion_change(df, group_cols=("method",), decimals=4, lam_sci_decimals=2):
    """Mean and std of delta_norm and elasticity, plus mean n_samples,
    with lambda range reported only when each group corresponds to a
    single seed.  When pooling across seeds, lambda ranges
    are on different scales per seed and are omitted entirely."""
    clean = df.dropna(subset=["delta_norm", "elasticity"])
    stats = clean.groupby(list(group_cols))[["delta_norm", "elasticity"]].agg(
        ["mean", "std"]
    )
    stats.columns = [f"{var}_{stat}" for var, stat in stats.columns]
    stats = stats.round(decimals)

    per_seed = "seed" in group_cols or "seed" not in clean.columns
    n_col_name = "n_samples" if per_seed else "n_samples_mean"
    n_stats = clean.groupby(list(group_cols))["n_samples"].mean().round(decimals).rename(n_col_name)
    stats = stats.join(n_stats)

    if per_seed:
        lam_stats = clean.groupby(list(group_cols))[["lam_lo", "lam_hi", "lam_range"]].mean()
        lam_stats = lam_stats.map(lambda x: f"{x:.{lam_sci_decimals}e}")
        return stats.join(lam_stats)

    return stats

### Runtime and iteration counts ###

def runtime_iteration_table(results):
    """Wall-clock runtime and iteration count for each method."""
    methods = [("MFVI", "mfvi"), ("SSVI-I", "ssvi_i"), ("SSVI-C", "ssvi_c"), ("Gibbs", "gibbs")]
    rows = [
        {
            "method": name,
            "runtime_s": results[key]["runtime"],
            "n_iterations": len(results[key]["elbo"]) if key != "gibbs" else np.nan,
        }
        for name, key in methods
    ]
    return pd.DataFrame(rows).set_index("method")

def runtime_table_seed(results_by_seed):
    """Mean wall-clock runtime per method, averaged over seeds."""
    methods = [("MFVI", "mfvi"), ("SSVI-I", "ssvi_i"), ("SSVI-C", "ssvi_c"), ("Gibbs", "gibbs")]
    rows = [
        {"method": name, "runtime_s": results[key]["runtime"]}
        for results in results_by_seed.values()
        for name, key in methods
    ]
    stats = pd.DataFrame(rows).groupby("method")["runtime_s"].mean().rename("runtime_mean_s").to_frame()
    return stats.loc[[name for name, _ in methods]]

### Comparison to true parameters ###

def coverage_table(results_by_seed, true_by_seed, param_key, methods,
                    levels=(0.5, 0.8, 0.95), axis_type="country"):
    """Compute empirical credible-interval coverage of a parameter, pooled
    across seeds (and countries, if applicable), for each method."""
    # axis_type: "country" (e.g. beta_c, gamma_c), "vector" (e.g. beta_0)
    hits = {method: {level: [] for level in levels} for method in methods}

    for method in methods:
        result_key = "results" if method == "gibbs" else "samples"
        for seed, results in results_by_seed.items():
            samples = np.array(results[method][result_key][param_key])
            true_vals = np.array(true_by_seed[seed][param_key])

            if axis_type == "country":
                for c in range(samples.shape[1]):
                    s, t = samples[:, c, :], true_vals[c]
                    for level in levels:
                        alpha = 1 - level
                        lo, hi = np.quantile(s, [alpha / 2, 1 - alpha / 2], axis=0)
                        hits[method][level].append((lo <= t) & (t <= hi))
            else:  # vector
                s, t = samples, true_vals
                for level in levels:
                    alpha = 1 - level
                    lo, hi = np.quantile(s, [alpha / 2, 1 - alpha / 2], axis=0)
                    hits[method][level].append((lo <= t) & (t <= hi))

    table = pd.DataFrame({
        method: {level: np.concatenate(hits[method][level]).mean() for level in levels}
        for method in methods
    }).T
    table.columns = [f"{int(l*100)}%" for l in table.columns]
    return table

