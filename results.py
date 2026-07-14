import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import invgamma, invwishart, gaussian_kde, wasserstein_distance
from scipy.linalg import eigh
from joblib import Parallel, delayed

from ssvi import calc_V_deltac, calc_mu_deltac
from ssvi_2 import calc_V_beta02, calc_mu_beta02, calc_V_deltac2, calc_mu_deltac2


"""Posterior sample reconstruction from each method's variational/MCMC output"""

def sample_from_mfvi(results_mfvi, mfvi_pack, C, N, K, T, n_samples=10000):
    mu_delta, V_delta, v_bar, s_bar, S_bar_sigma = results_mfvi.values()

    idx_deltac = mfvi_pack["idx_deltac"]
    size_deltac = mfvi_pack["size_deltac"]

    # --- delta samples (beta_0, beta_c, gamma_c, delta_c all come from this) ---
    L = np.linalg.cholesky(V_delta)
    deltas = mu_delta + (L @ np.random.normal(size=(len(mu_delta), n_samples))).T  # (n_samples, size_delta)

    beta_0_samples = deltas[:, :idx_deltac[0]]  # (n_samples, size_beta0)

    beta_c_samples = [deltas[:, idx_deltac[c]:idx_deltac[c] + N*K] for c in range(C)]              # list of (n_samples, N*K)
    gamma_c_samples = [deltas[:, idx_deltac[c] + N*K:idx_deltac[c] + size_deltac] for c in range(C)]  # list of (n_samples, size_gammac)
    delta_c_samples = [deltas[:, idx_deltac[c]:idx_deltac[c] + size_deltac] for c in range(C)]     # list of (n_samples, size_deltac)

    # --- lambda samples (cheap, vectorize trivially) ---
    lam_samples = invgamma.rvs(s_bar/2, scale=v_bar/2, size=n_samples)

    # --- Sigma_c samples (vectorized per country via scipy's size argument) ---
    Sigma_c_samples = [invwishart.rvs(T, S_bar_sigma[c], size=n_samples) for c in range(C)]  # each: (n_samples, N, N)

    return {
        'beta_0': list(beta_0_samples),
        'lam': list(lam_samples),
        'beta_c': [[beta_c_samples[c][n] for c in range(C)] for n in range(n_samples)],
        'gamma_c': [[gamma_c_samples[c][n] for c in range(C)] for n in range(n_samples)],
        'delta_c': [[delta_c_samples[c][n] for c in range(C)] for n in range(n_samples)],
        'Sigma_c': [[Sigma_c_samples[c][n] for c in range(C)] for n in range(n_samples)],
    }


def sample_from_ssvi(results_ssvi, ssvi_pack, C, N, K, T, n_samples=10000):
    mu_beta0, V_beta0, q_lambda_chain, S_bar_sigma, cov_deltac = results_ssvi.values()
    mu_sigma_inv = [T * np.linalg.inv(S_bar_sigma[c]) for c in range(C)]
    Y, F, FF, idx_deltac, size_deltac, Pc, Lambda_inv, Lambda_inv_sum = ssvi_pack.values()

    # lambda: sample from a KDE fit to the converged ULA chain (log-space, since lambda > 0)
    lam_chain = np.asarray(q_lambda_chain)
    kde_lam = gaussian_kde(np.log(lam_chain))
    lam_samples = np.exp(kde_lam.resample(n_samples).flatten())

    # beta_0: independent draw, paired with lam_samples by index
    L_beta0 = np.linalg.cholesky(V_beta0)
    beta_0_samples = mu_beta0 + (L_beta0 @ np.random.normal(size=(len(mu_beta0), n_samples))).T  # (n_samples, size_beta0)

    # V_deltac, mu_deltac: both batched over the same (lam, beta0) pairs
    V_deltac = calc_V_deltac(lam_samples, mu_sigma_inv, FF, Lambda_inv, size_deltac, Pc, C, N, K)
    mu_deltac = calc_mu_deltac(lam_samples, beta_0_samples, V_deltac, mu_sigma_inv, Y, F, Lambda_inv, size_deltac, Pc, C, N, K)

    delta_c_samples_arr = np.empty((n_samples, C, size_deltac))
    for c in range(C):
        L_c = np.linalg.cholesky(V_deltac[c])                 # (n_samples, size_deltac, size_deltac)
        z = np.random.normal(size=(n_samples, size_deltac))
        delta_c_samples_arr[:, c, :] = mu_deltac[c] + np.einsum('nij,nj->ni', L_c, z)

    beta_c_samples_arr = delta_c_samples_arr[:, :, :N*K]
    gamma_c_samples_arr = delta_c_samples_arr[:, :, N*K:]

    Sigma_c_samples = np.stack(
        [invwishart.rvs(T, S_bar_sigma[c], size=n_samples) for c in range(C)], axis=1
    )

    return {
        'beta_0': list(beta_0_samples),
        'lam': list(lam_samples),
        'beta_c': list(beta_c_samples_arr),
        'gamma_c': list(gamma_c_samples_arr),
        'delta_c': list(delta_c_samples_arr),
        'Sigma_c': list(Sigma_c_samples),
    }


def sample_from_ssvi2(results_ssvi2, ssvi_pack, C, N, K, T, n_samples=10000):
    q_lambda_chain, S_bar_sigma, cov_deltac = results_ssvi2.values()
    mu_sigma_inv = [T * np.linalg.inv(S_bar_sigma[c]) for c in range(C)]
    Y, F, FF, idx_deltac, size_deltac, Pc, Lambda_inv, Lambda_inv_sum = ssvi_pack.values()

    # lambda: sample from a KDE fit to the converged ULA chain (log-space, since lambda > 0)
    lam_chain = np.asarray(q_lambda_chain)
    kde_lam = gaussian_kde(np.log(lam_chain))
    lam_samples = np.exp(kde_lam.resample(n_samples).flatten())

    # V_deltac(lambda), computed once and reused for both beta0 and delta_c
    V_deltac = calc_V_deltac2(lam_samples, mu_sigma_inv, FF, Lambda_inv, size_deltac, Pc, C, N, K)

    # beta_0 | lambda
    V_beta0 = calc_V_beta02(lam_samples, V_deltac, Lambda_inv, Lambda_inv_sum, C, N, K)
    mu_beta0 = calc_mu_beta02(lam_samples, V_deltac, mu_sigma_inv, V_beta0, Y, F, Lambda_inv, Pc, C, N, K)

    size_beta0 = mu_beta0.shape[-1]
    L_beta0 = np.linalg.cholesky(V_beta0)
    z_beta0 = np.random.normal(size=(n_samples, size_beta0))
    beta_0_samples = mu_beta0 + np.einsum('nij,nj->ni', L_beta0, z_beta0)

    # delta_c | lambda, beta0
    mu_deltac = calc_mu_deltac2(lam_samples, beta_0_samples, V_deltac, mu_sigma_inv, Y, F, Lambda_inv, size_deltac, Pc, C, N, K)

    delta_c_samples_arr = np.empty((n_samples, C, size_deltac))
    for c in range(C):
        L_c = np.linalg.cholesky(V_deltac[c])
        z = np.random.normal(size=(n_samples, size_deltac))
        delta_c_samples_arr[:, c, :] = mu_deltac[c] + np.einsum('nij,nj->ni', L_c, z)

    beta_c_samples_arr = delta_c_samples_arr[:, :, :N*K]
    gamma_c_samples_arr = delta_c_samples_arr[:, :, N*K:]

    Sigma_c_samples = np.stack(
        [invwishart.rvs(T, S_bar_sigma[c], size=n_samples) for c in range(C)], axis=1
    )

    return {
        'beta_0': list(beta_0_samples),
        'lam': list(lam_samples),
        'beta_c': list(beta_c_samples_arr),
        'gamma_c': list(gamma_c_samples_arr),
        'delta_c': list(delta_c_samples_arr),
        'Sigma_c': list(Sigma_c_samples),
    }


"""UQF (Uncertainty Quantification Factor)"""

def compute_cov_true(results_gibbs, C):
    """Empirical delta_c covariance from Gibbs draws, treated as ground truth."""
    cov_true = []
    for c in range(C):
        delta_c_draws = np.array([results_gibbs["delta_c"][t][c] for t in range(len(results_gibbs["delta_c"]))])
        cov_true.append(np.cov(delta_c_draws.T))
    return cov_true


def extract_cov_mfvi(results_mfvi, mfvi_pack, C):
    """Per-country delta_c covariance block sliced out of MFVI's full V_delta."""
    idx_deltac = mfvi_pack["idx_deltac"]
    size_deltac = mfvi_pack["size_deltac"]
    V_delta = results_mfvi["V_delta"]

    cov_mfvi = []
    for c in range(C):
        start = idx_deltac[c]
        cov_mfvi.append(V_delta[start:start + size_deltac, start:start + size_deltac])
    return cov_mfvi


def UQF(cov_true, cov_est):
    eigenvalues = eigh(cov_true, cov_est, eigvals_only=True)
    return 1 / np.max(eigenvalues)


def compute_uqf(cov_true, cov_est_list, C):
    return [UQF(cov_true[c], cov_est_list[c]) for c in range(C)]


"""Accuracy Measure (Faes et al. 2011, ter Steege eq. 20) — vectorized + parallel"""

def faes_accuracy(vi_samples, gibbs_samples, positive_support=False, grid_size=500):
    vi_samples = np.asarray(vi_samples)
    gibbs_samples = np.asarray(gibbs_samples)

    if positive_support:
        vi_samples = np.log(vi_samples)
        gibbs_samples = np.log(gibbs_samples)

    kde_q = gaussian_kde(vi_samples)
    kde_p = gaussian_kde(gibbs_samples)

    # Bounds per paper: smallest/largest value across BOTH sample sets, no padding
    lo = min(vi_samples.min(), gibbs_samples.min())
    hi = max(vi_samples.max(), gibbs_samples.max())
    grid = np.linspace(lo, hi, grid_size)

    iae = np.trapezoid(np.abs(kde_q(grid) - kde_p(grid)), grid)
    return 100 * (1 - 0.5 * iae)


def _faes_grid(vi_arr, gibbs_arr, positive_support=False, n_jobs=-1):
    """
    vi_arr, gibbs_arr: shape (T, C, D) — draws x countries x scalar-dim.
    Returns shape (C, D) of accuracy scores, computed in parallel over (c,d) pairs.
    """
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
    """
    vi_arr, gibbs_arr: shape (T, D) — draws x scalar-dim (no country axis).
    Returns shape (D,) of accuracy scores.
    """
    T, D = vi_arr.shape
    results = Parallel(n_jobs=n_jobs)(
        delayed(faes_accuracy)(vi_arr[:, d], gibbs_arr[:, d], positive_support)
        for d in range(D)
    )
    return np.array(results)


def prepare_gibbs_faes_arrays(gibbs_samples):
    """Convert Gibbs sample lists to arrays once, for reuse across every VI method's Faes scoring."""
    beta_c_gibbs = np.array(gibbs_samples['beta_c'])          # (T, C, N*K)
    gamma_c_gibbs = np.array(gibbs_samples['gamma_c'])        # (T, C, N*n_zc)
    beta_0_gibbs = np.array(gibbs_samples['beta_0'])          # (T, N*K)

    # Sigma_c diagonals: extract diag once, vectorized, before parallel calls
    Sigma_c_gibbs_full = np.array(gibbs_samples['Sigma_c'])   # (T, C, N, N)
    Sigma_c_gibbs = np.diagonal(Sigma_c_gibbs_full, axis1=2, axis2=3)   # (T, C, N)

    return {
        'beta_c': beta_c_gibbs,
        'gamma_c': gamma_c_gibbs,
        'beta_0': beta_0_gibbs,
        'Sigma_c': Sigma_c_gibbs,
        'lam': gibbs_samples['lam'],
    }


def compute_faes_scores(vi_samples, gibbs_arrays):
    """Faes accuracy of a VI method's samples against Gibbs, for every parameter block.

    gibbs_arrays: output of prepare_gibbs_faes_arrays, computed once and shared
    across methods so the Gibbs conversion isn't redone per method.
    """
    beta_c_vi = np.array(vi_samples['beta_c'])          # (T, C, N*K)
    gamma_c_vi = np.array(vi_samples['gamma_c'])        # (T, C, N*n_zc)
    beta_0_vi = np.array(vi_samples['beta_0'])          # (T, N*K)

    # Sigma_c diagonals: extract diag once, vectorized, before parallel calls
    Sigma_c_vi_full = np.array(vi_samples['Sigma_c'])       # (T, C, N, N)
    Sigma_c_vi = np.diagonal(Sigma_c_vi_full, axis1=2, axis2=3)       # (T, C, N)

    scores = {}
    scores['beta_c'] = _faes_grid(beta_c_vi, gibbs_arrays['beta_c'])
    scores['gamma_c'] = _faes_grid(gamma_c_vi, gibbs_arrays['gamma_c'])
    scores['beta_0'] = _faes_vec(beta_0_vi, gibbs_arrays['beta_0'])
    scores['lam'] = faes_accuracy(vi_samples['lam'], gibbs_arrays['lam'], positive_support=True)
    scores['Sigma_c'] = _faes_grid(Sigma_c_vi, gibbs_arrays['Sigma_c'], positive_support=True)
    return scores


def plot_accuracy_boxplots(results_faes, method_name, C):
    """Boxplots of Faes accuracy per parameter block, for one VI method vs Gibbs."""
    country_labels = [f'C{c+1}' for c in range(C)]

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # beta_c: one box per country
    axes[0, 0].boxplot([results_faes['beta_c'][c] for c in range(C)], labels=country_labels)
    axes[0, 0].set_title(r'$\beta_c$')
    axes[0, 0].set_ylabel('Accuracy (%)')

    # gamma_c: one box per country
    axes[0, 1].boxplot([results_faes['gamma_c'][c] for c in range(C)], labels=country_labels)
    axes[0, 1].set_title(r'$\gamma_c$')
    axes[0, 1].set_ylabel('Accuracy (%)')

    # Sigma_c diagonals: one box per country
    axes[0, 2].boxplot([results_faes['Sigma_c'][c] for c in range(C)], labels=country_labels)
    axes[0, 2].set_title(r'$\Sigma_c$ diagonals')
    axes[0, 2].set_ylabel('Accuracy (%)')

    # beta_0: single box over all N*K coefficients
    axes[1, 0].boxplot([results_faes['beta_0']], labels=[r'$\beta_0$'])
    axes[1, 0].set_ylabel('Accuracy (%)')

    # lambda: single value, shown as a point
    axes[1, 1].scatter([1], [results_faes['lam']], s=100, zorder=5)
    axes[1, 1].set_xlim(0.5, 1.5)
    axes[1, 1].set_xticks([1])
    axes[1, 1].set_xticklabels([r'$\lambda$'])
    axes[1, 1].set_ylabel('Accuracy (%)')

    axes[1, 2].set_visible(False)

    for ax in axes.flat:
        if ax.get_visible():
            ax.set_ylim(0, 100)
            ax.axhline(y=95, color='grey', linestyle='--', linewidth=0.8)
            ax.grid(axis='y', alpha=0.3)

    fig.suptitle(f'Faes et al. Accuracy: {method_name} vs Gibbs', fontsize=14)
    plt.tight_layout()
    plt.show()


"""Impulse Response Functions"""

def _build_lag_matrices(beta_c, N, L, K):
    """
    beta_c: (N*K,) equation-stacked coefficient vector for one country, one draw.
    Matches kron(I_N, X_c) @ beta_c convention, where each equation's K-length
    block has columns ordered [y_lags (lag-major, N*L), w_lags (rest)].

    Returns A_list: list of L matrices, each (N, N), where A_list[l-1] = A_l,
    i.e. the coefficient matrix on y_{t-l} in y_t = A_1 y_{t-1} + ... + A_L y_{t-L} + ...
    A_list[l-1][i, j] = coefficient of equation i on variable j at lag l.
    """
    A_list = [np.zeros((N, N)) for _ in range(L)]
    for i in range(N):
        eq_block = beta_c[i * K: i * K + N * L]          # drop w_lags tail
        eq_lags = eq_block.reshape(L, N)                  # [lag, variable]
        for l in range(L):
            A_list[l][i, :] = eq_lags[l, :]
    return A_list


def _build_companion(A_list, N, L):
    """Stack A_1..A_L into the top block row of the NL x NL companion matrix."""
    NL = N * L
    Acomp = np.zeros((NL, NL))
    Acomp[:N, :] = np.hstack(A_list)          # [A_1 A_2 ... A_L]
    if L > 1:
        Acomp[N:, :NL - N] = np.eye(NL - N)   # shift-down identity blocks
    return Acomp


def _draw_admissible_G(Sigma_c, sign_pattern, shock_idx=2, max_tries=1000, rng=None):
    """
    Sigma_c: (N, N) covariance matrix.
    Finds G_c = P_c @ Q, Q = blockdiag(I_{N-2}, V) with V a random 2x2
    rotation matrix parameterized by a single angle theta ~ Uniform(0, 2*pi),
    such that G_c satisfies the full sign pattern on the 2x2 rotated block
    (rows 2,3 x columns 2,3, per eq. 11: interest/exchange rows against
    both the monetary policy shock column and the remaining column):
        G_c[row, col] * sign > 0   for each (row, col, sign) in sign_pattern
    Returns (G_c, n_tries). Raises RuntimeError if none found within max_tries.
    """
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


def compute_irfs(beta_samples, sigma_samples, N, L, K, C, H=25,
                  shock_idx=2,
                  sign_pattern=((2, 2, 1.0), (3, 2, -1.0), (2, 3, 1.0), (3, 3, 1.0)),
                  max_tries=1000, seed=None):
    """
    Compute impulse responses to a monetary policy shock for each posterior
    draw and country, stopping at the first admissible rotation per draw.

    Parameters
    ----------
    beta_samples : (n_draws, C, N*K) array
    sigma_samples : (n_draws, C, N, N) array
    N : number of endogenous variables
    L : number of lags
    K : total columns of X_c (N*L + w-lag columns); only the first N*L
        coefficients per equation are used (lag block), the rest (w_lags,
        analogous to gamma_c) are dropped as they cancel in the IRF.
    C : number of countries
    H : horizon (IRF computed for h = 0, ..., H)
    shock_idx : column index (0-based) of the monetary policy shock in G_c.
        Default 2 assumes ordering [output, price, interest rate, exchange
        rate] per eq. (11), so monetary policy is the 3rd shock -> index 2.
    sign_pattern : tuple of (row, col, sign) triples checked against G_c,
        covering the full 2x2 rotated block (both columns 2 and 3, per
        eq. 11), not just the monetary policy shock's own column.
    max_tries : max rejection-sampling attempts per (draw, country).
    seed : RNG seed for reproducibility.

    Returns
    -------
    irfs : (n_draws, C, H+1, N) array
        irfs[d, c, h, :] = response of the N endogenous variables at
        horizon h to a unit monetary policy shock, for draw d, country c.
    n_tries : (n_draws, C) array of how many rotation draws were needed.
    """
    rng = np.random.default_rng(seed)
    n_draws = beta_samples.shape[0]
    NL = N * L

    irfs = np.zeros((n_draws, C, H + 1, N))
    n_tries = np.zeros((n_draws, C), dtype=int)

    for d in range(n_draws):
        for c in range(C):
            beta_c = beta_samples[d, c]
            Sigma_c = sigma_samples[d, c]

            # --- reduced-form dynamics ---
            A_list = _build_lag_matrices(beta_c, N, L, K)
            Acomp = _build_companion(A_list, N, L)

            # --- structural impact matrix ---
            G_c, tries = _draw_admissible_G(
                Sigma_c, sign_pattern=sign_pattern,
                shock_idx=shock_idx, max_tries=max_tries, rng=rng
            )
            n_tries[d, c] = tries
            impact = G_c[:, shock_idx]           # period-0 response, length N

            # --- propagate through companion form ---
            Y = np.zeros(NL)
            Y[:N] = impact
            irfs[d, c, 0, :] = Y[:N]
            for h in range(1, H + 1):
                Y = Acomp @ Y
                irfs[d, c, h, :] = Y[:N]

    return irfs, n_tries


def plot_irfs_comparison(gibbs_irfs, vi_irfs, country_names, variable_names, vi_label="VI"):
    """
    gibbs_irfs: (n_draws_gibbs, C, H+1, N) array from compute_irfs — plotted as
        blue fanchart (5-95 percentile, 5% steps) + black solid median.
    vi_irfs: (n_draws_vi, C, H+1, N) array from compute_irfs — plotted as
        red solid median + red dashed 5/95 percentiles.
    """
    n_draws_g, C, H_plus_1, N = gibbs_irfs.shape
    horizons = np.arange(H_plus_1)

    gibbs_median = np.median(gibbs_irfs, axis=0)          # (C, H+1, N)
    vi_median = np.median(vi_irfs, axis=0)                # (C, H+1, N)
    vi_p5 = np.percentile(vi_irfs, 5, axis=0)
    vi_p95 = np.percentile(vi_irfs, 95, axis=0)

    # fanchart bands in 5% steps: (5,95), (10,90), ..., (45,55)
    band_pairs = [(p, 100 - p) for p in range(5, 50, 5)]

    fig, axes = plt.subplots(N, C, figsize=(3 * C, 2.2 * N), sharex=True)

    for n in range(N):
        for c in range(C):
            ax = axes[n, c]

            for lo, hi in band_pairs:
                lo_band = np.percentile(gibbs_irfs[:, c, :, n], lo, axis=0)
                hi_band = np.percentile(gibbs_irfs[:, c, :, n], hi, axis=0)
                ax.fill_between(horizons, lo_band, hi_band,
                                 color="tab:blue", alpha=0.08)

            ax.plot(horizons, gibbs_median[c, :, n], color="black", linewidth=1.2)

            ax.plot(horizons, vi_median[c, :, n], color="red", linewidth=1.2)
            ax.plot(horizons, vi_p5[c, :, n], color="red", linestyle="--", linewidth=1)
            ax.plot(horizons, vi_p95[c, :, n], color="red", linestyle="--", linewidth=1)

            ax.axhline(0, color="gray", linewidth=0.7, linestyle=":")

            if n == 0:
                ax.set_title(country_names[c])
            if c == 0:
                ax.set_ylabel(variable_names[n])
            if n == N - 1:
                ax.set_xlabel("Horizon")

    plt.tight_layout()
    plt.show()


def compute_wasserstein_curve(gibbs_irfs, vi_irfs):
    """
    gibbs_irfs: (n_draws_gibbs, C, H+1, N)
    vi_irfs:    (n_draws_vi, C, H+1, N)

    Returns: distances, shape (C, H+1, N) — Wasserstein distance between the
    Gibbs and VI empirical draw distributions at each country/horizon/variable.
    """
    _, C, H_plus_1, N = gibbs_irfs.shape
    distances = np.zeros((C, H_plus_1, N))

    for c in range(C):
        for h in range(H_plus_1):
            for n in range(N):
                distances[c, h, n] = wasserstein_distance(
                    gibbs_irfs[:, c, h, n], vi_irfs[:, c, h, n]
                )
    return distances


def plot_wasserstein_grid_comparison(distances_dict, country_names, variable_names):
    """
    distances_dict: dict mapping method label -> (C, H+1, N) array from
        compute_wasserstein_curve, e.g. {"mfvi": wass_mfvi, "SSVI": wass_ssvi,
        "SSVI2": wass_ssvi2}
    Layout matches the IRF grid: rows = variables, columns = countries.
    Each panel overlays one line per method.
    """
    method_names = list(distances_dict.keys())
    C, H_plus_1, N = next(iter(distances_dict.values())).shape
    horizons = np.arange(H_plus_1)

    colors = plt.cm.tab10(np.linspace(0, 1, len(method_names)))

    fig, axes = plt.subplots(N, C, figsize=(3 * C, 2.2 * N), sharex=True)

    for n in range(N):
        for c in range(C):
            ax = axes[n, c]
            for method, color in zip(method_names, colors):
                ax.plot(horizons, distances_dict[method][c, :, n],
                         color=color, linewidth=1.2, label=method)
            ax.axhline(0, color="gray", linewidth=0.7, linestyle=":")

            if n == 0:
                ax.set_title(country_names[c])
            if c == 0:
                ax.set_ylabel(variable_names[n])
            if n == N - 1:
                ax.set_xlabel("Horizon")

    axes[0, 0].legend(loc="upper right", fontsize=8)
    fig.suptitle("Wasserstein distance: Gibbs vs VI methods", y=1.02)
    plt.tight_layout()
    plt.show()
