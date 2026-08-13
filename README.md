# Structured Variational Inference for Hierarchical Panel VAR Models

Code accompanying the MSc dissertation "Structured Variational Inference for
Hierarchical Panel Vector Autoregressive Models" (Imperial College London, 2026).

## Overview

Implements two Star-Structured VI algorithms, SSVI-I and SSVI-C, for the
Bayesian hierarchical Panel VAR of Jarociński (2010), benchmarked against
mean-field VI (ter Steege, 2024) and a Gibbs sampler.

## Structure

- `main.ipynb` — main entry-point notebook; runs the full study end to end:
  loads and prepares the real Jarociński data, runs the real-data pipeline
  (Gibbs, MFVI, SSVI-I, SSVI-C) and plots the results, then repeats the
  simulation study across four scenarios (low/high `T`, low/high `C`) and
  four seeds each, using the Gibbs posterior from the real-data fit as the
  data-generating parameters
- `pipeline.py` — orchestrates a full run: data prep, inference, evaluation
- `gibbs.py` — Gibbs sampler (ground truth benchmark)
- `ssvi_i.py`, `ssvi_c.py` — the two SSVI variants
- `mfvi.py` — mean-field VI baseline
- `results.py` — evaluation metrics (sample reconstruction, UQF, Faes accuracy, IRFs, coverage)
- `figures.py` — plotting functions for the above
- `simulate.py` — synthetic data generation for the simulation study
- `data_prep.py` — real-data loading and preprocessing
- `Jarocinski Data/` — replication data accompanying Jarociński (2010)

## Requirements

Python 3.x, NumPy, SciPy, pandas, matplotlib, arviz, joblib, IPython.

## Data

Real data is a subset of the replication files accompanying Jarociński (2010),
covering five Western European countries at monthly frequency, 1987–1998.

## Reproducibility

Stochastic components (Gibbs sampler, ULA sampling, posterior resampling)
accept a `seed` argument for deterministic reruns. Results are cached to disk
and reused unless explicitly recomputed.

## Notes on tuning

- SSVI-I and SSVI-C's ULA step for sampling $\lambda$ requires a per-method base
  learning rate (`s`), which must be tuned. This learnign rate scales the RMSProp-style
  adaptive step size.
  Values that work for the real dataset are not guaranteed to transfer to
  substantially different data scales (e.g. different C or T) without
  retuning — check the ESS and $\log(\lambda)$ trace diagnostics after any change
  to the data or model configuration. `main.ipynb` sets these per scenario
  (e.g. `s` and `epsilon` differ between the low-`T`, high-`T`, low-`C`,
  and high-`C` simulation runs).
- The Gibbs sampler's convergence (ESS, R-hat) should be checked whenever
  the data or priors change; the default chain length/burn-in were chosen
  for the specific dataset used in this project and are not universally
  sufficient.

## Notes on Lambda_c

`prep_data` accepts `Lambda=None`, in which case it reconstructs $\Lambda_c$ from the
data it is given. For simulated data, pass the true `Lambda_sim` (returned
by `simulate_data`) explicitly rather than leaving this as `None`.

## Caching

Pipeline results are cached to disk, keyed by config name, and reused unless
explicitly recomputed (`force_recompute=True`). Writes are atomic (temp file
+ rename) to avoid leaving a corrupted cache file if a run is interrupted.

## Author

Kevin O'Loughlin, supervised by Randolf Altmeyer.
