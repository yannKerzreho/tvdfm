# tvdfm — Time-Varying Dynamic Factor Model

A JAX/Equinox library for **Time-Varying Dynamic Factor Models (tvDFM)**
with a GRU (or Neural CDE) exposure module that lets factor loadings $\Lambda(t)$
vary over time.

In economics, DFM models are Linear Gaussian SSMs in which the state is represented by factors from various economic time series (initialized via PCA and trained using the EM algorithm). The goal of a DFM model is not to control or estimate the state, but to estimate a low-frequency measured variable using information about the state provided by other higher-frequency variables.

The idea behind this project is to add a layer of supervised learning specifically designed to estimate this variable of interest (in this case, GDP). The neural network learns a deviation in the measurement/weight matrix $\delta\Lambda(t)$ based on macroeconomic variables. An early stopping mechanism is implemented for the last four years of the dataset.

---

## Key Empirical Result

Benchmark: GDP nowcasting with FRED-MD (127 series, 2015–2025, h=1–6).

The default configuration selects the deviation
penalty $\lambda$ each year by validation GDP MSE — no manual tuning.  It achieves:

| Period | h=1 | h=2 | h=3 |
|---|---|---|---|
| Pre-COVID (2015–2019) | −12% vs DFM | −30% vs DFM | −17% vs DFM |
| **COVID (2020–2021) ★** | **−27%** | **−31%** | **−13%** |
| Post-COVID (2022–2025) | −3% | ≈0% | ≈0% |

`★` The COVID advantage is the headline finding: the model detects the
structural break in factor–GDP co-movement in real time, whereas the static
DFM is anchored to pre-COVID loadings and overshoots by 4–5× the normal error.

RMSE ratios (tvDFM / DFM, full 2015–2025):

```
h=1   0.741  (−26%)
h=2   0.700  (−30%)
h=3   0.871  (−13%)
h=4   1.122  (+12%, DFM wins at long horizons)
h=5   1.034
h=6   1.047
```

Model Confidence Set (Hansen, Lunde & Nason 2011, $\alpha = 10\%$, circular block
bootstrap):

| Horizon | Period | Result |
|---|---|---|
| h=2 | Full 2015–2025 | DFM eliminated (p=0.004) |
| h=2 | Pre-COVID | DFM eliminated (p=0.045) |
| h=1–3 grouped | Full 2015–2025 | DFM eliminated (p=0.017) |
| h=1–3 grouped | Pre-COVID | DFM eliminated (p=0.037) |
| h=4 | COVID | tvDFM eliminated (p=0.089, DFM wins) |
| h=4–6 grouped | All periods | Inconclusive |

The short-horizon advantage (h=1–3) is statistically significant across the
full sample and pre-COVID — not only during the structural break.  The COVID
gains (−27–31%) are the largest in magnitude but the 8-quarter window is too
short for formal discrimination.  DFM recovers at long horizons (h=4–6).

The selected $\lambda$ varies across years (0.01–10.0); the year-to-year pattern is
noisy but spans the full grid, confirming that no single fixed $\lambda$ is universally
optimal.

---

## Model

$$x_t = \Lambda(t)\, f_t + \varepsilon_t, \qquad \varepsilon_t \sim \mathcal{N}(0, R)$$

$$f_t = A\, f_{t-1} + \eta_t, \qquad \eta_t \sim \mathcal{N}(0, Q)$$

$$\Lambda(t) = \Lambda_\text{base} + \delta\Lambda(t)$$

$$\delta\Lambda(t) = \text{MLP}(h_t), \qquad h_t = \text{GRU}(x_{1:t})$$

$\Lambda_\text{base}$ is fixed from EM initialisation. The GRU hidden state $h_t$ is
updated causally at each time step; the MLP maps it to a perturbation
$\delta\Lambda(t)$ of the same shape as $\Lambda_\text{base}$.

Training jointly minimises:

$$\mathcal{L} = \text{MSE}_\text{target}(\hat{y},\, y) \;+\; \lambda \,\|\delta\Lambda\|_F^2$$

over multiple forecast horizons (horizon augmentation), with the
Kalman filter used as a differentiable decoder.

### ssm_fullEM_autolam (default, recommended)

The default configuration combines three components:

1. **SSM imputation** — an AR(1) SSM is fitted to each covariate series and
   used to impute unobserved values before the GRU pass, rather than
   forward-filling.
2. **Full-window EM** — the DFM EM is run on the complete training window
   (including the validation split), giving a better-calibrated anchor for
   the TV perturbation.
3. **Data-driven penalty** — for each expanding window the penalty weight $\lambda$ is
   selected from a candidate grid {0.01, 0.1, 1.0, 10.0} by validation GDP
   MSE.  No manual tuning is required; the selected $\lambda$ varies across years
   (range 0.01–10.0), adapting to regime stability.

---

## Installation

```bash
git clone https://github.com/yannKerzreho/tvdfm
cd tvdfm
pip install -e ".[statsmodels]"
```

**Requirements**: JAX ≥ 0.4.20, Equinox ≥ 0.11, Diffrax ≥ 0.5,
Optax ≥ 0.1.9, statsmodels ≥ 0.14, NumPy, Pandas, SciPy, scikit-learn.

### Data

Download the FRED-MD and FRED-QD files from [fred.stlouisfed.org/releases/rp392](https://research.stlouisfed.org/econ/mccracken/fred-databases/)
and place them in `experiment/data/` as:

```
experiment/data/
├── 2026-02-MD.csv      # FRED-MD monthly panel  (filename reflects vintage)
└── 2026-02-QD.csv      # FRED-QD quarterly panel (GDP)
```

> **Note**: FRED-MD/QD filenames encode the vintage date (e.g. `2026-02-MD.csv`).
> When you download a newer vintage the filenames will differ; pass the correct
> path via `--data_dir` or update the default in `main.py`.

---

## Validation experiment

`main.py` runs the full expanding-window OOS benchmark from the command line.

### Quick start — reproduce the main result

```bash
python main.py
```

This runs DFM (full-history EM) vs **ssm_fullEM_autolam** over 2015–2025
at h=1–6 and prints the RMSE table, COVID callout, selected-penalty trajectory,
and MCS results.

### Common options

```bash
# Change the OOS window
python main.py --test_start 2010 --test_end 2025

# Expand the penalty candidate grid
python main.py --lam_candidates 0.001 0.01 0.1 1.0 10.0 100.0

# Fixed-penalty variant (skip per-year search, faster)
python main.py --model ssm_fullEM --lam 1.0

# GDP-only TV loading (perturbation applied to the target row only)
python main.py --model ssm_fullEM_target --lam 0.01

# Short-horizon focus, quiet training output, save to custom directory
python main.py --horizons 1 2 3 --quiet --output_dir results/short_hz

# Compare against strict-training DFM (holds out last 48 months from EM)
python main.py --baseline train

# Skip MCS (faster, useful for prototyping)
python main.py --no_mcs
```

### Full argument reference

```
Data
  --data_dir DIR        Path to FRED-MD CSV files (default: experiment/data/)
  --data_start DATE     Start of training history  (default: 1990-01-01)
  --test_start YYYY     First OOS test year        (default: 2015)
  --test_end   YYYY     Last OOS test year         (default: 2025)
  --horizons H [H ...]  Forecast horizons in months (default: 1 2 3 4 5 6)

Models
  --baseline {full,train}
      full  — DFM EM trained on the entire window (val_n=0)
      train — DFM EM trained on strict training set (val_n=48)
  --model {ssm_fullEM_autolam,ssm_fullEM,ssm_fullEM_target,short}
      ssm_fullEM_autolam— GRU + SSM imputation + full-history EM + auto penalty (default)
      ssm_fullEM        — Same, fixed penalty (set via --lam)
      ssm_fullEM_target — Same, TV loading restricted to the GDP row only
      short             — GRU + forward-fill imputation (faster)
  --lam_candidates L [L ...]
                        Candidate penalty values for ssm_fullEM_autolam
                        (default: 0.01 0.1 1.0 10.0)
  --lam L               Deviation penalty weight (default: 1.0)

Architecture
  --n_factors N         Latent factors           (default: 3)
  --n_select  N         Preselected covariates   (default: 20)
  --hidden_size N       GRU hidden size          (default: 16)
  --mlp_width N         MLP readout width        (default: 64)
  --mlp_depth N         MLP readout depth        (default: 2)

Training
  --val_n N             Validation window months (default: 48)
  --n_epochs N          Max GRU epochs           (default: 200)
  --patience N          Early-stopping patience  (default: 7)
  --seed N              Random seed              (default: 42)

Output
  --output_dir DIR      Save CSV results here
  --no_mcs              Skip MCS tests
  --quiet               Suppress per-epoch output
```

### Output files

```
results/tvdfm_<timestamp>/
├── eval_raw.csv          per-quarter predictions (all configs, all horizons)
├── rmse_overall.csv      RMSE by config × horizon
├── rmse_subperiod.csv    RMSE by config × horizon × period
├── ratio_to_baseline.csv RMSE ratios (tvDFM / DFM)
└── selected_lam.csv      penalty chosen per year (ssm_fullEM_autolam only)
```

---

## Library API

### Static DFM (no time variation)

```python
from tvdfm import TVDFModel

model = TVDFModel(n_factors=3, exposure=None, dfm_init="em")
model.fit(df_obs)               # df_obs: pd.DataFrame [T, N]
predictions = model.predict()   # np.ndarray [T, N]
factors     = model.transform() # np.ndarray [T, K]  (Kalman-filtered)
```

### tvDFM with GRU exposure + data-driven penalty (recommended)

Train one model per penalty candidate and keep the one with the lowest validation
GDP MSE — mirroring what `main.py` does per expanding window:

```python
LAM_CANDIDATES = [0.01, 0.1, 1.0, 10.0]

best_model, best_lam, best_val = None, None, float("inf")
for lam in LAM_CANDIDATES:
    model = TVDFModel(
        n_factors         = 3,
        exposure          = "gru",
        tv_Lambda         = True,
        hidden_size       = 16,
        imputation        = "ssm",      # AR(1) SSM imputation
        em_val_n          = 0,          # EM on full training window
        lambda_dev_weight = lam,
        dfm_init          = "preselected",
        target_col        = "GDPC1",
        n_select          = 20,
        val_n             = 48,
        val_full_kf       = True,
    )
    model.fit(df_obs, target_series="GDPC1")
    val = model.manager_.best_val_loss_
    if val < best_val:
        best_val, best_lam, best_model = val, lam, model

predictions = best_model.predict(df_obs_with_future_nans)
```

### tvDFM with Neural CDE exposure

```python
model = TVDFModel(
    n_factors     = 3,
    exposure      = "ncde",
    interpolation = "cubic",    # "cubic" | "linear" | "rectilinear"
    hidden_size   = 16,
    tv_Lambda     = True,
)
model.fit(df_obs)
```

### Target-specific TV loading (perturbation on one series only)

```python
# Restrict time-varying perturbation to the GDP loading row
model = TVDFModel(
    n_factors      = 3,
    exposure       = "gru",
    lambda_indices = (0,),   # index of the target series in df_obs
)
```

---

## Architecture

```
tvdfm/
├── __init__.py          Public API
├── model.py             TVDFModel — sklearn-compatible high-level wrapper
├── core.py              TVDFM     — Equinox module (JAX model)
├── ssm.py               DFMStateSpace — Kalman filter + RTS smoother (JAX/scan)
├── training.py          LTVTrainingManager, loss_e2e, JIT-compiled step
├── utils.py             Data helpers, statsmodels parameter extractor
└── exposure/
    ├── base.py          AbstractExposure, make_ncde_path
    ├── ncde.py          NCDEExposure (Diffrax)
    └── rnn.py           GRUExposure  (Equinox)
```

| Design decision | Solution |
|---|---|
| Mixed-frequency NCDE | `make_ncde_path` builds the interpolation from observed timestamps; no interior NaN fill inside JAX |
| Penalty without recompilation | `lambda_dev_weight`, `wd_exposure`, `wd_ssm`, `kf_ll_weight` are dynamic JAX scalars — single JIT kernel reused |
| Spectral radius constraint | 3-step power iteration $O(K^2)$ instead of SVD; applied to $A_\text{base} + \delta A$ |
| DFM anchor | $\Lambda_\text{base}$ is a frozen array; $\delta\Lambda$ and the GRU are the only trainable parameters |
| Missing observations | Kalman gain zeroed for NaN rows; log-likelihood summed only over observed entries |
| EM degeneracy guard | After statsmodels EM: $\Lambda$ row-norms clipped to 200; $R$ diagonal floored at $10^{-3}$ |
| KF numerical stability | Non-finite Kalman gain zeroed; $P_f$ falls back to $P_p$ (not $I$) |

---

## License

MIT
