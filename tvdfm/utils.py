"""
Utility functions for the TVDFM library.

Data helpers
------------
:func:`to_float_times`         Convert DatetimeIndex / PeriodIndex to float64.
:func:`parse_dataframe`        Extract (array, times, columns) from a DataFrame or ndarray.
:func:`parse_covariates`       Coerce covariates to a float32 numpy array.

Statsmodels helpers
-------------------
:func:`extract_statsmodels_params`
    Extract (Lambda, A, Q, R) from a fitted statsmodels DynamicFactor result
    and map them onto an arbitrary set of data columns.

Internal helpers
----------------
:func:`_spectral_norm`   Spectral-radius normalisation for the A matrix.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import jax.numpy as jnp


# ---------------------------------------------------------------------------
# Time conversion
# ---------------------------------------------------------------------------

def to_float_times(index) -> np.ndarray:
    """
    Convert a time index to a float64 numpy array.

    Supported inputs
    ----------------
    ``pd.PeriodIndex``      → converted to monthly timestamps first.
    ``pd.DatetimeIndex``    → fractional years (2020-07 → 2020.5).
    Any array-like          → cast to float64 directly.

    Examples
    --------
    >>> to_float_times(pd.date_range("2020-01", periods=3, freq="ME"))
    array([2020.    , 2020.0833, 2020.1667])
    """
    if isinstance(index, pd.PeriodIndex):
        index = index.to_timestamp()
    if isinstance(index, pd.DatetimeIndex):
        return (index.year + (index.month - 1) / 12.0).to_numpy(dtype=np.float64)
    return np.asarray(index, dtype=np.float64)


# ---------------------------------------------------------------------------
# DataFrame / array parsing
# ---------------------------------------------------------------------------

def parse_dataframe(
    X,
    times=None,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Coerce ``X`` to a float32 numpy array and extract times and column names.

    Parameters
    ----------
    X : pd.DataFrame or array-like [T, N]
        Observation matrix.  NaN marks missing values.
    times : array-like [T], optional
        Observation times.  When ``X`` is a DataFrame with a
        ``DatetimeIndex`` / ``PeriodIndex``, times are inferred from the index
        if this argument is omitted.  Falls back to ``np.arange(T)`` otherwise.

    Returns
    -------
    X_np : ndarray [T, N] float32
    times_np : ndarray [T] float64
    columns : list of str
    """
    if isinstance(X, pd.DataFrame):
        cols = list(X.columns)
        if times is None:
            if isinstance(X.index, (pd.DatetimeIndex, pd.PeriodIndex)):
                times_np = to_float_times(X.index)
            else:
                times_np = np.arange(len(X), dtype=np.float64)
        else:
            times_np = to_float_times(times)
        return X.to_numpy(dtype=np.float32), times_np, cols

    X_np = np.asarray(X, dtype=np.float32)
    times_np = (
        to_float_times(times) if times is not None
        else np.arange(X_np.shape[0], dtype=np.float64)
    )
    return X_np, times_np, [str(i) for i in range(X_np.shape[1])]


def parse_covariates(covariates) -> Optional[np.ndarray]:
    """
    Coerce ``covariates`` to a float32 numpy array, or return None.

    Parameters
    ----------
    covariates : pd.DataFrame, array-like [T, C], or None

    Returns
    -------
    ndarray [T, C] float32 or None
    """
    if covariates is None:
        return None
    if isinstance(covariates, pd.DataFrame):
        return covariates.to_numpy(dtype=np.float32)
    return np.asarray(covariates, dtype=np.float32)


def _spectral_norm(A: jnp.ndarray, bound: float) -> jnp.ndarray:
    """
    Scale A so its largest singular value σ_max(A) ≤ ``bound``.

    Uses 3-step power iteration to estimate σ_max — O(K²·3), no SVD.
    Only rescales when σ_max exceeds the bound.

    Design note — singular value vs spectral radius
    ------------------------------------------------
    VAR stationarity requires the spectral radius ρ(A) ≤ bound, not σ_max(A).
    Since ρ(A) ≤ σ_max(A), constraining σ_max is *sufficient* but not tight:
    for non-normal A (large off-diagonal entries) this can shrink A more than
    necessary and may under-estimate factor persistence.  The tradeoff is
    simplicity and differentiability — computing ρ(A) exactly requires an
    eigendecomposition and is not smoothly differentiable everywhere.
    In practice the gap is small for near-diagonal estimated factor matrices.
    """
    v = jnp.ones(A.shape[0])
    for _ in range(3):
        v = A.T @ (A @ v)
        v = v / (jnp.linalg.norm(v) + 1e-8)
    sigma = jnp.sqrt(jnp.dot(v, A.T @ (A @ v)) + 1e-8)
    return A / jnp.maximum(sigma / bound, 1.0)


# ---------------------------------------------------------------------------
# Statsmodels parameter extraction
# ---------------------------------------------------------------------------

def extract_statsmodels_params(
    result,
    data_columns: List[str],
    n_factors: int,
    dfm_columns: Optional[List[str]] = None,
    factor_order: int = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract ``(Lambda, A, Q, R)`` from a fitted statsmodels DynamicFactor result.

    Parameters
    ----------
    result : statsmodels DynamicFactorResults
        Any result object exposing ``result.model.ssm``.
    data_columns : list of str
        Column names of the **full** dataset (defines N = len(data_columns)).
    n_factors : int
        Number of factors K.
    dfm_columns : list of str or None
        Column names used during DFM fitting.  When different from
        ``data_columns``, Λ and R are zero-padded for excluded series.
    factor_order : int
        VAR lag order p for the factor process (default 1).

    Returns
    -------
    Lambda : [N, K]     float32
    A      : [K, K·p]   float32  — all p lag matrices in one array
    Q      : [K, K]     float32
    R      : [N]        float32  — diagonal of observation covariance
    """
    ssm      = result.model.ssm
    n_series = len(data_columns)
    K, p     = n_factors, factor_order
    Kp       = K * p

    # Loadings [N_dfm, K]
    design     = np.array(ssm["design"])
    Lambda_dfm = design[:, :K]

    # Companion transition [Kp, Kp] → first-block-row [K, Kp] = all p lags
    trans_full = np.array(ssm["transition"])[:Kp, :Kp]
    A          = trans_full[:K, :Kp].astype(np.float32)          # [K, K·p]

    # Process noise [K, K]
    Q = np.array(ssm["state_cov"])[:K, :K].astype(np.float32)

    # Observation noise diagonal [N_dfm]
    obs_cov = np.array(ssm["obs_cov"])
    R_dfm   = np.diag(obs_cov) if obs_cov.ndim == 2 else obs_cov
    n_dfm   = Lambda_dfm.shape[0]
    R_dfm   = R_dfm[:n_dfm]

    # Map onto full data_columns (zero-pad excluded series)
    dfm_cols = list(dfm_columns) if dfm_columns is not None else list(data_columns)
    if dfm_cols == list(data_columns):
        Lambda = Lambda_dfm[:n_series]
        R      = R_dfm[:n_series]
    else:
        col_to_idx = {c: i for i, c in enumerate(data_columns)}
        Lambda = np.zeros((n_series, K), dtype=np.float32)
        R      = np.full(n_series, 0.1, dtype=np.float32)
        for i, col in enumerate(dfm_cols):
            if col in col_to_idx:
                Lambda[col_to_idx[col]] = Lambda_dfm[i]
                R[col_to_idx[col]]      = R_dfm[i]

    return (
        Lambda.astype(np.float32),
        A,
        Q,
        R.astype(np.float32),
    )
