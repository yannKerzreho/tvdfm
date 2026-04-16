"""
Variable preselection for the TVDFM.

Statsmodels DFM EM is O(N³) per iteration and the NCDE vector field has
O(hidden_size × N_covariates) parameters, so fitting directly on a
200-variable panel is impractical.  This module provides two strategies
for reducing the input dimension to a manageable N_select ≈ 20 before
handing off to :func:`~tvdfm.core.TVDFM.fit_and_init`.

Functions
---------
select_top_corr
    Rank variables by |Pearson correlation| with the target (fast baseline).
select_iterative_pca
    Greedy forward selection: at each step add the variable that most
    improves the OLS R² of regressing the target on PCA factors extracted
    from the current set.
preselect
    Dispatcher that calls one of the above and returns the filtered panel.
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from typing import List, Optional, Tuple


# ---------------------------------------------------------------------------
# select_top_corr
# ---------------------------------------------------------------------------

def select_top_corr(
    X: pd.DataFrame,
    y: pd.Series,
    n_select: int,
    *,
    lag_x: int = 1,
    exclude: Optional[List[str]] = None,
) -> List[str]:
    """
    Select the ``n_select`` columns of ``X`` most correlated with ``y``.

    Parameters
    ----------
    X : pd.DataFrame [T, N]
        Covariate panel.  May contain NaNs (mixed frequency).
    y : pd.Series [T]
        Target series.  May contain NaNs (quarterly).
    n_select : int
        Number of variables to return.
    lag_x : int
        Shift ``X`` by this many periods before computing correlations
        (pseudo real-time: X known with a ``lag_x``-month lag).
    exclude : list of str or None
        Column names to always exclude (e.g. the target itself).

    Returns
    -------
    list of str
        ``n_select`` column names sorted by |correlation| with ``y``,
        descending.  Variables with fewer than 20 valid overlapping
        observations receive correlation 0.
    """
    exclude_set = set(exclude or [])
    X_lagged = X.shift(lag_x)

    correlations: dict[str, float] = {}
    for col in X.columns:
        if col in exclude_set:
            continue
        xi = X_lagged[col]
        valid = y.notna() & xi.notna()
        if valid.sum() < 20:
            correlations[col] = 0.0
        else:
            correlations[col] = float(
                np.corrcoef(y[valid].values, xi[valid].values)[0, 1]
            )

    ranked = sorted(correlations, key=lambda c: abs(correlations[c]), reverse=True)
    return ranked[:n_select]


# ---------------------------------------------------------------------------
# select_iterative_pca
# ---------------------------------------------------------------------------

def select_iterative_pca(
    X: pd.DataFrame,
    y: pd.Series,
    n_factors: int,
    n_select: int,
    *,
    lag_x: int = 1,
    exclude: Optional[List[str]] = None,
    max_iter: Optional[int] = None,
) -> List[str]:
    """
    Greedy forward selection based on PCA factors + OLS regression.

    At each step, the candidate that most improves the OLS R² of
    regressing ``y`` on PCA factors extracted from the current selected
    set is added.  The selection is seeded by the variable with the
    highest |correlation| with ``y``.

    Parameters
    ----------
    X : pd.DataFrame [T, N]
        Covariate panel.  May contain NaNs.
    y : pd.Series [T]
        Target series.  May contain NaNs (quarterly).
    n_factors : int
        Number of PCA components to extract at each trial step.
    n_select : int
        Final number of selected variables.
    lag_x : int
        Lag applied to ``X`` before correlations and PCA.
    exclude : list of str or None
        Columns to always exclude.
    max_iter : int or None
        Cap on the inner loop iterations (useful for very large N);
        defaults to ``len(X.columns)``.

    Returns
    -------
    list of str  (length ``n_select`` unless data is insufficient)
    """
    exclude_set = set(exclude or [])
    X_lagged   = X.shift(lag_x)
    y_clean    = y.dropna()
    candidates = [c for c in X.columns if c not in exclude_set]

    if not candidates:
        return []

    # ── Seed: variable with highest |correlation| with y ───────────────
    seed_corrs: dict[str, float] = {}
    for col in candidates:
        xi    = X_lagged[col]
        valid = y.notna() & xi.notna()
        seed_corrs[col] = (
            abs(float(np.corrcoef(y[valid].values, xi[valid].values)[0, 1]))
            if valid.sum() >= 20 else 0.0
        )
    selected = [max(seed_corrs, key=lambda c: seed_corrs[c])]
    candidates = [c for c in candidates if c != selected[0]]

    _max_iter = max_iter if max_iter is not None else len(X.columns)

    # ── Greedy forward loop ─────────────────────────────────────────────
    for _ in range(min(n_select - 1, _max_iter)):
        if not candidates:
            break

        best_r2  = -float("inf")
        best_var = None

        for xi_col in candidates:
            trial_set = selected + [xi_col]
            n_comp    = min(n_factors, len(trial_set))

            # Rows where ALL trial vars and y are non-NaN
            X_trial_raw = X_lagged[trial_set]
            common = (
                X_trial_raw.dropna(how="any").index.intersection(y_clean.index)
            )
            if len(common) < n_comp + 5:
                continue

            X_trial = X_trial_raw.loc[common]
            y_trial = y_clean.loc[common]

            # Standardise (critical for mixed-scale panels)
            std = X_trial.std().clip(lower=1e-8)
            X_std = (X_trial - X_trial.mean()) / std

            pca     = PCA(n_components=n_comp)
            factors = pca.fit_transform(X_std.to_numpy())   # [T_valid, n_comp]

            reg = LinearRegression(fit_intercept=True).fit(factors, y_trial.to_numpy())
            r2  = reg.score(factors, y_trial.to_numpy())

            if r2 > best_r2:
                best_r2  = r2
                best_var = xi_col

        if best_var is None:
            break
        selected.append(best_var)
        candidates = [c for c in candidates if c != best_var]

    return selected


# ---------------------------------------------------------------------------
# preselect  (dispatcher)
# ---------------------------------------------------------------------------

def preselect(
    X: pd.DataFrame,
    y: pd.Series,
    n_factors: int,
    n_select: int,
    *,
    method: str = "iterative_pca",
    lag_x: int = 1,
    exclude: Optional[List[str]] = None,
) -> Tuple[List[str], pd.DataFrame]:
    """
    Select ``n_select`` columns from ``X`` most informative for ``y``.

    Parameters
    ----------
    X : pd.DataFrame [T, N]
        Full covariate panel.  NaN structure is preserved in the output.
    y : pd.Series [T]
        Target series (NaNs allowed; never filled internally).
    n_factors : int
        Number of PCA components (used only by ``iterative_pca``).
    n_select : int
        Number of variables to select.
    method : str
        ``'iterative_pca'`` (default, recommended) or ``'top_corr'``.
    lag_x : int
        Lag applied to ``X`` before measuring association with ``y``.
    exclude : list of str or None
        Columns to always exclude from selection.

    Returns
    -------
    selected_columns : list of str
    X_selected : pd.DataFrame
        ``X[selected_columns]`` — original index and NaN structure preserved.

    Raises
    ------
    ValueError
        If ``method`` is not recognised.
    """
    if method == "top_corr":
        selected = select_top_corr(
            X, y, n_select, lag_x=lag_x, exclude=exclude,
        )
    elif method == "iterative_pca":
        selected = select_iterative_pca(
            X, y, n_factors, n_select, lag_x=lag_x, exclude=exclude,
        )
    else:
        raise ValueError(
            f"Unknown preselection method {method!r}. "
            "Choose 'top_corr' or 'iterative_pca'."
        )

    print(f"[preselection] {method}: selected {len(selected)} variables from {X.shape[1]}")
    return selected, X[selected]
