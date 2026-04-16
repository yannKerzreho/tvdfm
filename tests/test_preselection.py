"""
Tests for tvdfm.preselection.

All tests use synthetic data (T=300, N=50) with a quarterly target
(NaN on non-quarter-end months) to exercise the mixed-frequency NaN
handling that is the main correctness concern.
"""

import numpy as np
import pandas as pd
import pytest

from tvdfm.preselection import select_top_corr, select_iterative_pca, preselect


# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------

def make_data(T: int = 300, N: int = 50, seed: int = 0):
    """Monthly covariate panel + quarterly target (NaN on non-quarter-end)."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        rng.standard_normal((T, N)),
        columns=[f"x{i}" for i in range(N)],
    )
    y = pd.Series(rng.standard_normal(T), name="GDP")
    # Simulate quarterly: NaN on the first two months of every quarter
    for i in range(T):
        if i % 3 != 2:
            y.iloc[i] = np.nan
    return X, y


# ---------------------------------------------------------------------------
# select_top_corr
# ---------------------------------------------------------------------------

class TestSelectTopCorr:
    def test_returns_correct_count(self):
        X, y = make_data()
        sel = select_top_corr(X, y, n_select=10)
        assert len(sel) == 10

    def test_no_duplicates(self):
        X, y = make_data()
        sel = select_top_corr(X, y, n_select=10)
        assert len(set(sel)) == 10

    def test_all_columns_are_valid(self):
        X, y = make_data()
        sel = select_top_corr(X, y, n_select=10)
        assert all(c in X.columns for c in sel)

    def test_excludes_specified_columns(self):
        X, y = make_data()
        sel = select_top_corr(X, y, n_select=10, exclude=["x0", "x1"])
        assert "x0" not in sel
        assert "x1" not in sel

    def test_low_overlap_variable_deprioritised(self):
        X, y = make_data(seed=1)
        # x0: only 1 valid row → below the 20-observation threshold
        X = X.copy()
        X["x0"] = np.nan
        X.iloc[0, X.columns.get_loc("x0")] = 1.0
        sel = select_top_corr(X, y, n_select=5)
        assert "x0" not in sel, "Variable with <20 valid rows should not be selected"

    def test_lag_shifts_x(self):
        """With lag_x=1, X should be shifted; result may differ from lag_x=0."""
        X, y = make_data(seed=2)
        sel0 = select_top_corr(X, y, n_select=10, lag_x=0)
        sel1 = select_top_corr(X, y, n_select=10, lag_x=1)
        # Both are valid — just confirm they run and have the right length
        assert len(sel0) == 10
        assert len(sel1) == 10

    def test_n_select_larger_than_candidates(self):
        """When n_select > valid candidates, return as many as possible."""
        X, y = make_data(N=5)
        sel = select_top_corr(X, y, n_select=10)
        # Only 5 columns available
        assert len(sel) <= 5


# ---------------------------------------------------------------------------
# select_iterative_pca
# ---------------------------------------------------------------------------

class TestSelectIterativePCA:
    def test_returns_correct_count(self):
        X, y = make_data()
        sel = select_iterative_pca(X, y, n_factors=3, n_select=10)
        assert len(sel) == 10

    def test_no_duplicates(self):
        X, y = make_data()
        sel = select_iterative_pca(X, y, n_factors=3, n_select=10)
        assert len(set(sel)) == 10

    def test_all_columns_are_valid(self):
        X, y = make_data()
        sel = select_iterative_pca(X, y, n_factors=3, n_select=10)
        assert all(c in X.columns for c in sel)

    def test_excludes_specified_columns(self):
        X, y = make_data()
        sel = select_iterative_pca(X, y, n_factors=3, n_select=10,
                                   exclude=["x0", "x1"])
        assert "x0" not in sel
        assert "x1" not in sel

    def test_performance_n128_n_select20(self):
        """Must complete in under 10 seconds for N=128, n_select=20."""
        import time
        X, y = make_data(T=300, N=128, seed=5)
        t0  = time.perf_counter()
        sel = select_iterative_pca(X, y, n_factors=3, n_select=20)
        elapsed = time.perf_counter() - t0
        assert len(sel) == 20
        assert elapsed < 10.0, f"select_iterative_pca took {elapsed:.1f}s (limit 10s)"


# ---------------------------------------------------------------------------
# preselect dispatcher
# ---------------------------------------------------------------------------

class TestPreselect:
    def test_top_corr_dispatch(self):
        X, y = make_data()
        cols, X_sel = preselect(X, y, 3, 10, method="top_corr")
        assert len(cols) == 10
        assert X_sel.shape[1] == 10
        assert list(X_sel.columns) == cols

    def test_iterative_pca_dispatch(self):
        X, y = make_data()
        cols, X_sel = preselect(X, y, 3, 10, method="iterative_pca")
        assert len(cols) == 10
        assert X_sel.shape[1] == 10
        assert list(X_sel.columns) == cols

    def test_unknown_method_raises(self):
        X, y = make_data()
        with pytest.raises(ValueError, match="Unknown preselection method"):
            preselect(X, y, 3, 10, method="magic")

    def test_nan_structure_preserved(self):
        """Output must preserve original NaN structure — no filling."""
        X, y = make_data()
        # Introduce some NaNs in the panel
        X_nan = X.copy()
        X_nan.iloc[::4, :5] = np.nan

        _, X_sel = preselect(X_nan, y, 3, 10, method="top_corr")
        # For every selected column, NaN positions must match the original
        for col in X_sel.columns:
            orig_nan = X_nan[col].isna()
            sel_nan  = X_sel[col].isna()
            assert orig_nan.equals(sel_nan), (
                f"NaN structure changed for column {col}"
            )

    def test_index_preserved(self):
        """Output DataFrame must share the original DatetimeIndex."""
        T = 100
        idx = pd.date_range("2010-01", periods=T, freq="ME")
        X = pd.DataFrame(
            np.random.default_rng(0).standard_normal((T, 30)),
            index=idx,
            columns=[f"x{i}" for i in range(30)],
        )
        y = pd.Series(np.random.default_rng(0).standard_normal(T),
                      index=idx, name="GDP")
        for i in range(T):
            if i % 3 != 2:
                y.iloc[i] = np.nan

        _, X_sel = preselect(X, y, 2, 5, method="top_corr")
        assert X_sel.index.equals(idx)

    def test_prints_summary(self, capsys):
        X, y = make_data(N=20)
        preselect(X, y, 2, 5, method="top_corr")
        captured = capsys.readouterr()
        assert "[preselection]" in captured.out
        assert "top_corr" in captured.out
        assert "20" in captured.out   # total variable count
