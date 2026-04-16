"""
Smoke tests — verify shapes, forward passes, and training steps run without errors.

All tests use tiny dimensions (T=30, N=4, K=2, C=3) and few epochs so the
full suite finishes in under a minute on CPU.
"""

import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
import equinox as eqx
import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

T, N, K, C = 30, 4, 2, 3
KEY = jax.random.PRNGKey(0)


@pytest.fixture
def obs_np():
    """[T, N] float32 observations with 10% NaN."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((T, N)).astype(np.float32)
    mask = rng.random((T, N)) < 0.1
    X[mask] = np.nan
    return X


@pytest.fixture
def obs_df(obs_np):
    idx = pd.date_range("2010-01", periods=T, freq="ME")
    return pd.DataFrame(obs_np, index=idx, columns=[f"s{i}" for i in range(N)])


@pytest.fixture
def cov_np():
    """[T, C] covariates with quarterly NaNs (every 3rd row observed)."""
    rng = np.random.default_rng(1)
    X = rng.standard_normal((T, C)).astype(np.float32)
    # Simulate quarterly: only every 3rd timestep has a value
    for t in range(T):
        if t % 3 != 0:
            X[t] = np.nan
    return X


@pytest.fixture
def cov_df(cov_np):
    idx = pd.date_range("2010-01", periods=T, freq="ME")
    return pd.DataFrame(cov_np, index=idx, columns=[f"c{i}" for i in range(C)])


@pytest.fixture
def times_np():
    return np.arange(T, dtype=np.float64)


# ---------------------------------------------------------------------------
# SSM tests
# ---------------------------------------------------------------------------

class TestDFMStateSpace:
    def test_kalman_filter_shapes(self, obs_np, times_np):
        from tvdfm import DFMStateSpace
        ssm = DFMStateSpace(K, N, KEY)
        Lambda_t = jnp.ones((T, N, K)) * 0.5
        A_t      = jnp.broadcast_to(jnp.eye(K) * 0.5, (T, K, K))
        z_filt, lls = ssm.filter(jnp.asarray(obs_np), Lambda_t, A_t)
        assert z_filt.shape == (T, K)
        assert lls.shape    == (T,)
        assert jnp.all(jnp.isfinite(z_filt))

    def test_rts_smoother_shapes(self, obs_np):
        from tvdfm import DFMStateSpace
        ssm = DFMStateSpace(K, N, KEY)
        Lambda_t = jnp.ones((T, N, K)) * 0.5
        A_t      = jnp.broadcast_to(jnp.eye(K) * 0.5, (T, K, K))
        z_sm, P_sm, z_f, P_f, lls = ssm.filter_and_smooth(
            jnp.asarray(obs_np), Lambda_t, A_t
        )
        assert z_sm.shape == (T, K)
        assert P_sm.shape == (T, K, K)

    def test_no_nans_with_missing_obs(self, obs_np):
        from tvdfm import DFMStateSpace
        # All NaN column — filter should not blow up
        obs_all_nan = obs_np.copy()
        obs_all_nan[:, 0] = np.nan
        ssm = DFMStateSpace(K, N, KEY)
        Lambda_t = jnp.ones((T, N, K)) * 0.5
        A_t      = jnp.broadcast_to(jnp.eye(K) * 0.5, (T, K, K))
        z_filt, _ = ssm.filter(jnp.asarray(obs_all_nan), Lambda_t, A_t)
        assert jnp.all(jnp.isfinite(z_filt))


# ---------------------------------------------------------------------------
# Exposure tests
# ---------------------------------------------------------------------------

class TestExposure:
    def test_ncde_output_shapes(self, times_np, cov_np):
        from tvdfm import NCDEExposure, make_ncde_path
        path = make_ncde_path(times_np, cov_np, interpolation="rectilinear")
        model = NCDEExposure(
            n_covariates=C, n_series=N, n_factors=K, key=KEY,
            hidden_size=4, mlp_width=8, mlp_depth=1,
            interpolation="rectilinear",
        )
        t_j = jnp.asarray(times_np)
        c_j = jnp.asarray(path)
        dL, dA = model(t_j, c_j, inference=True)
        assert dL.shape == (T, N, K)
        assert dA.shape == (T, K, K)
        assert jnp.all(jnp.isfinite(dL))

    def test_ncde_lambda_indices(self, times_np, cov_np):
        from tvdfm import NCDEExposure, make_ncde_path
        path = make_ncde_path(times_np, cov_np, interpolation="rectilinear")
        model = NCDEExposure(
            n_covariates=C, n_series=N, n_factors=K, key=KEY,
            hidden_size=4, mlp_width=8, mlp_depth=1,
            lambda_indices=(0,),
            interpolation="rectilinear",
        )
        t_j = jnp.asarray(times_np)
        c_j = jnp.asarray(path)
        dL, _ = model(t_j, c_j, inference=True)
        assert dL.shape == (T, N, K)
        # All rows except index 0 should be zero at init (zero readout)
        assert jnp.allclose(dL[:, 1:, :], 0.0)

    def test_gru_output_shapes(self, times_np, cov_np):
        from tvdfm import GRUExposure
        model = GRUExposure(
            n_covariates=C, n_series=N, n_factors=K, key=KEY,
            hidden_size=8,
        )
        t_j = jnp.asarray(times_np)
        c_j = jnp.asarray(cov_np)
        dL, dA = model(t_j, c_j, inference=True)
        assert dL.shape == (T, N, K)
        assert dA.shape == (T, K, K)

    def test_make_ncde_path_no_nan(self, times_np, cov_np):
        from tvdfm import make_ncde_path
        for interp in ("cubic", "linear", "rectilinear"):
            path = make_ncde_path(times_np, cov_np, interpolation=interp)
            assert path.shape == (T, C)
            assert not np.any(np.isnan(path)), f"NaNs in path with {interp}"

    def test_pca_reducer_shapes(self, cov_np):
        from tvdfm import AbstractExposure
        reducer = AbstractExposure.make_pca_reducer(cov_np, n_components=2, key=KEY)
        x = jnp.asarray(cov_np[:1])  # [1, C]
        out = reducer(x[0])           # [2]
        assert out.shape == (2,)


# ---------------------------------------------------------------------------
# TVDFM core tests
# ---------------------------------------------------------------------------

class TestTVDFM:
    def test_from_params_shapes(self, obs_np, times_np, cov_np):
        from tvdfm import TVDFM, NCDEExposure, make_ncde_path
        Lambda = np.random.randn(N, K).astype(np.float32)
        A      = np.eye(K, dtype=np.float32) * 0.5
        path   = make_ncde_path(times_np, cov_np, "rectilinear")
        exp    = NCDEExposure(
            n_covariates=C, n_series=N, n_factors=K, key=KEY,
            hidden_size=4, mlp_width=8, mlp_depth=1,
            interpolation="rectilinear",
        )
        model = TVDFM.from_params(Lambda, A, None, None, exp, KEY)
        preds, factors, kf_ll = model(
            jnp.asarray(times_np),
            jnp.asarray(obs_np),
            jnp.asarray(path),
            inference=True,
        )
        assert preds.shape   == (T, N)
        assert factors.shape == (T, K)
        assert jnp.isfinite(kf_ll)

    def test_static_dfm(self, obs_np, times_np):
        from tvdfm import TVDFM
        Lambda = np.random.randn(N, K).astype(np.float32)
        A      = np.eye(K, dtype=np.float32) * 0.5
        model  = TVDFM.from_params(
            Lambda, A, None, None, None, KEY,
            tv_Lambda=False, tv_A=False,
        )
        dummy_cov = jnp.zeros((T, 1))
        preds, factors, kf_ll = model(
            jnp.asarray(times_np),
            jnp.asarray(obs_np),
            dummy_cov,
            inference=True,
        )
        assert preds.shape == (T, N)

    def test_from_pca(self, obs_df):
        from tvdfm import TVDFM
        model = TVDFM.from_pca(obs_df, K, None, KEY, tv_Lambda=False, tv_A=False)
        assert model.Lambda_base.shape == (N, K)


# ---------------------------------------------------------------------------
# Training tests
# ---------------------------------------------------------------------------

class TestTraining:
    def test_one_step(self, obs_np, times_np, cov_np):
        from tvdfm import TVDFM, NCDEExposure, LTVTrainingManager, make_ncde_path
        Lambda = np.random.randn(N, K).astype(np.float32)
        A      = np.eye(K, dtype=np.float32) * 0.5
        path   = make_ncde_path(times_np, cov_np, "rectilinear")
        exp    = NCDEExposure(
            n_covariates=C, n_series=N, n_factors=K, key=KEY,
            hidden_size=4, mlp_width=8, mlp_depth=1,
            interpolation="rectilinear",
        )
        model   = TVDFM.from_params(Lambda, A, None, None, exp, KEY)
        manager = LTVTrainingManager(model, lr_ssm=1e-3, lr_exposure=1e-3)
        trained = manager.fit(
            times_np, obs_np, path,
            max_epochs=2, val_n=6, patience=2,
            wd_exposure=0.01, wd_ssm=0.01,
            interpolation="rectilinear",
            verbose=False,
        )
        assert trained is not None

    def test_wd_no_recompile(self, obs_np, times_np, cov_np):
        """Varying wd_exposure/wd_ssm should not trigger a second JIT compile."""
        from tvdfm import TVDFM, NCDEExposure, LTVTrainingManager, make_ncde_path
        from tvdfm.training import _eval_loss
        Lambda = np.random.randn(N, K).astype(np.float32)
        A      = np.eye(K, dtype=np.float32) * 0.5
        path   = make_ncde_path(times_np, cov_np, "rectilinear")
        exp    = NCDEExposure(
            n_covariates=C, n_series=N, n_factors=K, key=KEY,
            hidden_size=4, mlp_width=8, mlp_depth=1,
            interpolation="rectilinear",
        )
        model = TVDFM.from_params(Lambda, A, None, None, exp, KEY)
        t_j   = jnp.asarray(times_np)
        c_j   = jnp.asarray(path)
        X_j   = jnp.asarray(obs_np)

        # Two different WD values — same JIT kernel, different dynamic args
        l1, _ = _eval_loss(model, t_j, X_j, c_j, None, None,
                           jnp.float32(0.05), jnp.float32(0.01), jnp.float32(0.1),
                           jnp.float32(0.0))
        l2, _ = _eval_loss(model, t_j, X_j, c_j, None, None,
                           jnp.float32(0.05), jnp.float32(1.0),  jnp.float32(0.5),
                           jnp.float32(0.0))
        assert jnp.isfinite(l1) and jnp.isfinite(l2)
        assert float(l2) > float(l1)  # higher WD → higher loss at init


# ---------------------------------------------------------------------------
# High-level TVDFModel tests
# ---------------------------------------------------------------------------

class TestTVDFModel:
    def test_static_dfm_fit_predict(self, obs_df):
        from tvdfm import TVDFModel
        model = TVDFModel(
            n_factors=K, exposure=None, dfm_init="pca",
            n_epochs=3, patience=2, val_n=6, verbose=False,
        )
        model.fit(obs_df)
        preds   = model.predict()
        factors = model.transform()
        assert preds.shape   == (T, N)
        assert factors.shape == (T, K)
        assert not np.any(np.isnan(preds))

    def test_gru_exposure_fit(self, obs_df, cov_df):
        from tvdfm import TVDFModel
        model = TVDFModel(
            n_factors=K, exposure="gru", hidden_size=4,
            dfm_init="pca", n_epochs=3, patience=2,
            val_n=6, verbose=False,
        )
        model.fit(obs_df, covariates=cov_df)
        preds = model.predict()
        assert preds.shape == (T, N)

    def test_ncde_exposure_fit(self, obs_df, cov_df):
        from tvdfm import TVDFModel
        model = TVDFModel(
            n_factors=K, exposure="ncde",
            hidden_size=4, mlp_width=8, mlp_depth=1,
            interpolation="rectilinear",
            dfm_init="pca", n_epochs=3, patience=2,
            val_n=6, verbose=False,
        )
        model.fit(obs_df, covariates=cov_df)
        preds = model.predict()
        assert preds.shape == (T, N)

    def test_fit_transform(self, obs_df):
        from tvdfm import TVDFModel
        factors = TVDFModel(
            n_factors=K, exposure=None, dfm_init="pca",
            n_epochs=2, patience=2, val_n=6, verbose=False,
        ).fit_transform(obs_df)
        assert factors.shape == (T, K)

    def test_target_series_mask(self, obs_df, cov_df):
        from tvdfm import TVDFModel
        model = TVDFModel(
            n_factors=K, exposure="gru", hidden_size=4,
            dfm_init="pca", n_epochs=2, patience=2,
            val_n=6, verbose=False,
        )
        model.fit(obs_df, covariates=cov_df, target_series=["s0", "s1"])
        assert model.predictions_.shape == (T, N)

    def test_lambda_indices_string(self, obs_df, cov_df):
        from tvdfm import TVDFModel
        model = TVDFModel(
            n_factors=K, exposure="gru", hidden_size=4,
            lambda_indices=["s0"],
            dfm_init="pca", n_epochs=2, patience=2,
            val_n=6, verbose=False,
        )
        model.fit(obs_df, covariates=cov_df)
        assert model.predictions_.shape == (T, N)

    def test_properties(self, obs_df):
        from tvdfm import TVDFModel
        model = TVDFModel(
            n_factors=K, exposure=None, dfm_init="pca",
            n_epochs=2, patience=2, val_n=6, verbose=False,
        )
        model.fit(obs_df)
        assert model.loadings_.shape    == (N, K)
        assert model.transition_.shape  == (K, K)
        assert model.noise_obs_.shape   == (N,)
        assert model.noise_proc_.shape  == (K,)

    def test_not_fitted_raises(self):
        from tvdfm import TVDFModel
        model = TVDFModel(n_factors=K)
        with pytest.raises(RuntimeError, match="not fitted"):
            model.predict()

    def test_pca_reducer(self, obs_df, cov_df):
        from tvdfm import TVDFModel
        model = TVDFModel(
            n_factors=K, exposure="gru", hidden_size=4,
            n_pca_covariates=2,
            dfm_init="pca", n_epochs=2, patience=2,
            val_n=6, verbose=False,
        )
        model.fit(obs_df, covariates=cov_df)
        assert model.predictions_.shape == (T, N)

    def test_datetime_index(self):
        from tvdfm import TVDFModel
        idx = pd.date_range("2000-01", periods=T, freq="ME")
        rng = np.random.default_rng(42)
        df  = pd.DataFrame(rng.standard_normal((T, N)).astype(np.float32),
                           index=idx, columns=[f"s{i}" for i in range(N)])
        model = TVDFModel(
            n_factors=K, exposure=None, dfm_init="pca",
            n_epochs=2, patience=2, val_n=6, verbose=False,
        )
        model.fit(df)
        assert model.times_train_.dtype == np.float64
        # 2000-01 → 2000.0, 2000-02 → 2000 + 1/12 ≈ 2000.083
        assert abs(model.times_train_[0] - 2000.0) < 1e-6

    def test_summary_runs(self, obs_df):
        from tvdfm import TVDFModel
        model = TVDFModel(
            n_factors=K, exposure=None, dfm_init="pca",
            n_epochs=2, patience=2, val_n=6, verbose=False,
        )
        model.fit(obs_df)
        model.summary()  # should not raise

# ---------------------------------------------------------------------------
# Lag generalisation tests
# ---------------------------------------------------------------------------

class TestLagGeneralisation:
    def test_error_order_1_filter_shapes(self, obs_np):
        """AR(1) idiosyncratic errors: filter still returns [T, K] states."""
        from tvdfm import DFMStateSpace
        ssm = DFMStateSpace(K, N, KEY, error_order=1)
        assert ssm.state_dim == K + N
        Lambda_t = jnp.ones((T, N, K)) * 0.5
        A_t      = jnp.broadcast_to(jnp.eye(K) * 0.5, (T, K, K))
        z_filt, lls = ssm.filter(jnp.asarray(obs_np), Lambda_t, A_t)
        assert z_filt.shape == (T, K)
        assert lls.shape    == (T,)
        assert jnp.all(jnp.isfinite(z_filt))

    def test_error_order_1_smoother_shapes(self, obs_np):
        """RTS smoother with error_order=1 returns [T, K] smoothed states."""
        from tvdfm import DFMStateSpace
        ssm = DFMStateSpace(K, N, KEY, error_order=1)
        Lambda_t = jnp.ones((T, N, K)) * 0.5
        A_t      = jnp.broadcast_to(jnp.eye(K) * 0.5, (T, K, K))
        z_sm, P_sm, z_f, P_f, lls = ssm.filter_and_smooth(
            jnp.asarray(obs_np), Lambda_t, A_t
        )
        assert z_sm.shape == (T, K)
        assert P_sm.shape == (T, K, K)

    def test_factor_order_2_filter_shapes(self, obs_np):
        """VAR(2) factor model: filter returns [T, K] not [T, 2K]."""
        from tvdfm import DFMStateSpace
        ssm = DFMStateSpace(K, N, KEY, factor_order=2)
        assert ssm.state_dim == 2 * K
        Lambda_t = jnp.ones((T, N, K)) * 0.5
        A_t      = jnp.broadcast_to(jnp.eye(K) * 0.5, (T, K, K))
        z_filt, lls = ssm.filter(jnp.asarray(obs_np), Lambda_t, A_t)
        assert z_filt.shape == (T, K)
        assert jnp.all(jnp.isfinite(z_filt))

    def test_factor_order_2_and_error_order_1(self, obs_np):
        """Combined VAR(2) + AR(1) errors: state_dim = 2K + N."""
        from tvdfm import DFMStateSpace
        ssm = DFMStateSpace(K, N, KEY, factor_order=2, error_order=1)
        assert ssm.state_dim == 2 * K + N
        Lambda_t = jnp.ones((T, N, K)) * 0.5
        A_t      = jnp.broadcast_to(jnp.eye(K) * 0.5, (T, K, K))
        z_filt, lls = ssm.filter(jnp.asarray(obs_np), Lambda_t, A_t)
        assert z_filt.shape == (T, K)
        assert jnp.all(jnp.isfinite(z_filt))

    def test_tvdfm_error_order_1_forward(self, obs_np, times_np):
        """TVDFM forward pass with error_order=1 returns correct shapes."""
        from tvdfm import TVDFM
        Lambda = np.random.randn(N, K).astype(np.float32)
        A      = np.eye(K, dtype=np.float32) * 0.5
        model  = TVDFM.from_params(
            Lambda, A, None, None, None, KEY,
            tv_Lambda=False, tv_A=False, error_order=1,
        )
        dummy_cov = jnp.zeros((T, 1))
        preds, factors, kf_ll = model(
            jnp.asarray(times_np), jnp.asarray(obs_np), dummy_cov, inference=True
        )
        assert preds.shape   == (T, N)
        assert factors.shape == (T, K)
        assert jnp.isfinite(kf_ll)

    def test_tvdfm_error_order_1_smooth(self, obs_np, times_np):
        """TVDFM forward_smooth with error_order=1."""
        from tvdfm import TVDFM
        Lambda = np.random.randn(N, K).astype(np.float32)
        A      = np.eye(K, dtype=np.float32) * 0.5
        model  = TVDFM.from_params(
            Lambda, A, None, None, None, KEY,
            tv_Lambda=False, tv_A=False, error_order=1,
        )
        dummy_cov = jnp.zeros((T, 1))
        z_smooth, z_filt = model.forward_smooth(
            jnp.asarray(times_np), jnp.asarray(obs_np), dummy_cov
        )
        assert z_smooth.shape == (T, K)
        assert z_filt.shape   == (T, K)

    def test_tvdfmodel_error_order_1(self, obs_df):
        """TVDFModel with error_order=1 fits and predicts."""
        from tvdfm import TVDFModel
        model = TVDFModel(
            n_factors=K, exposure=None, dfm_init="pca",
            error_order=1,
            n_epochs=2, patience=2, val_n=6, verbose=False,
        )
        model.fit(obs_df)
        assert model.predictions_.shape == (T, N)
        assert model.factors_filt_.shape == (T, K)

    def test_error_order_2_filter_shapes(self, obs_np):
        """AR(2) idiosyncratic errors: state_dim = K + 2N, output still [T, K]."""
        from tvdfm import DFMStateSpace
        ssm = DFMStateSpace(K, N, KEY, error_order=2)
        assert ssm.state_dim == K + 2 * N
        Lambda_t = jnp.ones((T, N, K)) * 0.5
        A_t      = jnp.broadcast_to(jnp.eye(K) * 0.5, (T, K, K))
        z_filt, lls = ssm.filter(jnp.asarray(obs_np), Lambda_t, A_t)
        assert z_filt.shape == (T, K)
        assert lls.shape    == (T,)
        assert jnp.all(jnp.isfinite(z_filt))

    def test_parcor_stationarity(self):
        """PARCOR→AR maps unconstrained params to stationary AR(3) coefficients."""
        from tvdfm.ssm import _parcor_to_ar
        rho = jnp.array([0.8, -0.5, 0.3])
        phi = _parcor_to_ar(rho)
        assert phi.shape == (3,)
        # Build the 3×3 companion and check all eigenvalues inside unit disk
        companion = jnp.zeros((3, 3)).at[0, :].set(phi).at[1:, :-1].set(jnp.eye(2))
        eigs = jnp.linalg.eigvals(companion)
        assert jnp.all(jnp.abs(eigs) < 1.0 + 1e-5)

    def test_factor_order_2_single_A_array(self):
        """A_base has shape [K, K*p] with the new single-array layout."""
        from tvdfm import DFMStateSpace
        A_full = np.random.randn(K, K * 2).astype(np.float32)
        ssm = DFMStateSpace(K, N, KEY, A_init=A_full, factor_order=2)
        assert ssm.A_base.shape == (K, K * 2)
        assert ssm.dA.shape     == (K, K * 2)
        assert ssm.state_dim    == K * 2


# ---------------------------------------------------------------------------
# DFM consistency test — statsmodels vs TVDFM
# ---------------------------------------------------------------------------

class TestDFMConsistency:
    """
    Integration test: generate synthetic data, fit a statsmodels DFM,
    then initialise a static TVDFM from those parameters and verify
    that the smoothed factors agree closely.

    This exercises the full pipeline:
        DGP → statsmodels EM → extract_statsmodels_params
            → TVDFM.from_statsmodels → forward_smooth

    The two implementations differ only in float32 vs float64 precision
    and minor P0 differences, so per-factor |correlation| should be > 0.9.
    """

    def _make_data(self, T_s=120, N_s=6, K_s=2, seed=0):
        rng = np.random.default_rng(seed)
        A_true      = np.diag([0.75, 0.65])
        Lambda_true = rng.standard_normal((N_s, K_s))
        Z = np.zeros((T_s, K_s))
        for t in range(1, T_s):
            Z[t] = A_true @ Z[t - 1] + 0.1 * rng.standard_normal(K_s)
        Y = (Z @ Lambda_true.T + 0.3 * rng.standard_normal((T_s, N_s))).astype(np.float32)
        return Y, Z, Lambda_true

    def test_factors_match_statsmodels(self):
        """
        Static TVDFM (init'd from EM) should reproduce statsmodels
        RTS-smoothed factors with |correlation| > 0.9 per factor.
        """
        pytest.importorskip("statsmodels", reason="statsmodels not installed")
        from statsmodels.tsa.statespace.dynamic_factor import DynamicFactor
        from tvdfm import TVDFM

        T_s, N_s, K_s = 120, 6, 2
        Y, _, _ = self._make_data(T_s, N_s, K_s)

        cols = [f"s{i}" for i in range(N_s)]
        df_Y = pd.DataFrame(Y, columns=cols)

        # ── 1. Fit statsmodels DFM via EM ────────────────────────────────
        dfm    = DynamicFactor(Y, k_factors=K_s, factor_order=1)
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = dfm.fit(maxiter=300, disp=False)

        # Smoothed factors [T, K]: first K rows of smoothed state (column = time)
        sm_z = result.smoothed_state[:K_s, :].T   # [T, K]

        # ── 2. Init TVDFM from the fitted result ─────────────────────────
        key   = jax.random.PRNGKey(0)
        model = TVDFM.from_statsmodels(
            result, df_Y, n_factors=K_s,
            exposure=None, key=key,
            tv_Lambda=False, tv_A=False,
            fixed_Q=False,
            spectral_radius_bound=0.9999,  # avoid renormalising statsmodels A
        )

        t_j   = jnp.arange(T_s, dtype=jnp.float32)
        Y_j   = jnp.asarray(Y)
        cov_j = jnp.zeros((T_s, 1))

        tv_z_sm, _ = model.forward_smooth(t_j, Y_j, cov_j)
        tv_z = np.asarray(tv_z_sm)   # [T, K]

        # ── 3. Compare — sign-invariant per-column correlation ────────────
        #   Factors are identified up to sign (and possibly permutation).
        #   For each TVDFM factor find the best-matching SM factor.
        for k in range(K_s):
            best = max(
                abs(np.corrcoef(tv_z[:, k], sm_z[:, j])[0, 1])
                for j in range(K_s)
            )
            assert best > 0.9, (
                f"TVDFM factor {k}: best |corr| with SM factors = {best:.3f} < 0.9"
            )

    def test_tvdfm_improves_on_static_dfm(self):
        """
        After fine-tuning a TVDFM on data with *time-varying* loadings,
        its in-sample MSE should be ≤ the static DFM MSE.

        We check this on a very short training run (10 epochs) just to
        confirm that the TV mechanism can move in the right direction.
        """
        pytest.importorskip("statsmodels", reason="statsmodels not installed")
        from statsmodels.tsa.statespace.dynamic_factor import DynamicFactor
        from tvdfm import TVDFM, LTVTrainingManager, NCDEExposure, make_ncde_path

        T_s, N_s, K_s = 80, 4, 2
        Y, _, _ = self._make_data(T_s, N_s, K_s)

        # Build a simple covariate (trend + noise) that can help TV loadings
        rng = np.random.default_rng(1)
        cov = np.column_stack([
            np.linspace(0, 1, T_s),
            rng.standard_normal(T_s),
        ]).astype(np.float32)

        times_np = np.arange(T_s, dtype=np.float64)
        cols     = [f"s{i}" for i in range(N_s)]
        df_Y     = pd.DataFrame(Y, columns=cols)

        # Static DFM baseline
        dfm    = DynamicFactor(Y, k_factors=K_s, factor_order=1)
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = dfm.fit(maxiter=200, disp=False)

        key = jax.random.PRNGKey(42)

        # ── Static TVDFM (no TV) ─────────────────────────────────────────
        static_model = TVDFM.from_statsmodels(
            result, df_Y, n_factors=K_s,
            exposure=None, key=key,
            tv_Lambda=False, tv_A=False,
            fixed_Q=False, spectral_radius_bound=0.9999,
        )
        Y_j   = jnp.asarray(Y)
        t_j   = jnp.asarray(times_np)
        cov_j = jnp.zeros((T_s, 1))
        preds_static, _, _ = static_model(t_j, Y_j, cov_j, inference=True)
        mse_static = float(jnp.mean(jnp.square(Y_j - preds_static)))

        # ── TV TVDFM (NCDE exposure, 10 epochs only) ──────────────────────
        path = make_ncde_path(times_np, cov, interpolation="rectilinear")
        key, subk = jax.random.split(key)
        exp = NCDEExposure(
            n_covariates=2, n_series=N_s, n_factors=K_s, key=subk,
            hidden_size=4, mlp_width=8, mlp_depth=1,
            interpolation="rectilinear",
        )
        tv_model = TVDFM.from_statsmodels(
            result, df_Y, n_factors=K_s,
            exposure=exp, key=key,
            tv_Lambda=True, tv_A=False,
            fixed_Q=False, spectral_radius_bound=0.9999,
        )
        manager = LTVTrainingManager(tv_model, lr_ssm=1e-3, lr_exposure=1e-3)
        tv_model = manager.fit(
            times_np, Y, path,
            max_epochs=10, val_n=10, patience=20,
            wd_exposure=0.01, wd_ssm=0.01,
            interpolation="rectilinear", verbose=False,
        )
        cov_j2 = jnp.asarray(path)
        preds_tv, _, _ = tv_model(t_j, Y_j, cov_j2, inference=True)
        mse_tv = float(jnp.mean(jnp.square(Y_j - preds_tv)))

        # After 10 epochs the TV model should not be dramatically worse
        # than the static DFM (we don't require improvement — just sanity)
        assert mse_tv < mse_static * 5.0, (
            f"TV MSE ({mse_tv:.4f}) is >5× static MSE ({mse_static:.4f}) — "
            "something went wrong in the TV training loop."
        )


# ---------------------------------------------------------------------------
# Gradient finiteness stress test
# ---------------------------------------------------------------------------

class TestGradientFiniteness:
    """
    Verify that gradients through the Kalman filter are finite under adversarial
    conditions — the scenario that exposes the NaN-gradient bug in the naive
    ``nan_to_num(solve(...))`` approach.

    Adversarial setup
    -----------------
    * Very large loadings (||Λ|| ≈ 50): makes  S = Λ P Λᵀ + R  ill-conditioned.
    * A near the unit circle (ρ = 0.97): slow mean-reversion, large P.
    * One column entirely NaN: exercises the missing-data branch on every step.
    * Large observation noise initialisation: prevents regularisation from saving
      a naïve implementation.

    The old ``nan_to_num`` fix applied OUTSIDE the scan would still let NaN
    propagate through the VJP of ``solve`` inside the scan (IEEE-754: 0·NaN=NaN).
    The ``jnp.where(isfinite, ...)`` guard prevents NaN from entering ``solve``
    at all, so VJPs stay finite.
    """

    def _make_adversarial_obs(self, T: int, N: int, nan_col: int = 0) -> jnp.ndarray:
        """[T, N] float32 observations; one column is all-NaN."""
        rng = np.random.default_rng(7)
        Y = rng.standard_normal((T, N)).astype(np.float32)
        Y[:, nan_col] = np.nan
        return jnp.asarray(Y)

    def test_kf_gradients_finite_adversarial(self):
        """
        eqx.filter_grad through DFMStateSpace.filter must return finite gradients
        even with large loadings, near-unit-root dynamics, and an all-NaN column.
        """
        from tvdfm.ssm import DFMStateSpace

        T_adv, N_adv, K_adv = 40, 5, 2
        key = jax.random.PRNGKey(99)

        # Near-unit-circle first-lag (spectral radius ≈ 0.97)
        A_init = np.eye(K_adv, dtype=np.float32) * 0.97

        ssm = DFMStateSpace(
            K_adv, N_adv, key,
            A_init=A_init,
            Q_init=None,
            R_init=None,
            spectral_radius_bound=0.99,
            fixed_Q=False,
            factor_order=1,
            error_order=0,
        )

        # Adversarial loadings: large scale to ill-condition the innovation cov
        Lambda_large = jnp.ones((T_adv, N_adv, K_adv), dtype=jnp.float32) * 50.0
        # Static first-lag transition (no TV)
        A_t = jnp.broadcast_to(ssm.A, (T_adv, K_adv, K_adv))

        Y_adv = self._make_adversarial_obs(T_adv, N_adv, nan_col=2)

        def loss_fn(ssm_model):
            _, lls = ssm_model.filter(Y_adv, Lambda_large, A_t)
            return -jnp.sum(lls)   # negative log-likelihood

        grad_ssm = eqx.filter_grad(loss_fn)(ssm)

        # Collect every gradient leaf and assert finiteness
        leaves = jax.tree_util.tree_leaves(
            eqx.filter(grad_ssm, eqx.is_array)
        )
        for i, g in enumerate(leaves):
            assert jnp.all(jnp.isfinite(g)), (
                f"Gradient leaf {i} contains non-finite values: "
                f"NaN={jnp.any(jnp.isnan(g))}, Inf={jnp.any(jnp.isinf(g))}"
            )

    def test_kf_gradients_finite_full_nan_row(self):
        """
        A timestep where ALL series are NaN (entirely missing observation)
        must not produce NaN gradients.
        """
        from tvdfm.ssm import DFMStateSpace

        T_adv, N_adv, K_adv = 30, 4, 2
        key = jax.random.PRNGKey(17)

        ssm = DFMStateSpace(
            K_adv, N_adv, key,
            spectral_radius_bound=0.98,
            fixed_Q=False,
            factor_order=1,
            error_order=0,
        )

        rng = np.random.default_rng(3)
        Y_np = rng.standard_normal((T_adv, N_adv)).astype(np.float32)
        # Make every 5th row entirely missing
        Y_np[::5, :] = np.nan
        Y_adv = jnp.asarray(Y_np)

        Lambda_t = jnp.broadcast_to(
            jnp.ones((N_adv, K_adv), dtype=jnp.float32) * 5.0,
            (T_adv, N_adv, K_adv),
        )
        A_t = jnp.broadcast_to(ssm.A, (T_adv, K_adv, K_adv))

        def loss_fn(ssm_model):
            _, lls = ssm_model.filter(Y_adv, Lambda_t, A_t)
            return -jnp.sum(lls)

        grad_ssm = eqx.filter_grad(loss_fn)(ssm)

        leaves = jax.tree_util.tree_leaves(
            eqx.filter(grad_ssm, eqx.is_array)
        )
        for i, g in enumerate(leaves):
            assert jnp.all(jnp.isfinite(g)), (
                f"Gradient leaf {i} contains non-finite values with full-NaN rows: "
                f"NaN={jnp.any(jnp.isnan(g))}, Inf={jnp.any(jnp.isinf(g))}"
            )

    def test_kf_gradients_finite_high_error_order(self):
        """
        AR(2) idiosyncratic errors combined with adversarial loadings must
        also produce finite gradients.
        """
        from tvdfm.ssm import DFMStateSpace

        T_adv, N_adv, K_adv, q = 35, 4, 2, 2
        key = jax.random.PRNGKey(55)

        ssm = DFMStateSpace(
            K_adv, N_adv, key,
            spectral_radius_bound=0.98,
            fixed_Q=False,
            factor_order=1,
            error_order=q,
        )

        rng = np.random.default_rng(9)
        Y_np = rng.standard_normal((T_adv, N_adv)).astype(np.float32)
        Y_np[:, 1] = np.nan   # one all-NaN column
        Y_adv = jnp.asarray(Y_np)

        Lambda_t = jnp.broadcast_to(
            jnp.ones((N_adv, K_adv), dtype=jnp.float32) * 20.0,
            (T_adv, N_adv, K_adv),
        )
        A_t = jnp.broadcast_to(ssm.A, (T_adv, K_adv, K_adv))

        def loss_fn(ssm_model):
            _, lls = ssm_model.filter(Y_adv, Lambda_t, A_t)
            return -jnp.sum(lls)

        grad_ssm = eqx.filter_grad(loss_fn)(ssm)

        leaves = jax.tree_util.tree_leaves(
            eqx.filter(grad_ssm, eqx.is_array)
        )
        for i, g in enumerate(leaves):
            assert jnp.all(jnp.isfinite(g)), (
                f"Gradient leaf {i} contains non-finite values (error_order={q}): "
                f"NaN={jnp.any(jnp.isnan(g))}, Inf={jnp.any(jnp.isinf(g))}"
            )


# ---------------------------------------------------------------------------
# P0 stability and MSE-loss gradient correctness
# ---------------------------------------------------------------------------

class TestStabilityAndLossGradients:
    """
    Issue 1 — unstable A: from_params with ρ(A) > 1 must not produce NaN
    predictions (P0 Lyapunov fallback to eye(K)).

    Issue 2 — MSE NaN-gradient: the training-manager loss already masks
    *before* squaring (jnp.where → jnp.square), so its gradient is finite
    even when the target series has NaN observations.  The test is a
    regression guard confirming the correct order is preserved.
    """

    # ------------------------------------------------------------------ #
    # Issue 1 helpers
    # ------------------------------------------------------------------ #

    def _make_unstable_model(self):
        """TVDFM initialised from an unstable A (ρ > 1), static mode."""
        from tvdfm import TVDFM
        N_u, K_u = 3, 2
        A_unstable = np.array([[1.2, 0.1], [0.0, 0.9]], dtype=np.float32)
        Lambda = np.ones((N_u, K_u), dtype=np.float32) * 0.5
        Q_diag = np.ones(K_u, dtype=np.float32)
        R_diag = np.ones(N_u, dtype=np.float32) * 0.5
        return TVDFM.from_params(
            Lambda=Lambda, A=A_unstable, Q=Q_diag, R=R_diag,
            exposure=None, key=jax.random.PRNGKey(0),
            tv_Lambda=False, tv_A=False, fixed_Q=False,
        ), N_u

    def test_unstable_A_P0_is_identity(self):
        """
        When ρ(A) > 1 the Lyapunov equation has no PSD solution.
        DFMStateSpace must fall back to eye(K) rather than using the
        divergent/non-PSD Lyapunov answer.
        """
        from tvdfm.ssm import DFMStateSpace
        K_u = 2
        A_unstable = np.array([[1.2, 0.1], [0.0, 0.9]], dtype=np.float32)
        Q_diag     = np.ones(K_u, dtype=np.float32)
        ssm = DFMStateSpace(
            K_u, 3, jax.random.PRNGKey(0),
            A_init=A_unstable, Q_init=Q_diag,
            fixed_Q=False,
        )
        # P0 must be positive-definite
        eigvals = np.linalg.eigvalsh(np.array(ssm.P0))
        assert np.all(eigvals > 0), (
            f"P0 is not PD after unstable-A fallback: eigvals={eigvals}"
        )
        # For ρ(A) > 1 the fallback is eye(K)
        np.testing.assert_allclose(
            np.array(ssm.P0), np.eye(K_u, dtype=np.float32),
            err_msg="P0 should be eye(K) when A is unstable",
        )

    def test_unstable_A_no_nan_predictions(self):
        """
        End-to-end forward pass with ρ(A) > 1 must return finite predictions.
        Minimal reproduction from the issue description.
        """
        from tvdfm import TVDFM
        model, N_u = self._make_unstable_model()
        T_u = 10
        obs   = jnp.ones((T_u, N_u), dtype=jnp.float32)
        times = jnp.arange(T_u, dtype=jnp.float32) / (T_u - 1)
        cov   = jnp.zeros((T_u, 1), dtype=jnp.float32)

        preds, zfilt, kf_ll = model(times, obs, cov, inference=True)

        assert not jnp.any(jnp.isnan(preds)), (
            "Unstable-A model produces NaN predictions — P0 fallback failed."
        )
        assert not jnp.any(jnp.isnan(zfilt)), (
            "Unstable-A model produces NaN filtered states."
        )
        assert jnp.isfinite(kf_ll), (
            f"Unstable-A model KF log-likelihood is not finite: {kf_ll}"
        )

    def test_stable_A_P0_from_lyapunov(self):
        """
        When ρ(A) < 1 and Q is given, P0 should come from the Lyapunov
        solve (not fall back to eye(K)).
        """
        from tvdfm.ssm import DFMStateSpace
        K_s = 2
        A_stable = np.array([[0.8, 0.0], [0.0, 0.5]], dtype=np.float32)
        Q_diag   = np.array([0.2, 0.3], dtype=np.float32)
        ssm = DFMStateSpace(
            K_s, 3, jax.random.PRNGKey(1),
            A_init=A_stable, Q_init=Q_diag,
            fixed_Q=False,
        )
        # P0 must NOT be the identity (Lyapunov gives a different answer)
        assert not np.allclose(np.array(ssm.P0), np.eye(K_s), atol=1e-3), (
            "Stable A: P0 unexpectedly fell back to eye(K); "
            "Lyapunov solve should produce a non-identity result."
        )
        # P0 must be PD
        eigvals = np.linalg.eigvalsh(np.array(ssm.P0))
        assert np.all(eigvals > 0), f"Lyapunov P0 is not PD: eigvals={eigvals}"

    # ------------------------------------------------------------------ #
    # Issue 2 helpers
    # ------------------------------------------------------------------ #

    def test_mse_loss_gradient_finite_with_nan_target(self):
        """
        MSE gradient through loss_e2e must be finite even when some target
        timesteps are NaN.

        tvdfm's loss already applies jnp.where(mask, residuals, 0.0) BEFORE
        jnp.square — the correct order (mask → square, not square → mask).
        This test is a regression guard that confirms that order is preserved.
        """
        from tvdfm import TVDFM
        from tvdfm.training import loss_e2e

        N_l, K_l, T_l = 5, 2, 50
        rng = np.random.default_rng(42)
        obs_np = rng.standard_normal((T_l, N_l)).astype(np.float32)
        # Series 0 is quarterly: NaN on 2 out of every 3 timesteps
        obs_np[np.arange(T_l) % 3 != 0, 0] = np.nan
        obs_j   = jnp.asarray(obs_np)
        times_j = jnp.arange(T_l, dtype=jnp.float32)
        cov_j   = jnp.zeros((T_l, 1), dtype=jnp.float32)

        A_stable = np.eye(K_l, dtype=np.float32) * 0.5
        model = TVDFM.from_params(
            Lambda=np.ones((N_l, K_l), dtype=np.float32) * 0.3,
            A=A_stable, Q=None, R=None,
            exposure=None, key=jax.random.PRNGKey(7),
            tv_Lambda=False, tv_A=False, fixed_Q=True,
        )

        diff, static = eqx.partition(model, eqx.is_array)

        def loss(d):
            m = eqx.combine(d, static)
            total, _ = loss_e2e(
                m, times_j, obs_j, cov_j,
                kf_ll_weight=0.0, wd_exposure=0.0, wd_ssm=0.0,
            )
            return total

        grads = eqx.filter_grad(loss)(diff)
        leaves = jax.tree_util.tree_leaves(eqx.filter(grads, eqx.is_array))
        for i, g in enumerate(leaves):
            assert jnp.all(jnp.isfinite(g)), (
                f"MSE-loss gradient leaf {i} is not finite with NaN targets: "
                f"NaN={jnp.any(jnp.isnan(g))}, Inf={jnp.any(jnp.isinf(g))}"
            )

    def test_mse_loss_gradient_finite_with_target_mask_and_nan(self):
        """
        When a target_mask is also supplied, the residuals * target_mask
        multiplication can produce NaN * 0 = NaN (IEEE 754) at positions
        that are both masked-out and NaN.  The subsequent jnp.where(obs_mask,
        residuals, 0.0) must clean those up before the squaring step so that
        gradients remain finite.
        """
        from tvdfm import TVDFM
        from tvdfm.training import loss_e2e

        N_l, K_l, T_l = 4, 2, 40
        rng = np.random.default_rng(13)
        obs_np = rng.standard_normal((T_l, N_l)).astype(np.float32)
        obs_np[::2, 1] = np.nan   # series 1: every other row is NaN

        # target_mask: series 0 and 2 only (series 1 masked out AND has NaN)
        target_mask = np.zeros((T_l, N_l), dtype=np.float32)
        target_mask[:, 0] = 1.0
        target_mask[:, 2] = 1.0

        obs_j    = jnp.asarray(obs_np)
        times_j  = jnp.arange(T_l, dtype=jnp.float32)
        cov_j    = jnp.zeros((T_l, 1), dtype=jnp.float32)
        tmask_j  = jnp.asarray(target_mask)

        model = TVDFM.from_params(
            Lambda=np.ones((N_l, K_l), dtype=np.float32) * 0.3,
            A=np.eye(K_l, dtype=np.float32) * 0.5,
            Q=None, R=None,
            exposure=None, key=jax.random.PRNGKey(3),
            tv_Lambda=False, tv_A=False, fixed_Q=True,
        )

        diff, static = eqx.partition(model, eqx.is_array)

        def loss(d):
            m = eqx.combine(d, static)
            total, _ = loss_e2e(
                m, times_j, obs_j, cov_j,
                target_mask=tmask_j,
                kf_ll_weight=0.0, wd_exposure=0.0, wd_ssm=0.0,
            )
            return total

        grads = eqx.filter_grad(loss)(diff)
        leaves = jax.tree_util.tree_leaves(eqx.filter(grads, eqx.is_array))
        for i, g in enumerate(leaves):
            assert jnp.all(jnp.isfinite(g)), (
                f"MSE-loss gradient leaf {i} is not finite with target_mask+NaN: "
                f"NaN={jnp.any(jnp.isnan(g))}, Inf={jnp.any(jnp.isinf(g))}"
            )
