"""Tests for regularisation terms in loss_e2e."""
import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
import equinox as eqx
import pytest

from tvdfm import TVDFM, DFMStateSpace
from tvdfm.training import (
    RegularisationConfig,
    loss_e2e,
    compute_regularisation_terms,
)

# ---------------------------------------------------------------------------
# Shared fixture helpers
# ---------------------------------------------------------------------------

T, N, K = 30, 4, 2

def _make_static_model():
    """Minimal static DFM (no exposure, tv_Lambda=False)."""
    key = jax.random.PRNGKey(0)
    return TVDFM.from_params(
        Lambda=np.ones((N, K), dtype=np.float32) * 0.3,
        A=np.eye(K, dtype=np.float32) * 0.5,
        Q=None, R=None,
        exposure=None, key=key,
        tv_Lambda=False, tv_A=False, fixed_Q=True,
    )

def _make_tv_model():
    """TVDFM with GRUExposure (tv_Lambda=True)."""
    from tvdfm.exposure import GRUExposure
    key = jax.random.PRNGKey(1)
    exp = GRUExposure(N, N, K, key, hidden_size=8)
    return TVDFM.from_params(
        Lambda=np.ones((N, K), dtype=np.float32) * 0.3,
        A=np.eye(K, dtype=np.float32) * 0.5,
        Q=None, R=None,
        exposure=exp, key=key,
        tv_Lambda=True, tv_A=False, fixed_Q=True,
    )

def _make_inputs():
    rng = np.random.default_rng(42)
    obs = jnp.asarray(rng.standard_normal((T, N)).astype(np.float32))
    times = jnp.arange(T, dtype=jnp.float32)
    cov   = jnp.zeros((T, N), dtype=jnp.float32)
    return times, obs, cov


# ---------------------------------------------------------------------------
# TestRegularisationConfig
# ---------------------------------------------------------------------------

class TestRegularisationConfig:

    def test_default_valid(self):
        RegularisationConfig().validate()

    def test_negative_lambda_dev_raises(self):
        with pytest.raises(ValueError):
            RegularisationConfig(lambda_dev_weight=-1.0).validate()


# ---------------------------------------------------------------------------
# TestComputeRegularisationTerms
# ---------------------------------------------------------------------------

class TestComputeRegularisationTerms:

    def test_lambda_dev_zero_for_static_model(self):
        """Static model: tv_Lambda=False → lambda_dev is always 0 regardless of delta_Lambda."""
        model = _make_static_model()
        # Even if we pass a non-zero delta_Lambda, tv_Lambda=False means penalty=0
        dummy_delta = jnp.ones((T, N, K), dtype=jnp.float32)
        terms = compute_regularisation_terms(model, delta_Lambda=dummy_delta)
        assert float(terms["lambda_dev"]) == 0.0

    def test_lambda_dev_zero_when_delta_is_zero(self):
        """delta_Lambda=0 → penalty = 0 even for tv model."""
        model = _make_tv_model()
        zero_delta = jnp.zeros((T, N, K), dtype=jnp.float32)
        terms = compute_regularisation_terms(model, delta_Lambda=zero_delta)
        assert float(terms["lambda_dev"]) < 1e-6, float(terms["lambda_dev"])

    def test_lambda_dev_zero_when_none(self):
        """delta_Lambda=None → penalty = 0."""
        model = _make_tv_model()
        terms = compute_regularisation_terms(model, delta_Lambda=None)
        assert float(terms["lambda_dev"]) == 0.0

    def test_lambda_dev_nonnegative(self):
        model = _make_tv_model()
        times, _, cov = _make_inputs()
        delta_Lambda, _ = model.exposure(times, cov, inference=True)
        terms = compute_regularisation_terms(model, delta_Lambda=delta_Lambda)
        assert float(terms["lambda_dev"]) >= 0.0


# ---------------------------------------------------------------------------
# TestLossE2E
# ---------------------------------------------------------------------------

class TestLossE2E:

    def test_returns_tuple(self):
        model = _make_static_model()
        times, obs, cov = _make_inputs()
        result = loss_e2e(model, times, obs, cov)
        assert isinstance(result, tuple) and len(result) == 2

    def test_breakdown_keys(self):
        model = _make_static_model()
        times, obs, cov = _make_inputs()
        _, bd = loss_e2e(model, times, obs, cov)
        assert set(bd.keys()) == {"mse", "kf_ll", "lambda_dev", "total"}

    def test_total_is_finite(self):
        model = _make_static_model()
        times, obs, cov = _make_inputs()
        total, _ = loss_e2e(model, times, obs, cov)
        assert jnp.isfinite(total)

    def test_total_equals_breakdown_total(self):
        model = _make_static_model()
        times, obs, cov = _make_inputs()
        total, bd = loss_e2e(model, times, obs, cov)
        np.testing.assert_allclose(float(total), float(bd["total"]), rtol=1e-5)

    def test_total_formula(self):
        """total = mse - kf_ll_w*kf_ll + wd_exp*l2_exp + wd_ssm*l2_ssm + dev_w*dev + anc_w*anc"""
        model = _make_static_model()
        times, obs, cov = _make_inputs()
        total, bd = loss_e2e(
            model, times, obs, cov,
            kf_ll_weight=0.1, wd_exposure=0.0, wd_ssm=0.0,
            lambda_dev_weight=0.5,
        )
        expected = (
            bd["mse"]
            - 0.1 * bd["kf_ll"]
            + 0.5 * bd["lambda_dev"]
        )
        np.testing.assert_allclose(float(total), float(expected), rtol=1e-5)

    def test_gradient_finite_with_nan_obs(self):
        model = _make_static_model()
        rng = np.random.default_rng(7)
        obs_np = rng.standard_normal((T, N)).astype(np.float32)
        obs_np[::3, 0] = np.nan
        times = jnp.arange(T, dtype=jnp.float32)
        obs   = jnp.asarray(obs_np)
        cov   = jnp.zeros((T, N), dtype=jnp.float32)

        diff, static = eqx.partition(model, eqx.is_array)
        def loss_fn(d):
            m = eqx.combine(d, static)
            total, _ = loss_e2e(m, times, obs, cov, kf_ll_weight=0.0,
                                wd_exposure=0.0, wd_ssm=0.0)
            return total

        grads = eqx.filter_grad(loss_fn)(diff)
        for g in jax.tree_util.tree_leaves(eqx.filter(grads, eqx.is_array)):
            assert jnp.all(jnp.isfinite(g))

    def test_lambda_dev_weight_adds_gradient(self):
        """lambda_dev_weight > 0 increases gradient magnitude on TV model."""
        model = _make_tv_model()
        times, obs, cov = _make_inputs()
        diff, static = eqx.partition(model, eqx.is_array)

        def make_loss(lam_w):
            def loss_fn(d):
                m = eqx.combine(d, static)
                total, _ = loss_e2e(m, times, obs, cov,
                                    kf_ll_weight=0.0, wd_exposure=0.0, wd_ssm=0.0,
                                    lambda_dev_weight=lam_w)
                return total
            return loss_fn

        _, g0 = jax.value_and_grad(make_loss(0.0))(diff)
        _, g1 = jax.value_and_grad(make_loss(10.0))(diff)

        def _max_grad(g_tree):
            vals = [float(jnp.abs(g).max()) for g in jax.tree_util.tree_leaves(eqx.filter(g_tree, eqx.is_array)) if g.size > 0]
            return max(vals) if vals else 0.0

        max0 = _max_grad(g0)
        max1 = _max_grad(g1)
        assert max1 >= max0

