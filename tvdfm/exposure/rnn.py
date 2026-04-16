"""
RNN-based exposure models.

Missing-value strategies for mixed-frequency covariates
-------------------------------------------------------
``"forward"``
    Forward-fill: last observed value persists until next release.
    Fast, causal, good baseline.

``"ssm"``
    Kalman-smoother imputation: a pre-fitted :class:`~tvdfm.TVDFM` on the
    covariate panel replaces NaN entries with smoother means.  More
    principled for auto-correlated macro indicators.  Pass the fitted model
    via the ``coeffs`` argument at call time.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
from typing import Any, Literal, Optional, Tuple

from .base import AbstractExposure, forward_fill_nans


class GRUExposure(AbstractExposure):
    """
    Exposure model based on a Gated Recurrent Unit (GRU).

    Hidden state:  h_t = GRU(h_{t-1}, x_t),   h_0 = 0
    where x_t is the imputed covariate vector at time t.
    Readout layers map h_t -> (delta_Lambda_t, delta_A_t).

    Mixed-frequency covariates
    --------------------------
    ``imputation="forward"`` (default)
        Forward-fill NaN entries before the scan. Fast and causal.

    ``imputation="ssm"``
        Use a pre-fitted Kalman smoother to replace NaN entries with smoother
        means.  Pass a fitted :class:`~tvdfm.TVDFM` instance for the covariate
        panel via the ``coeffs`` argument at call time.

    Parameters
    ----------
    lambda_indices : Optional[Tuple[int, ...]]
        - ``None``        → delta_Lambda for all N series.
        - ``(i, j, ...)`` → only those rows (scattered into [T, N, K]).
    imputation : {"forward", "ssm"}
    input_reducer : Optional[eqx.nn.Linear]
        Optional linear projection [C_in] -> [n_covariates] before the GRU.
        Build from PCA::

            reducer = AbstractExposure.make_pca_reducer(X_train, 10, key)
            model   = GRUExposure(..., n_covariates=10, input_reducer=reducer)
    """

    gru_cell: eqx.nn.GRUCell
    readout: eqx.nn.Linear
    readout_A: eqx.nn.Linear

    hidden_size: int = eqx.field(static=True)
    dropout: float = eqx.field(static=True)
    imputation: str = eqx.field(static=True)

    def __init__(
        self,
        n_covariates: int,
        n_series: int,
        n_factors: int,
        key: jax.Array,
        hidden_size: int = 32,
        dropout: float = 0.1,
        lambda_indices: Optional[Tuple[int, ...]] = None,
        imputation: Literal["forward", "ssm"] = "forward",
        input_reducer: Optional[eqx.nn.Linear] = None,
    ):
        self.n_series = n_series
        self.n_factors = n_factors
        self.n_covariates = n_covariates
        self.lambda_indices = lambda_indices
        self.input_reducer = input_reducer

        self.hidden_size = hidden_size
        self.dropout = dropout
        self.imputation = imputation

        keys = jax.random.split(key, 3)
        n_out = self.n_lambda_out

        self.gru_cell = eqx.nn.GRUCell(n_covariates, hidden_size, key=keys[0])

        # Zero-initialize readouts.
        raw_ro = eqx.nn.Linear(hidden_size, n_out * n_factors, key=keys[1])
        self.readout = eqx.tree_at(lambda l: l.weight, raw_ro, jnp.zeros_like(raw_ro.weight))
        self.readout = eqx.tree_at(lambda l: l.bias, self.readout, jnp.zeros_like(self.readout.bias))

        raw_ro_A = eqx.nn.Linear(hidden_size, n_factors * n_factors, key=keys[2])
        self.readout_A = eqx.tree_at(lambda l: l.weight, raw_ro_A, jnp.zeros_like(raw_ro_A.weight))
        self.readout_A = eqx.tree_at(lambda l: l.bias, self.readout_A, jnp.zeros_like(self.readout_A.bias))

    # ------------------------------------------------------------------
    # Imputation helpers
    # ------------------------------------------------------------------

    def _impute_ssm(self, covariates: jnp.ndarray, covariate_ssm: Any) -> jnp.ndarray:
        """
        Replace NaN entries using a pre-fitted Kalman smoother.

        ``covariate_ssm`` must be a fitted :class:`~tvdfm.TVDFM` instance whose
        ``Lambda_base`` [C, K_cov] and ``ssm`` are configured for the covariate
        panel.  Smoother means are reconstructed as ``Lambda_base @ z_smooth``.
        Original observed values are preserved; only NaN positions are replaced.

        Note: pass the full ``TVDFM`` object, **not** a bare ``DFMStateSpace``.
        ``DFMStateSpace`` does not carry ``Lambda_base``.
        """
        T, C = covariates.shape
        K_cov = covariate_ssm.ssm.n_factors
        Lambda_cov = jnp.asarray(covariate_ssm.Lambda_base)   # [C, K_cov]
        A_static = covariate_ssm.ssm.A                         # [K_cov, K_cov]
        Lambda_t = jnp.broadcast_to(Lambda_cov, (T, C, K_cov))
        A_t = jnp.broadcast_to(A_static, (T, K_cov, K_cov))

        z_smooth, *_ = covariate_ssm.ssm.filter_and_smooth(covariates, Lambda_t, A_t)
        reconstructed = jax.vmap(lambda z: Lambda_cov @ z)(z_smooth)   # [T, C]
        return jnp.where(jnp.isnan(covariates), reconstructed, covariates)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def __call__(
        self,
        times: jnp.ndarray,
        covariates: jnp.ndarray,
        *,
        inference: bool = True,
        key: Optional[jax.Array] = None,
        coeffs: Optional[Any] = None,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Parameters
        ----------
        times : [T]
            Not used by the GRU directly but kept for interface consistency.
        covariates : [T, C_in]
            Raw covariate panel, possibly with trailing NaN rows for the
            information-horizon period (last h months missing).
        coeffs : Optional
            When ``imputation="ssm"``: pass a fitted :class:`~tvdfm.TVDFM`
            instance for the covariate panel here.
            Ignored for ``imputation="forward"``.

        Notes
        -----
        **Horizon-aware GRU scan (freeze-at-last-observation).**

        When the last ``h`` covariate rows are all-NaN (i.e. no monthly data
        is available for the nowcast horizon), the GRU hidden state is
        *frozen* at the last observed month instead of being updated with
        forward-filled stale values.  This means:

        * The GRU only processes observed months — no out-of-distribution
          repeated inputs.
        * ``delta_Lambda(t)`` for the missing months equals
          ``delta_Lambda(T - h)`` (constant loadings for the nowcast period).
        * Training with random horizon masking (``horizon_augment_max > 0``)
          is then distribution-matched to test time: the GRU always freezes
          at the last available month, whether in training or inference.

        With ``imputation="ssm"``, NaN filling is handled at the Python level
        in ``_run_forward`` before this call (see ``TVDFModel._impute_covariates_numpy``).
        The ``all_nan`` freeze mask is still computed from the original raw
        covariates so that ``imputation="forward"`` falls back correctly.
        """
        cov_raw = self._reduce(covariates)   # [T, n_covariates]

        # Only use _impute_ssm when coeffs is a genuine covariate DFM object
        # (has a .ssm attribute).  During training, coeffs may be diffrax
        # interpolation coefficients (a tuple) — those must not be used here.
        # With imputation="ssm" and _run_forward, NaN filling is done at the
        # Python level before JAX, so cov_raw arrives already clean.
        if self.imputation == "ssm" and coeffs is not None and hasattr(coeffs, "ssm"):
            cov = self._impute_ssm(cov_raw, covariate_ssm=coeffs)
        else:
            cov = forward_fill_nans(cov_raw)

        # Detect all-NaN time steps: no new monthly data available (horizon gap).
        # [T] bool — True  → freeze GRU hidden state (no data for this month)
        #            False → normal GRU update
        all_nan = jnp.all(jnp.isnan(cov_raw), axis=1)   # [T]

        def step(h, inputs):
            x_t, freeze = inputs
            h_new = self.gru_cell(x_t, h)
            # Keep previous hidden state when no new data is available.
            h_out = jnp.where(freeze, h, h_new)
            return h_out, h_out

        h0 = jnp.zeros(self.hidden_size, dtype=cov.dtype)
        _, hidden_states = jax.lax.scan(step, h0, (cov, all_nan))   # [T, H]

        if not inference and key is not None:
            # Variational dropout (Gal & Ghahramani, 2016): a single spatial mask
            # is shared across all T timesteps, which is more appropriate for
            # recurrent networks than independent per-step masking.
            mask = jax.random.bernoulli(key, 1 - self.dropout, (self.hidden_size,))
            hidden_states = hidden_states * (mask / (1 - self.dropout))

        T = hidden_states.shape[0]
        n_out = self.n_lambda_out
        delta_lambda_active = jax.vmap(self.readout)(hidden_states).reshape(
            T, n_out, self.n_factors
        )
        delta_A = jax.vmap(self.readout_A)(hidden_states).reshape(
            T, self.n_factors, self.n_factors
        )

        return self._scatter_lambda(delta_lambda_active, T), delta_A
