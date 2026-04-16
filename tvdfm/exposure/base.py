import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from sklearn.decomposition import PCA
from typing import Optional, Tuple
from .utils import forward_fill_nans, make_ncde_path


class AbstractExposure(eqx.Module):
    """
    Abstract base class for all LTV exposure models.

    Maps a covariate path to time-varying DFM parameter perturbations:

        (times [T], covariates [T, C_in])
            -> (delta_Lambda [T, N, K], delta_A [T, K, K])

    Mixed-frequency workflow
    ------------------------
    Pre-process covariates **once** before training using :func:`make_ncde_path`
    (for NCDE models) or :func:`forward_fill_nans` (for RNN models).  Pass the
    resulting clean array as ``covariates``, or pre-compute interpolation
    coefficients and pass them as ``coeffs``:

        path   = make_ncde_path(times, raw_covariates, interpolation="cubic")
        coeffs = diffrax.backward_hermite_coefficients(times_jax, jnp.array(path))
        model(times_jax, obs, jnp.array(path), coeffs=coeffs)  # no fill inside model

    Parameters
    ----------
    n_series : int  N
    n_factors : int  K
    n_covariates : int
        Covariate dimension entering the backbone **after** the optional reducer.
    lambda_indices : Optional[Tuple[int, ...]]
        Which rows of delta_Lambda are actively predicted.
        - ``None``          → all N series (former "symmetric")
        - ``(target_idx,)`` → target series only (former "asymmetric")
        - any subset        → new generalisation
    input_reducer : Optional[eqx.nn.Linear]
        Linear projection [C_in] -> [n_covariates] applied to raw covariates.
        Build from PCA::

            reducer = AbstractExposure.make_pca_reducer(X_train, 10, key)
            model   = NCDEExposure(..., n_covariates=10, input_reducer=reducer)
    """

    n_series: int = eqx.field(static=True)
    n_factors: int = eqx.field(static=True)
    n_covariates: int = eqx.field(static=True)
    lambda_indices: Optional[Tuple[int, ...]] = eqx.field(static=True)
    input_reducer: Optional[eqx.nn.Linear]

    # ------------------------------------------------------------------
    # Static factory: PCA-based input reducer
    # ------------------------------------------------------------------

    @staticmethod
    def make_pca_reducer(
        X: np.ndarray,
        n_components: int,
        key: jax.Array,
        use_bias: bool = False,
    ) -> eqx.nn.Linear:
        """
        Fit PCA on ``X [T, C_in]`` and return an ``eqx.nn.Linear`` whose
        weight matrix is the first ``n_components`` principal components.

        NaNs in X are column-mean imputed before the PCA fit.

            reducer = AbstractExposure.make_pca_reducer(X_train, 10, key)
            model   = NCDEExposure(n_covariates=10, ..., input_reducer=reducer)
        """
        X_filled = np.where(np.isnan(X), np.nanmean(X, axis=0, keepdims=True), X)
        pca = PCA(n_components=n_components)
        pca.fit(X_filled)
        components = jnp.array(pca.components_, dtype=jnp.float32)  # [n_components, C_in]

        layer = eqx.nn.Linear(X.shape[1], n_components, use_bias=use_bias, key=key)
        return eqx.tree_at(lambda l: l.weight, layer, components)

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    @property
    def n_lambda_out(self) -> int:
        return self.n_series if self.lambda_indices is None else len(self.lambda_indices)

    def _reduce(self, covariates: jnp.ndarray) -> jnp.ndarray:
        """Project [T, C_in] -> [T, n_covariates]. No-op when input_reducer is None."""
        if self.input_reducer is None:
            return covariates
        return jax.vmap(self.input_reducer)(covariates)

    def _scatter_lambda(self, delta_lambda_active: jnp.ndarray, T: int) -> jnp.ndarray:
        """Scatter [T, n_lambda_out, K] -> [T, N, K]. No-op when lambda_indices is None."""
        if self.lambda_indices is None:
            return delta_lambda_active
        full = jnp.zeros((T, self.n_series, self.n_factors))
        return full.at[:, jnp.array(self.lambda_indices), :].set(delta_lambda_active)

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    def __call__(
        self,
        times: jnp.ndarray,
        covariates: jnp.ndarray,
        *,
        inference: bool = True,
        key: Optional[jax.Array] = None,
        coeffs: Optional[Tuple] = None,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Parameters
        ----------
        times : [T]
        covariates : [T, C_in]
            Should be a clean path (no interior NaNs for NCDE).
            Pre-process with :func:`make_ncde_path` before calling.
        inference : bool
        key : Optional[jax.Array]
        coeffs : Optional[Tuple]
            Pre-computed interpolation artefacts (avoids recomputation).

        Returns
        -------
        delta_Lambda : [T, N, K]
        delta_A      : [T, K, K]
        """
        raise NotImplementedError
