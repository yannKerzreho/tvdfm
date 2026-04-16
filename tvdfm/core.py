"""
Time-Varying Dynamic Factor Model (TVDFM).

Parameterisation
----------------
  Lambda(t) = Lambda_base + dLambda  [+ delta_Lambda(t)  if tv_Lambda]
  A_1(t)    = spectral_norm(A_base[:,:K] + dA[:,:K]  [+ delta_A(t)  if tv_A])
  A_j       = A_base[:,K(j-1):Kj] + dA[:,K(j-1):Kj]   for j = 2, …, p
  Q         = diag(exp(log_diag_Q))            [identity if fixed_Q=True]
  R         = diag(exp(log_diag_R))

Trainable leaves
----------------
  TVDFM.dLambda            [N, K]    static Λ correction  (when learn_Lambda=True)
  TVDFM.exposure.*                   all exposure model weights
  DFMStateSpace.dA         [K, K·p]  A correction (all p lags in one array)
  DFMStateSpace.log_diag_Q [K]       process noise  (frozen when fixed_Q=True)
  DFMStateSpace.log_diag_R [N]       observation noise
  DFMStateSpace.ar_Psi     [N, q]    unconstrained PARCOR params  (q=error_order)
  DFMStateSpace.log_diag_Sigma [N]   error innovation std  (empty when q=0)

Frozen leaves  (np.ndarray → invisible to JAX grad)
----------------------------------------------------
  DFMStateSpace.A_base    [K, K·p]
  DFMStateSpace.P0        [K, K]
  TVDFM.Lambda_base       [N, K]  (jnp, but labeled "other" by the trainer)
"""

import jax
import jax.numpy as jnp
import equinox as eqx
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from typing import List, Optional, Tuple, Any

from .ssm import DFMStateSpace, _spectral_norm
from .exposure import AbstractExposure
from .exposure.rnn import GRUExposure
from .utils import extract_statsmodels_params as _extract_statsmodels_params


class TVDFM(eqx.Module):
    """
    Time-Varying Dynamic Factor Model.

    See module docstring for the full parameterisation and trainability table.
    """

    Lambda_base: jnp.ndarray   # [N, K]  frozen anchor (labeled 'other' by trainer)
    dLambda: jnp.ndarray       # [N, K]  static Λ correction, init=0
    exposure: Optional[eqx.Module]
    ssm: DFMStateSpace

    n_series: int = eqx.field(static=True)
    n_factors: int = eqx.field(static=True)
    tv_Lambda: bool = eqx.field(static=True)
    tv_A: bool = eqx.field(static=True)
    learn_Lambda: bool = eqx.field(static=True)

    def __init__(
        self,
        Lambda_base: np.ndarray,
        ssm: DFMStateSpace,
        exposure: Optional[eqx.Module] = None,
        *,
        tv_Lambda: bool = True,
        tv_A: bool = False,
        learn_Lambda: bool = False,
    ):
        if (tv_Lambda or tv_A) and exposure is None:
            raise ValueError(
                "exposure must be provided when tv_Lambda=True or tv_A=True. "
                "Pass exposure=None only when both flags are False (static DFM)."
            )
        self.Lambda_base = jnp.asarray(Lambda_base, dtype=jnp.float32)
        self.dLambda     = jnp.zeros_like(self.Lambda_base)
        self.exposure    = exposure
        self.ssm         = ssm
        self.n_series, self.n_factors = self.Lambda_base.shape
        self.tv_Lambda   = tv_Lambda
        self.tv_A        = tv_A
        self.learn_Lambda = learn_Lambda

    # ------------------------------------------------------------------
    # Internal helper: build Lambda_t and A_t
    # ------------------------------------------------------------------

    def _build_Lambda_A(
        self,
        times: jnp.ndarray,
        covariates: jnp.ndarray,
        *,
        inference: bool = True,
        key: Optional[jax.Array] = None,
        coeffs: Optional[Any] = None,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Compute per-timestep loadings Lambda_t [T, N, K] and transition A_t [T, K, K].

        Runs the exposure model when tv_Lambda or tv_A is True.
        """
        T = times.shape[0]

        if self.tv_Lambda or self.tv_A:
            delta_Lambda, delta_A = self.exposure(
                times=times, covariates=covariates,
                inference=inference, key=key, coeffs=coeffs,
            )

        Lambda_eff = (
            self.Lambda_base + self.dLambda if self.learn_Lambda
            else self.Lambda_base
        )
        if self.tv_Lambda:
            Lambda_t = Lambda_eff[None] + delta_Lambda
        else:
            Lambda_t = jnp.broadcast_to(Lambda_eff, (T, self.n_series, self.n_factors))

        # A_1: spectral-normalised first-lag [K, K]
        A1_normed = self.ssm.A                                   # [K, K]
        if self.tv_A:
            K       = self.n_factors
            A_eff_1 = self.ssm.A_eff[:, :K]                     # [K, K]
            A_eff_t = A_eff_1[None] + delta_A                   # [T, K, K]
            A_t = jax.vmap(_spectral_norm, in_axes=(0, None))(
                A_eff_t, self.ssm.spectral_radius_bound
            )
        else:
            A_t = jnp.broadcast_to(A1_normed, (T, self.n_factors, self.n_factors))

        return Lambda_t, A_t

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def __call__(
        self,
        times: jnp.ndarray,          # [T]
        observations: jnp.ndarray,   # [T, N]   NaNs mark missing
        covariates: jnp.ndarray,     # [T, C]   pass zeros when exposure=None
        *,
        inference: bool = True,
        key: Optional[jax.Array] = None,
        coeffs: Optional[Any] = None,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, float]:
        """
        Full forward pass.

        Returns
        -------
        predictions    [T, N]   ŷ_t = Λ(t) ẑ_t
        filtered_states [T, K]  Kalman-filtered latent factors
        kf_ll          float    summed Kalman-filter log-likelihood
        """
        Lambda_t, A_t = self._build_Lambda_A(
            times, covariates, inference=inference, key=key, coeffs=coeffs
        )
        filtered_states, lls = self.ssm.filter(observations, Lambda_t, A_t)
        predictions = jax.vmap(lambda L, z: L @ z)(Lambda_t, filtered_states)
        return predictions, filtered_states, jnp.sum(lls)

    def compute_lambda_t(
        self,
        times: jnp.ndarray,
        covariates: jnp.ndarray,
        *,
        inference: bool = True,
        coeffs=None,
    ) -> jnp.ndarray:
        """Return Lambda_t [T, N, K] without running the Kalman filter."""
        Lambda_t, _ = self._build_Lambda_A(
            times, covariates, inference=inference, coeffs=coeffs
        )
        return Lambda_t

    def forward_smooth(
        self,
        times: jnp.ndarray,
        observations: jnp.ndarray,
        covariates: jnp.ndarray,
        *,
        inference: bool = True,
        key: Optional[jax.Array] = None,
        coeffs: Optional[Any] = None,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Forward pass + RTS smoother.

        Returns
        -------
        z_smooth [T, K]  RTS-smoothed latent factors
        z_filt   [T, K]  Kalman-filtered latent factors
        """
        Lambda_t, A_t = self._build_Lambda_A(
            times, covariates, inference=inference, key=key, coeffs=coeffs
        )
        z_smooth, _, z_filt, _, _ = self.ssm.filter_and_smooth(
            observations, Lambda_t, A_t
        )
        return z_smooth, z_filt

    # ------------------------------------------------------------------
    # Factory classmethods
    # ------------------------------------------------------------------

    @classmethod
    def from_params(
        cls,
        Lambda: np.ndarray,
        A: np.ndarray,
        Q: Optional[np.ndarray],
        R: Optional[np.ndarray],
        exposure: Optional[eqx.Module],
        key: jax.Array,
        *,
        tv_Lambda: bool = True,
        tv_A: bool = False,
        learn_Lambda: bool = False,
        fixed_Q: bool = False,
        spectral_radius_bound: float = 0.98,
        factor_order: int = 1,
        error_order: int = 0,
    ) -> "TVDFM":
        """
        Initialise from pre-fitted DFM parameter matrices.

        Parameters
        ----------
        Lambda : [N, K]           loadings
        A      : [K, K] or [K, K·p]  all lag coefficient matrices in one array.
                 Pass [K, K] for p=1; the SSM auto-pads zeros for p>1.
        Q      : [K, K] or [K] or None  process noise
        R      : [N, N] or [N] or None  observation noise diagonal
        fixed_Q : bool
            When True, Q is fixed to the identity.
        factor_order : int
            VAR lag order for the factor process.
        error_order : int
            AR order for idiosyncratic errors (0 = white noise).
        """
        n_series, n_factors = Lambda.shape
        ssm = DFMStateSpace(
            n_factors, n_series, key,
            A_init=A, Q_init=Q, R_init=R,
            spectral_radius_bound=spectral_radius_bound,
            fixed_Q=fixed_Q,
            factor_order=factor_order,
            error_order=error_order,
        )
        return cls(Lambda, ssm, exposure, tv_Lambda=tv_Lambda, tv_A=tv_A,
                   learn_Lambda=learn_Lambda)

    @classmethod
    def from_pca(
        cls,
        data: pd.DataFrame,
        n_factors: int,
        exposure: Optional[eqx.Module],
        key: jax.Array,
        *,
        tv_Lambda: bool = True,
        tv_A: bool = False,
        learn_Lambda: bool = False,
        fixed_Q: bool = True,
        spectral_radius_bound: float = 0.98,
        factor_order: int = 1,
        error_order: int = 0,
    ) -> "TVDFM":
        """
        Initialise Lambda_base from PCA on ``data``.

        A, Q, R are left at their defaults in :class:`DFMStateSpace`.
        ``data`` may contain NaNs; they are imputed with column means for PCA.
        """
        z = (data - data.mean()) / data.std()
        pca = PCA(n_components=n_factors)
        pca.fit(z.fillna(0))
        Lambda_base = pca.components_.T   # [N, K]

        return cls.from_params(
            Lambda_base, np.eye(n_factors) * 0.5, None, None,
            exposure, key,
            tv_Lambda=tv_Lambda, tv_A=tv_A, learn_Lambda=learn_Lambda,
            fixed_Q=fixed_Q, spectral_radius_bound=spectral_radius_bound,
            factor_order=factor_order, error_order=error_order,
        )

    @classmethod
    def from_statsmodels(
        cls,
        result: Any,
        data: pd.DataFrame,
        n_factors: int,
        exposure: Optional[eqx.Module],
        key: jax.Array,
        *,
        dfm_columns: Optional[List[str]] = None,
        tv_Lambda: bool = True,
        tv_A: bool = False,
        learn_Lambda: bool = False,
        fixed_Q: bool = False,
        spectral_radius_bound: float = 0.98,
        factor_order: int = 1,
        error_order: int = 0,
    ) -> "TVDFM":
        """
        Initialise from a fitted statsmodels DynamicFactor result.

        Parameters
        ----------
        dfm_columns : list of str or None
            Columns used during DFM fitting.  Supply when the DFM was fitted
            on a subset of ``data`` so Lambda and R are correctly mapped.
        fixed_Q : bool
            When True, Lambda_base is rescaled to absorb Q's diagonal variance.
        factor_order : int
            VAR lag order (must match the order used when fitting statsmodels).
        error_order : int
            0 = white noise, 1 = AR(1) idiosyncratic errors.
        """
        Lambda, A, Q, R = _extract_statsmodels_params(
            result, list(data.columns), n_factors, dfm_columns,
            factor_order=factor_order,
        )

        # ── Post-EM sanity guards ────────────────────────────────────────
        # (1) Clip degenerate Lambda rows.  For standardised data with K=3
        #     factors, row-norms should be O(1–10).  Row-norms above 200
        #     indicate EM divergence (explosive factor process or near-zero R).
        row_norms = np.linalg.norm(Lambda, axis=1, keepdims=True)
        max_row_norm = 200.0
        scale = np.maximum(row_norms / max_row_norm, 1.0)
        Lambda = Lambda / scale

        # (2) Floor the measurement noise R so no series can have variance
        #     so small that it drives Lambda to infinity.
        r_diag = np.diag(R)
        r_diag = np.maximum(r_diag, 1e-3)
        R = np.diag(r_diag)
        # ────────────────────────────────────────────────────────────────

        if fixed_Q:
            q_diag = np.diag(Q) if Q.ndim == 2 else Q
            Lambda = Lambda * np.sqrt(np.clip(q_diag, 1e-8, None))

        return cls.from_params(
            Lambda, A, Q, R, exposure, key,
            tv_Lambda=tv_Lambda, tv_A=tv_A, learn_Lambda=learn_Lambda,
            fixed_Q=fixed_Q, spectral_radius_bound=spectral_radius_bound,
            factor_order=factor_order, error_order=error_order,
        )

    @classmethod
    def fit_and_init(
        cls,
        data: pd.DataFrame,
        n_factors: int,
        exposure: Optional[eqx.Module],
        key: jax.Array,
        *,
        dfm_columns: Optional[List[str]] = None,
        factor_order: int = 1,
        em_iter: int = 100,
        tv_Lambda: bool = True,
        tv_A: bool = False,
        learn_Lambda: bool = False,
        fixed_Q: bool = False,
        spectral_radius_bound: float = 0.98,
        error_order: int = 0,
    ) -> "TVDFM":
        """
        Fit a statsmodels DynamicFactor model via EM, then initialise TVDFM.

        This is the primary entry point when no pre-fitted DFM is available.

        Parameters
        ----------
        data : pd.DataFrame [T, N]
            Full dataset (NaNs allowed; EM handles missing data).
        dfm_columns : list of str or None
            Subset of columns to use for DFM fitting.
        factor_order : int  (default 1)
            AR order for the factor process.
        em_iter : int  (default 100)
            Maximum EM iterations.
        error_order : int
            0 = white noise, 1 = AR(1) idiosyncratic errors.
        """
        from statsmodels.tsa.statespace.dynamic_factor import DynamicFactor

        # Guard: GRUExposure processes every row — daily data would be very slow
        # and is almost certainly a mistake.  Use NCDEExposure for daily inputs.
        if isinstance(exposure, GRUExposure) and isinstance(data.index, pd.DatetimeIndex):
            inferred = pd.infer_freq(data.index)
            if inferred is not None and inferred.upper().startswith(("D", "B")):
                raise ValueError(
                    "GRUExposure received daily data (inferred freq: "
                    f"'{inferred}').  GRU processes every row, which is "
                    "extremely slow at daily frequency.  Either aggregate "
                    "your DataFrame to monthly frequency first, or use "
                    "NCDEExposure which handles continuous-time paths."
                )

        fit_cols = dfm_columns if dfm_columns is not None else list(data.columns)
        fit_data = data[fit_cols]
        fit_std = (fit_data - fit_data.mean()) / fit_data.std()

        dfm = DynamicFactor(
            fit_std.ffill().fillna(0),
            k_factors=n_factors,
            factor_order=factor_order,
            enforce_stationarity=False,
        )
        result = dfm.fit(maxiter=em_iter, disp=False)

        return cls.from_statsmodels(
            result, data, n_factors, exposure, key,
            dfm_columns=fit_cols,
            tv_Lambda=tv_Lambda, tv_A=tv_A, learn_Lambda=learn_Lambda,
            fixed_Q=fixed_Q, spectral_radius_bound=spectral_radius_bound,
            factor_order=factor_order, error_order=error_order,
        )


    @classmethod
    def from_preselected(
        cls,
        data: pd.DataFrame,
        target_col: str,
        n_factors: int,
        exposure: Optional[eqx.Module],
        key: jax.Array,
        *,
        n_select: int = 20,
        presel_method: str = "iterative_pca",
        lag_x: int = 1,
        em_iter: int = 100,
        factor_order: int = 1,
        error_order: int = 0,
        tv_Lambda: bool = True,
        tv_A: bool = False,
        learn_Lambda: bool = False,
        fixed_Q: bool = True,
        spectral_radius_bound: float = 0.98,
    ) -> "TVDFM":
        """
        Preselect variables, fit DFM via EM, then initialise TVDFM.

        This is the recommended entry point for high-dimensional panels
        (N > 30) where fitting DFM EM on all variables is too costly.

        Parameters
        ----------
        data : pd.DataFrame [T, N]
            Full mixed-frequency panel.  May contain NaNs.
        target_col : str
            Name of the target variable (e.g. 'GDPC1').
            Excluded from covariate preselection; placed first in the DFM.
        n_select : int
            Number of covariates to pass to the DFM (default 20).
            Rule of thumb: n_select >= 5 * n_factors, <= 30 for EM speed.
        presel_method : str
            ``'iterative_pca'`` (recommended) or ``'top_corr'``.
        lag_x : int
            Lag applied to X before computing association with target
            (pseudo real-time).

        Returns
        -------
        TVDFM
            Instance initialised from the fitted DFM.  ``n_series`` equals
            the number of columns in ``data`` (full panel); Lambda rows for
            series not in the DFM subset are initialised to zero and learned
            during SGD fine-tuning.

        Notes
        -----
        The model is trained **only** on ``[target_col] + selected_cols``
        (1 + n_select series).  The full panel is never seen after preselection,
        so Lambda has shape ``(1 + n_select, n_factors)`` (not N_full).
        At inference the caller is responsible for slicing the input to the
        same column set (TVDFModel does this automatically via
        ``selected_columns_``).
        """
        from .preselection import preselect

        X = data.drop(columns=[target_col])
        y = data[target_col]

        selected_cols, _ = preselect(
            X, y, n_factors, n_select,
            method=presel_method, lag_x=lag_x, exclude=None,
        )

        # Model trains on target + selected covariates ONLY (reduced panel)
        dfm_cols    = [target_col] + selected_cols
        data_reduced = data[dfm_cols]

        tvdfm = cls.fit_and_init(
            data_reduced, n_factors, exposure, key,
            dfm_columns=None,       # all columns of data_reduced are used for EM
            factor_order=factor_order,
            em_iter=em_iter,
            tv_Lambda=tv_Lambda, tv_A=tv_A, learn_Lambda=learn_Lambda,
            fixed_Q=fixed_Q, spectral_radius_bound=spectral_radius_bound,
            error_order=error_order,
        )
        # Return selected monthly columns so the caller can store them
        return tvdfm, selected_cols
