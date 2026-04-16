"""
High-level sklearn-compatible interface for the Time-Varying Dynamic Factor Model.
"""
from __future__ import annotations

from typing import List, Optional, Tuple, Union

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

from .core import TVDFM
from .exposure import AbstractExposure, GRUExposure, NCDEExposure, make_ncde_path
from .training import LTVTrainingManager, _get_coeffs
from .utils import parse_dataframe, parse_covariates, to_float_times, _spectral_norm

_ACTIVATIONS = {
    "tanh": jax.nn.tanh,
    "relu": jax.nn.relu,
    "gelu": jax.nn.gelu,
    "silu": jax.nn.silu,
}


class TVDFModel:
    """
    Time-Varying Dynamic Factor Model — sklearn-compatible estimator.

    This is the main entry point for using TVDFM in downstream projects.
    Internally wraps :class:`~tvdfm.TVDFM` (the Equinox module) and
    :class:`~tvdfm.LTVTrainingManager`.

    Quick start
    -----------
    Static DFM (no time variation, EM-initialized)::

        model = TVDFModel(n_factors=3, exposure=None)
        model.fit(df_obs)                 # df_obs: pd.DataFrame [T, N]
        preds   = model.predict()         # ndarray [T, N]
        factors = model.transform()       # ndarray [T, K]

    NCDE exposure (time-varying loadings driven by covariates)::

        model = TVDFModel(n_factors=3, exposure="ncde")
        model.fit(df_obs, covariates=df_cov)

    Parameters
    ----------
    n_factors : int
        Number of latent factors K.
    exposure : {"ncde", "gru", None}
        Exposure model type.  ``None`` → static DFM (no time variation).
    tv_Lambda : bool
        Enable time-varying loadings Λ(t).  Requires ``exposure != None``.
    tv_A : bool
        Enable time-varying transition matrix A(t).  Requires ``exposure != None``.
    learn_Lambda : bool
        Include a trainable static correction dΛ [N, K] in the loadings.
    fixed_Q : bool
        Fix process noise Q = I.  When True, ``Lambda_base`` columns absorb
        Q's variance (scale ∝ √Q_kk).
    factor_order : int
        VAR lag order for the factor process (default 1 = AR(1)).
        Initialised from the statsmodels companion matrix when ``dfm_init="em"``.
    error_order : int
        0 = white-noise idiosyncratic errors (default).
        1 = AR(1) idiosyncratic errors (extra N states added to the SSM).
    hidden_size : int
        Hidden state size for the exposure backbone (NCDE / GRU).
    mlp_width : int
        Width of each hidden layer in the NCDE vector-field MLP.
    mlp_depth : int
        Depth of the NCDE vector-field MLP.
    dropout : float
        Dropout rate applied to the hidden state during training.
    activation : {"tanh", "relu", "gelu", "silu"}
        Activation function for the NCDE MLP.
    interpolation : {"cubic", "linear", "rectilinear"}
        Covariate interpolation scheme for the NCDE control path.
    imputation : {"forward", "ssm"}
        Missing-value strategy for GRU covariates.
    em_val_n : int, optional
        Number of time steps withheld from the EM initialisation.  When set,
        overrides ``val_n`` for the EM / preselection / covariate-DFM step only.
        - ``None`` (default): use ``val_n`` (same window as GRU training).
        - ``0``: EM sees the full training set including the validation window.
          Combined with ``val_n > 0``, this gives a better-initialised DFM
          anchor while the GRU is still trained and validated on strict windows.
          Note: the DFM anchor then encodes validation-window information, which
          may influence the GRU validation baseline (mild potential for leakage).
    lambda_indices : None, int, str, or list
        Which series rows of delta_Λ are non-zero.
        ``None`` → all N series.
    n_pca_covariates : int, optional
        Reduce covariates to this many dimensions via PCA.
    dfm_init : {"em", "pca", "preselected"}
        How to initialize Λ, A, Q, R.
        ``'preselected'`` automatically selects ``n_select`` informative
        covariates before the EM step; requires ``target_col`` to be set.
    dfm_columns : list of str, optional
        Subset of ``X`` columns used for the initial DFM fit.
        Ignored when ``dfm_init='preselected'``.
    target_col : str, optional
        Name of the target series.  Required when ``dfm_init='preselected'``.
        This series is excluded from covariate preselection and placed first
        in the DFM.
    n_select : int
        Number of covariates selected by the preselection step (default 20).
        Effective only when ``dfm_init='preselected'``.
    presel_method : {"iterative_pca", "top_corr"}
        Preselection algorithm (default ``'iterative_pca'``).
    lag_x : int
        Lag applied to covariates before measuring association with the target
        (pseudo real-time; default 1).
    em_iter : int
        Maximum EM iterations for statsmodels initialisation.
    spectral_radius_bound : float
        Upper bound on the spectral radius of A (default 0.98).
    lr_ssm : float
        Learning rate for SSM parameters.
    lr_exposure : float
        Learning rate for the exposure model.
    wd_exposure : float
        L2 weight decay on exposure parameters (dynamic, no JIT recompile).
    wd_ssm : float
        L2 weight decay on SSM parameters (dynamic, no JIT recompile).
    kf_ll_weight : float
        Weight on the Kalman-filter log-likelihood term in the loss.
    lambda_dev_weight : float
        Weight on the time-varying loading deviation penalty
        ``mean_t ‖δΛ(t)‖_F²``.  Penalises drift of the neural loadings away
        from the EM anchor.  Default 0 (disabled).
    n_epochs : int
        Maximum training epochs.
    patience : int
        Early-stopping patience (epochs without validation improvement).
    val_n : int
        Number of time steps reserved for validation (chronological tail).
    random_state : int, optional
        Seed for JAX PRNG.
    verbose : bool
        Print training progress every 10 epochs.
    """

    def __init__(
        self,
        n_factors: int = 2,
        *,
        # Exposure model
        exposure: Optional[str] = "ncde",
        tv_Lambda: bool = True,
        tv_A: bool = False,
        learn_Lambda: bool = False,
        fixed_Q: bool = False,
        # State-space structure
        factor_order: int = 1,
        error_order: int = 0,
        # Backbone architecture
        hidden_size: int = 16,
        mlp_width: int = 64,
        mlp_depth: int = 2,
        dropout: float = 0.1,
        activation: str = "tanh",
        interpolation: str = "cubic",
        imputation: str = "forward",
        em_val_n: Optional[int] = None,
        lambda_indices=None,
        n_pca_covariates: Optional[int] = None,
        # DFM initialization
        dfm_init: str = "em",
        dfm_columns: Optional[List[str]] = None,
        target_col: Optional[str] = None,
        n_select: int = 20,
        presel_method: str = "iterative_pca",
        lag_x: int = 1,
        em_iter: int = 100,
        spectral_radius_bound: float = 0.98,
        # Training
        lr_ssm: float = 1e-3,
        lr_exposure: float = 1e-3,
        wd_exposure: float = 0.1,
        wd_ssm: float = 0.1,
        kf_ll_weight: float = 0.05,
        lambda_dev_weight: float = 0.0,
        val_target_series=None,
        horizon_augment_max: int = 0,
        val_horizon: int = 0,
        val_full_kf: bool = False,
        n_epochs: int = 500,
        patience: int = 20,
        val_n: int = 20,
        random_state: Optional[int] = None,
        verbose: bool = False,
    ):
        self.n_factors             = n_factors
        self.exposure              = exposure
        self.tv_Lambda             = tv_Lambda
        self.tv_A                  = tv_A
        self.learn_Lambda          = learn_Lambda
        self.fixed_Q               = fixed_Q
        self.factor_order          = factor_order
        self.error_order           = error_order
        self.hidden_size           = hidden_size
        self.mlp_width             = mlp_width
        self.mlp_depth             = mlp_depth
        self.dropout               = dropout
        self.activation            = activation
        self.interpolation         = interpolation
        self.imputation            = imputation
        self.em_val_n              = em_val_n
        self.n_factors_cov         = n_factors          # covariate DFM uses same K
        self.lambda_indices        = lambda_indices
        self.n_pca_covariates      = n_pca_covariates
        self.dfm_init              = dfm_init
        self.dfm_columns           = dfm_columns
        self.target_col            = target_col
        self.n_select              = n_select
        self.presel_method         = presel_method
        self.lag_x                 = lag_x
        self.em_iter               = em_iter
        self.spectral_radius_bound = spectral_radius_bound
        self.lr_ssm                = lr_ssm
        self.lr_exposure           = lr_exposure
        self.wd_exposure           = wd_exposure
        self.wd_ssm                = wd_ssm
        self.kf_ll_weight          = kf_ll_weight
        self.lambda_dev_weight       = lambda_dev_weight
        self.horizon_augment_max     = horizon_augment_max
        self.val_horizon             = val_horizon
        self.val_full_kf             = val_full_kf
        self.val_target_series     = val_target_series
        self.n_epochs              = n_epochs
        self.patience              = patience
        self.val_n                 = val_n
        self.random_state          = random_state
        self.verbose               = verbose

    # -------------------------------------------------------------------
    # Private helpers
    # -------------------------------------------------------------------

    def _make_key(self) -> jax.Array:
        return jax.random.PRNGKey(self.random_state if self.random_state is not None else 0)

    def _resolve_lambda_indices(
        self, columns: List[str]
    ) -> Optional[Tuple[int, ...]]:
        idx = self.lambda_indices
        if idx is None:
            return None
        if isinstance(idx, (int, str)):
            idx = [idx]
        return tuple(
            columns.index(i) if isinstance(i, str) else int(i)
            for i in idx
        )

    def _resolve_target_mask(
        self, target_series, target_mask, T: int, columns: List[str]
    ) -> Optional[np.ndarray]:
        if target_mask is not None:
            return np.asarray(target_mask, dtype=np.float32)
        if target_series is None:
            return None
        if isinstance(target_series, (str, int)):
            target_series = [target_series]
        mask = np.zeros((T, len(columns)), dtype=np.float32)
        for s in target_series:
            idx = columns.index(s) if isinstance(s, str) else int(s)
            mask[:, idx] = 1.0
        return mask

    def _build_exposure_model(
        self,
        n_cov_in: int,
        n_series: int,
        lambda_indices_res: Optional[Tuple[int, ...]],
        cov_np: Optional[np.ndarray],
        key: jax.Array,
    ) -> Tuple[Optional[eqx.Module], jax.Array]:
        """Build the exposure module (or None for static DFM)."""
        if self.exposure is None:
            return None, key

        input_reducer = None
        n_cov = n_cov_in
        if self.n_pca_covariates is not None and self.n_pca_covariates < n_cov_in:
            key, subkey = jax.random.split(key)
            input_reducer = AbstractExposure.make_pca_reducer(
                cov_np, self.n_pca_covariates, subkey
            )
            n_cov = self.n_pca_covariates

        act = _ACTIVATIONS.get(self.activation, jax.nn.tanh)
        key, subkey = jax.random.split(key)

        if self.exposure == "ncde":
            exp_model = NCDEExposure(
                n_covariates=n_cov,
                n_series=n_series,
                n_factors=self.n_factors,
                key=subkey,
                hidden_size=self.hidden_size,
                mlp_width=self.mlp_width,
                mlp_depth=self.mlp_depth,
                dropout=self.dropout,
                activation=act,
                lambda_indices=lambda_indices_res,
                interpolation=self.interpolation,
                input_reducer=input_reducer,
            )
        elif self.exposure == "gru":
            exp_model = GRUExposure(
                n_covariates=n_cov,
                n_series=n_series,
                n_factors=self.n_factors,
                key=subkey,
                hidden_size=self.hidden_size,
                dropout=self.dropout,
                lambda_indices=lambda_indices_res,
                imputation=self.imputation,
                input_reducer=input_reducer,
            )
        else:
            raise ValueError(
                f"Unknown exposure: {self.exposure!r}. Use 'ncde', 'gru', or None."
            )

        return exp_model, key

    def _preprocess_cov(
        self, times_np: np.ndarray, cov_np: Optional[np.ndarray]
    ) -> Optional[np.ndarray]:
        """Return NaN-free covariate array (NCDE applies make_ncde_path)."""
        if cov_np is None:
            return None
        if self.exposure == "ncde":
            return make_ncde_path(times_np, cov_np, self.interpolation)
        return cov_np  # GRU handles NaNs internally

    def _setup(
        self,
        X,
        covariates,
        times,
        target_series,
        target_mask,
        val_target_series=None,
    ):
        """
        Parse inputs, build TVDFM, preprocess covariates.

        Returns
        -------
        X_np, times_np, columns, cov_np, cov_clean, tmask, val_tmask, tvdfm
        """
        X_np, times_np, columns = parse_dataframe(X, times)
        cov_np = parse_covariates(covariates)
        T, N   = X_np.shape

        if (self.exposure is not None and cov_np is None
                and self.dfm_init != "preselected"):
            raise ValueError(
                f"covariates must be provided when using an exposure model "
                f"(exposure={self.exposure!r}).  Alternatively, set "
                f"dfm_init='preselected' and the model will auto-build "
                f"covariates from the selected monthly series."
            )

        # T_em is the number of rows passed to statsmodels EM,
        # preselection, and PCA reducer fitting.
        # em_val_n overrides val_n for the EM step only (e.g. em_val_n=0 lets
        # the EM see the full training set while the GRU val window stays val_n).
        em_cutoff = self.em_val_n if self.em_val_n is not None else self.val_n
        T_em = max(T - em_cutoff, 1) if em_cutoff > 0 else T

        n_cov_in           = cov_np.shape[1] if cov_np is not None else 0
        lambda_indices_res = self._resolve_lambda_indices(columns)
        tmask              = self._resolve_target_mask(target_series, target_mask, T, columns)

        key = self._make_key()
        # PCA reducer fitted on train-only portion to avoid val leakage.
        cov_for_pca = cov_np[:T_em] if cov_np is not None else None
        exp_model, key = self._build_exposure_model(
            n_cov_in, N, lambda_indices_res, cov_for_pca, key
        )

        # tv flags are silently forced off when no exposure model
        tv_Lambda = self.tv_Lambda and self.exposure is not None
        tv_A      = self.tv_A      and self.exposure is not None

        df_X    = pd.DataFrame(X_np, columns=columns)
        df_X_em = df_X.iloc[:T_em]          # train-only slice for EM / preselection
        if self.dfm_init == "em":
            tvdfm = TVDFM.fit_and_init(
                df_X_em, self.n_factors, exp_model, key,
                dfm_columns=self.dfm_columns,
                factor_order=self.factor_order,
                error_order=self.error_order,
                em_iter=self.em_iter,
                tv_Lambda=tv_Lambda, tv_A=tv_A,
                learn_Lambda=self.learn_Lambda,
                fixed_Q=self.fixed_Q,
                spectral_radius_bound=self.spectral_radius_bound,
            )
        elif self.dfm_init == "preselected":
            if self.target_col is None:
                raise ValueError(
                    "dfm_init='preselected' requires target_col to be set."
                )
            if self.target_col not in columns:
                raise ValueError(
                    f"target_col={self.target_col!r} not found in X columns."
                )
            # ── Step 1: preselect to know exact column set before building
            #            the exposure model (n_cov = n_select, not N_full).
            # Preselection uses only the EM training portion so that the
            # selected series are not influenced by the validation window.
            from .preselection import preselect as _preselect
            _X_full = df_X_em.drop(columns=[self.target_col])
            _y      = df_X_em[self.target_col]
            _sel, _ = _preselect(
                _X_full, _y, self.n_factors, self.n_select,
                method=self.presel_method, lag_x=self.lag_x,
            )
            sel_cols = [self.target_col] + _sel   # target first, then monthly

            # ── Step 2: slice observation matrix to selected columns only.
            # Full T rows kept so the training manager sees the complete series.
            col_idx  = [list(columns).index(c) for c in sel_cols]
            X_np     = X_np[:, col_idx].astype(np.float32)
            columns  = sel_cols
            N        = len(sel_cols)
            T        = X_np.shape[0]
            df_X     = pd.DataFrame(X_np, columns=columns)
            df_X_em  = df_X.iloc[:T_em]    # re-slice after column selection

            # ── Step 3: build covariates for the exposure model.
            # If the caller did not provide explicit covariates, auto-build from
            # the selected monthly columns (target excluded).
            mon_idx = [sel_cols.index(c) for c in _sel]
            if cov_np is None and self.exposure is not None:
                cov_np = X_np[:, mon_idx].astype(np.float32)
            n_cov_in           = cov_np.shape[1] if cov_np is not None else 0
            lambda_indices_res = self._resolve_lambda_indices(columns)

            # ── Step 4: rebuild exposure model with reduced dimensions.
            # PCA reducer fitted on train-only portion (cov[:T_em]).
            cov_for_pca = cov_np[:T_em] if cov_np is not None else None
            exp_model, key = self._build_exposure_model(
                n_cov_in, N, lambda_indices_res, cov_for_pca, key
            )
            tmask = self._resolve_target_mask(target_series, target_mask, T, columns)

            # ── Step 5: init TVDFM on the train-only slice (df_X_em).
            # from_preselected receives an already-reduced df_X_em and will
            # run preselection internally once more (cheap, same result).
            tvdfm, _ = TVDFM.from_preselected(
                df_X_em, self.target_col, self.n_factors, exp_model, key,
                n_select=self.n_select,
                presel_method=self.presel_method,
                lag_x=self.lag_x,
                factor_order=self.factor_order,
                error_order=self.error_order,
                em_iter=self.em_iter,
                tv_Lambda=tv_Lambda, tv_A=tv_A,
                learn_Lambda=self.learn_Lambda,
                fixed_Q=self.fixed_Q,
                spectral_radius_bound=self.spectral_radius_bound,
            )
            # Store selected columns so predict() / transform() can auto-slice.
            self.selected_columns_ = sel_cols   # [target] + selected monthly
            self.selected_monthly_ = _sel       # selected monthly only
        else:  # "pca"
            tvdfm = TVDFM.from_pca(
                df_X_em, self.n_factors, exp_model, key,
                tv_Lambda=tv_Lambda, tv_A=tv_A,
                learn_Lambda=self.learn_Lambda,
                fixed_Q=self.fixed_Q,
                spectral_radius_bound=self.spectral_radius_bound,
                factor_order=self.factor_order,
                error_order=self.error_order,
            )

        # ── Covariate DFM for SSM imputation ──────────────────────────────
        # When imputation="ssm", fit a separate static DFM on the monthly
        # covariates (train-only slice). At inference time, its KF one-step
        # predictions replace NaN covariate entries so the GRU continues to
        # receive model-predicted inputs instead of freezing at last observation.
        if self.imputation == "ssm" and cov_np is not None and self.exposure is not None:
            cov_for_ssm = cov_np[:T_em]  # train-only (or full if em_val_n=0)
            cov_cols     = [f"cov_{i}" for i in range(cov_for_ssm.shape[1])]
            df_cov_em    = pd.DataFrame(cov_for_ssm, columns=cov_cols)
            df_cov_em    = df_cov_em.fillna(df_cov_em.mean())
            cov_key      = jax.random.fold_in(key, 9999)
            # Retry with progressively tighter spectral radius if EM produces
            # a near-unit-root A that causes Schur decomposition failures.
            # Final fallback: covariate_model_=None (reverts to freeze behaviour).
            self.covariate_model_ = None
            for srb in (self.spectral_radius_bound, 0.90, 0.80):
                try:
                    self.covariate_model_ = TVDFM.fit_and_init(
                        df_cov_em,
                        self.n_factors_cov,
                        None,
                        cov_key,
                        factor_order=self.factor_order,
                        error_order=self.error_order,
                        em_iter=self.em_iter,
                        tv_Lambda=False,
                        tv_A=False,
                        learn_Lambda=False,
                        fixed_Q=False,
                        spectral_radius_bound=srb,
                    )
                    break   # success
                except Exception:
                    continue
            if self.covariate_model_ is None:
                import warnings as _w
                _w.warn("Covariate DFM fitting failed at all spectral-radius bounds; "
                        "falling back to freeze imputation for this window.")
        elif self.imputation == "ssm" and not hasattr(self, "covariate_model_"):
            self.covariate_model_ = None

        cov_clean = self._preprocess_cov(times_np, cov_np)
        # Static DFM: pass a dummy zero covariate (unused by the model)
        if cov_clean is None:
            cov_clean = np.zeros((T, 1), dtype=np.float32)

        # Compute val target mask for early stopping (may differ from train mask).
        # val_target_series=None means reuse the same mask as training.
        _val_ts = val_target_series  # local alias
        if _val_ts is None:
            val_tmask = None   # caller uses tmask for both train and val
        else:
            val_tmask = self._resolve_target_mask(_val_ts, None, T, list(columns))

        return X_np, times_np, columns, cov_np, cov_clean, tmask, val_tmask, tvdfm

    # ------------------------------------------------------------------
    # Column-selection helper (used by inference methods)
    # ------------------------------------------------------------------

    def _slice_to_selected(
        self,
        X: "pd.DataFrame",
        covariates=None,
    ):
        """
        When the model was fitted with ``dfm_init='preselected'``, slice an
        input DataFrame to the stored column set and, if no explicit covariates
        are provided, auto-build them from the selected monthly columns.

        Returns ``(X_sliced, covariates_or_None)`` ready to pass to
        ``parse_dataframe`` / ``parse_covariates``.
        """
        sel = getattr(self, "selected_columns_", None)
        if sel is None:
            return X, covariates          # no preselection: pass through

        if not isinstance(X, pd.DataFrame):
            return X, covariates          # cannot slice non-DataFrame inputs

        missing = [c for c in sel if c not in X.columns]
        if missing:
            raise ValueError(
                f"Input is missing columns required by the preselected model: "
                f"{missing}.  Pass the full panel so the model can slice it."
            )

        X_sliced = X[sel]

        if covariates is None and self.exposure is not None:
            mon = getattr(self, "selected_monthly_", None)
            if mon is not None:
                covariates = X[mon]       # selected monthly series, no target

        return X_sliced, covariates

    def _run_forward(self, times_np, X_np, cov_np):
        """Full TVDFM forward pass. Returns (predictions, factors, kf_ll)."""
        times_j = jnp.asarray(times_np)
        X_j     = jnp.asarray(X_np)

        cov_clean = self._preprocess_cov(times_np, cov_np)
        if cov_clean is None:
            cov_clean = np.zeros((len(times_np), 1), dtype=np.float32)
        # SSM imputation: fill NaN covariate entries with Kalman-smoother
        # predictions from the pre-fitted covariate DFM, BEFORE handing to JAX.
        # This avoids passing a Python object through eqx.filter_jit.
        if self.imputation == "ssm":
            cov_clean = self._impute_covariates_numpy(cov_clean)

        cov_j  = jnp.asarray(cov_clean)
        coeffs = _get_coeffs(times_j, cov_j, self.interpolation) \
                 if self.exposure == "ncde" else None

        return self.model_(times_j, X_j, cov_j, inference=True, coeffs=coeffs)

    def _run_smooth(self, times_np, X_np, cov_np):
        """
        Forward pass + RTS smoother.

        Returns (z_smooth [T, K], z_filt [T, K]) as numpy arrays.
        """
        times_j = jnp.asarray(times_np)
        X_j     = jnp.asarray(X_np)

        cov_clean = self._preprocess_cov(times_np, cov_np)
        if cov_clean is None:
            cov_clean = np.zeros((len(times_np), 1), dtype=np.float32)
        cov_j  = jnp.asarray(cov_clean)
        coeffs = _get_coeffs(times_j, cov_j, self.interpolation) \
                 if self.exposure == "ncde" else None

        z_smooth, z_filt = self.model_.forward_smooth(
            times_j, X_j, cov_j, coeffs=coeffs
        )
        return np.asarray(z_smooth), np.asarray(z_filt)

    def _impute_covariates_numpy(self, cov_np: np.ndarray) -> np.ndarray:
        """
        Replace NaN entries in ``cov_np`` with Kalman-smoother reconstructions
        from the pre-fitted covariate DFM (``self.covariate_model_``).

        Called at inference time when ``imputation="ssm"``.  The imputation is
        done at the Python / NumPy level BEFORE passing covariates into JAX,
        so it never enters a JIT-compiled context and there are no tracing
        complications.

        Parameters
        ----------
        cov_np : ndarray [T, C]  — raw covariates, NaN where data is missing.

        Returns
        -------
        ndarray [T, C]  — same array with NaN entries replaced by DFM predictions.
        """
        cov_model = getattr(self, "covariate_model_", None)
        if cov_model is None or not np.any(np.isnan(cov_np)):
            return cov_np

        T, C   = cov_np.shape
        K_cov  = cov_model.ssm.n_factors
        Lam    = jnp.asarray(cov_model.Lambda_base)          # [C, K]
        Lambda_t = jnp.broadcast_to(Lam, (T, C, K_cov))     # [T, C, K]
        A_t      = jnp.broadcast_to(
            jnp.asarray(cov_model.ssm.A), (T, K_cov, K_cov)  # [T, K, K]
        )
        # KF smoother handles NaN observations as missing values.
        z_smooth, *_ = cov_model.ssm.filter_and_smooth(
            jnp.asarray(cov_np), Lambda_t, A_t
        )
        reconstructed = np.asarray(
            jax.vmap(lambda z: Lam @ z)(z_smooth)             # [T, C]
        )
        result            = cov_np.copy()
        nan_mask          = np.isnan(cov_np)
        result[nan_mask]  = reconstructed[nan_mask]
        return result

    def _cache_results(self, times_np, X_np, cov_np):
        preds, factors, kf_ll = self._run_forward(times_np, X_np, cov_np)
        self.predictions_  = np.asarray(preds)
        self.factors_filt_ = np.asarray(factors)
        self.kf_ll_        = float(kf_ll)

    def _make_manager(self, tvdfm: TVDFM) -> LTVTrainingManager:
        return LTVTrainingManager(
            tvdfm, lr_ssm=self.lr_ssm, lr_exposure=self.lr_exposure
        )

    def _fit_kwargs(self, max_epochs=None, **extra):
        """Common keyword arguments passed to manager.fit()."""
        return dict(
            max_epochs=max_epochs if max_epochs is not None else self.n_epochs,
            val_n=self.val_n,
            patience=self.patience,
            kf_ll_weight=self.kf_ll_weight,
            wd_exposure=self.wd_exposure,
            wd_ssm=self.wd_ssm,
            lambda_dev_weight=self.lambda_dev_weight,
            horizon_augment_max=self.horizon_augment_max,
            val_horizon=self.val_horizon,
            val_full_kf=self.val_full_kf,
            interpolation=self.interpolation,
            verbose=self.verbose,
            **extra,
        )

    # -------------------------------------------------------------------
    # Public API — fit variants
    # -------------------------------------------------------------------

    def fit(
        self,
        X,
        covariates=None,
        times=None,
        *,
        target_series=None,
        target_mask=None,
        max_epochs: Optional[int] = None,
        freeze_ssm: bool = False,
        freeze_exposure: bool = False,
        key: Optional[jax.Array] = None,
    ) -> "TVDFModel":
        """
        Fit the model to observations ``X``.

        Parameters
        ----------
        X : DataFrame [T, N] or ndarray [T, N]
            Observed series.  NaN marks missing observations.
        covariates : DataFrame [T, C] or ndarray [T, C], optional
            Exogenous covariates for the exposure model.
        times : array-like [T], optional
            Observation times.  Inferred from ``X.index`` for DataFrames.
        target_series : str, int, or list, optional
            Restrict the MSE loss to specific series.
        target_mask : ndarray [T, N], optional
            Binary mask alternative to ``target_series``.
        max_epochs : int, optional
            Override ``self.n_epochs`` for this call.
        freeze_ssm : bool
            Keep SSM parameters fixed (train exposure only).
        freeze_exposure : bool
            Keep exposure model fixed (train SSM only).
        key : jax.Array, optional
            PRNG key for dropout.

        Returns
        -------
        self
        """
        (X_np, times_np, columns,
         cov_np, cov_clean, tmask, val_tmask, tvdfm) = self._setup(
            X, covariates, times, target_series, target_mask,
            val_target_series=self.val_target_series,
        )
        self.X_train_      = X_np
        self.times_train_  = times_np
        self.cov_train_    = cov_np
        self.cov_clean_    = cov_clean
        self.columns_      = columns

        manager = self._make_manager(tvdfm)
        self.manager_ = manager
        self.model_   = manager.fit(
            times_np, X_np, cov_clean,
            target_mask=tmask,
            val_target_mask=val_tmask,
            freeze_ssm=freeze_ssm,
            freeze_exposure=freeze_exposure,
            key=key,
            **self._fit_kwargs(max_epochs),
        )
        self._cache_results(times_np, X_np, cov_np)
        return self

    # -------------------------------------------------------------------
    # Public API — inference
    # -------------------------------------------------------------------

    def predict(
        self,
        X=None,
        covariates=None,
        times=None,
    ) -> np.ndarray:
        """
        Return predicted observations ŷ_t = Λ(t) ẑ_t  [T, N].

        If ``X`` is None, returns the in-sample predictions stored during fit.
        """
        self._check_fitted()
        if X is None:
            return self.predictions_
        X, covariates = self._slice_to_selected(X, covariates)
        X_np, times_np, _ = parse_dataframe(X, times)
        cov_np = parse_covariates(covariates)
        preds, _, _ = self._run_forward(times_np, X_np, cov_np)
        return np.asarray(preds)

    def transform(
        self,
        X=None,
        covariates=None,
        times=None,
    ) -> np.ndarray:
        """
        Return Kalman-filtered latent factors ẑ_t  [T, K].

        If ``X`` is None, returns the in-sample factors stored during fit.
        """
        self._check_fitted()
        if X is None:
            return self.factors_filt_
        X, covariates = self._slice_to_selected(X, covariates)
        X_np, times_np, _ = parse_dataframe(X, times)
        cov_np = parse_covariates(covariates)
        _, factors, _ = self._run_forward(times_np, X_np, cov_np)
        return np.asarray(factors)

    def fit_transform(
        self,
        X,
        covariates=None,
        times=None,
        **fit_params,
    ) -> np.ndarray:
        """Fit and return filtered latent factors."""
        return self.fit(X, covariates, times, **fit_params).transform()

    # -------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------

    @property
    def loadings_(self) -> np.ndarray:
        """Effective loading matrix [N, K]."""
        self._check_fitted()
        m = self.model_
        L = m.Lambda_base + m.dLambda if m.learn_Lambda else m.Lambda_base
        return np.asarray(L)

    @property
    def transition_(self) -> np.ndarray:
        """Spectral-normalised transition matrix A [K, K]."""
        self._check_fitted()
        return np.asarray(self.model_.ssm.A)

    @property
    def noise_obs_(self) -> np.ndarray:
        """Observation noise standard deviations R diag [N]."""
        self._check_fitted()
        return np.asarray(jnp.exp(self.model_.ssm.log_diag_R))

    @property
    def noise_proc_(self) -> np.ndarray:
        """Process noise standard deviations Q diag [K]."""
        self._check_fitted()
        return np.asarray(jnp.exp(self.model_.ssm.log_diag_Q))

    # -------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------

    def summary(self) -> None:
        """Print a concise model summary."""
        self._check_fitted()
        m = self.model_
        y    = self.X_train_[~np.isnan(self.X_train_)]
        yhat = self.predictions_[~np.isnan(self.X_train_)]
        ss_res = float(np.sum((y - yhat) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

        print("=" * 58)
        print("  Time-Varying Dynamic Factor Model  (TVDFModel)")
        print("=" * 58)
        print(f"  Series (N)        : {m.n_series}")
        print(f"  Factors (K)       : {m.n_factors}")
        print(f"  factor_order      : {m.ssm.factor_order}")
        print(f"  error_order       : {m.ssm.error_order}")
        print(f"  Exposure          : {self.exposure or 'None  (static DFM)'}")
        print(f"  tv_Lambda         : {m.tv_Lambda}")
        print(f"  tv_A              : {m.tv_A}")
        print(f"  learn_Lambda      : {m.learn_Lambda}")
        print(f"  fixed_Q           : {m.ssm.fixed_Q}")
        if self.exposure == "ncde":
            print(f"  Interpolation     : {self.interpolation}")
            print(f"  Hidden / MLP      : {self.hidden_size} / "
                  f"{self.mlp_width}×{self.mlp_depth}")
        elif self.exposure == "gru":
            print(f"  Imputation        : {self.imputation}")
            print(f"  Hidden size       : {self.hidden_size}")
            if self.em_val_n is not None:
                print(f"  em_val_n          : {self.em_val_n} (EM uses {'full train' if self.em_val_n == 0 else f'T-{self.em_val_n}'})")
        print("-" * 58)
        print(f"  DFM init          : {self.dfm_init}")
        if self.dfm_init == "preselected":
            print(f"  target_col        : {self.target_col}")
            print(f"  presel_method     : {self.presel_method}")
            print(f"  n_select / lag_x  : {self.n_select} / {self.lag_x}")
        print(f"  lr_ssm / exposure : {self.lr_ssm} / {self.lr_exposure}")
        print(f"  wd_ssm / exposure : {self.wd_ssm} / {self.wd_exposure}")
        print(f"  kf_ll_weight      : {self.kf_ll_weight}")
        print("-" * 58)
        print(f"  KF log-lik.       : {self.kf_ll_:.4f}")
        print(f"  In-sample R²      : {r2:.4f}")
        print("=" * 58)

    # -------------------------------------------------------------------
    # Internal
    # -------------------------------------------------------------------

    def _check_fitted(self) -> None:
        if not hasattr(self, "model_"):
            raise RuntimeError("Model is not fitted yet. Call .fit() first.")
        