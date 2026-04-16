import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import numpy as np
from dataclasses import dataclass, field as dc_field
from typing import Dict, Optional, Tuple, Any, List

from .core import TVDFM


# ---------------------------------------------------------------------------
# Interpolation coefficients helper (shared with model.py)
# ---------------------------------------------------------------------------

def _get_coeffs(times_jax, covariates_jax, interpolation: str = "cubic"):
    """Pre-compute diffrax interpolation coefficients."""
    import diffrax
    if interpolation == "cubic":
        return diffrax.backward_hermite_coefficients(times_jax, covariates_jax)
    elif interpolation == "rectilinear":
        rt, ry = diffrax.rectilinear_interpolation(times_jax, covariates_jax)
        return (rt, ry)
    return (times_jax, covariates_jax)


# ---------------------------------------------------------------------------
# L2 helpers
# ---------------------------------------------------------------------------

def _l2_leaves(pytree) -> jnp.ndarray:
    """Sum of squared values across all array leaves of a pytree."""
    leaves = jax.tree_util.tree_leaves(eqx.filter(pytree, eqx.is_array))
    if not leaves:
        return jnp.zeros(())
    return sum(jnp.sum(jnp.square(x)) for x in leaves)


# ---------------------------------------------------------------------------
# RegularisationConfig
# ---------------------------------------------------------------------------

@dataclass
class RegularisationConfig:
    """
    Convenience container for all regularisation hyperparameters.

    Use this to organise your search grid; unpack into fit() / loss_e2e()
    as keyword arguments.  All weights are treated as dynamic JAX scalars
    inside JIT functions — changing them does NOT trigger recompilation.

    Parameters
    ----------
    kf_ll_weight : float
        Weight on the normalised KF log-likelihood term.
    wd_exposure : float
        L2 weight decay on exposure model parameters (loss term, not AdamW).
    wd_ssm : float
        L2 weight decay on SSM parameters.
    lambda_dev_weight : float
        Weight on the Lambda deviation penalty:
        alpha * mean_t ‖delta_Lambda(t)‖_F^2
        where delta_Lambda(t) is the exposure model output (= Lambda(t) - Lambda_base_eff).
        Penalises how far the time-varying loadings deviate from the DFM anchor.
        The static dLambda correction (when learn_Lambda=True) is not penalised here.
    """
    kf_ll_weight:         float = 0.05
    wd_exposure:          float = 10.0
    wd_ssm:               float = 0.1
    lambda_dev_weight:    float = 0.0

    def validate(self):
        if self.lambda_dev_weight < 0:
            raise ValueError("lambda_dev_weight must be >= 0")


# ---------------------------------------------------------------------------
# Regularisation terms
# ---------------------------------------------------------------------------

def compute_regularisation_terms(
    model: TVDFM,
    delta_Lambda: Optional[jnp.ndarray] = None,   # [T, N, K] pre-computed, or None
) -> Dict[str, jnp.ndarray]:
    """
    Compute raw Lambda deviation penalty (unweighted).

    lambda_dev = mean_t ‖delta_Lambda(t)‖_F^2
        where delta_Lambda(t) is the exposure model output
        (= Lambda(t) - Lambda_base_eff).  Pass pre-computed ``delta_Lambda``
        to avoid calling the exposure model a second time inside the loss.
        Zero when tv_Lambda=False or delta_Lambda is None.

    Python branch on model.tv_Lambda is static — fully JIT-traceable.

    Returns dict with scalar jnp array 'lambda_dev'.
    """
    if model.tv_Lambda and delta_Lambda is not None:
        lambda_dev = jnp.mean(jnp.sum(delta_Lambda ** 2, axis=(-2, -1)))
    else:
        lambda_dev = jnp.zeros(())

    return {"lambda_dev": lambda_dev}


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def loss_e2e(
    model: TVDFM,
    times, observations, covariates,
    *,
    target_mask=None,
    kf_ll_weight: float = 0.05,
    wd_exposure: float = 0.0,
    wd_ssm: float = 0.0,
    lambda_dev_weight: float = 0.0,
    coeffs=None,
    key=None,
) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
    """
    End-to-end loss with full regularisation breakdown.

    Returns
    -------
    total_loss : scalar jnp array
    breakdown  : dict with keys 'mse', 'kf_ll', 'lambda_dev', 'total'

    All weight parameters are dynamic JAX scalars — varying them does NOT
    trigger JIT recompilation.
    """
    # Build Lambda_t / A_t once — reused for both the KF and the deviation penalty.
    inference_mode = (key is None)
    Lambda_t, A_t = model._build_Lambda_A(
        times, covariates, inference=inference_mode, key=key, coeffs=coeffs,
    )
    # delta_Lambda is Lambda_t - Lambda_base_eff; extract it if tv_Lambda is on.
    if model.tv_Lambda:
        Lambda_eff = (
            model.Lambda_base + model.dLambda if model.learn_Lambda
            else model.Lambda_base
        )
        delta_Lambda_for_reg = Lambda_t - Lambda_eff[None]   # [T, N, K]
    else:
        delta_Lambda_for_reg = None

    filtered_states, lls = model.ssm.filter(observations, Lambda_t, A_t)
    predictions = jax.vmap(lambda L, z: L @ z)(Lambda_t, filtered_states)
    kf_ll = jnp.sum(lls)

    residuals = observations - predictions
    obs_mask  = ~jnp.isnan(observations)
    if target_mask is not None:
        residuals = residuals * target_mask
        obs_mask  = obs_mask & (target_mask > 0)

    residuals = jnp.where(obs_mask, residuals, 0.0)
    n_mse     = jnp.maximum(obs_mask.sum().astype(jnp.float32), 1.0)
    mse       = jnp.sum(jnp.square(residuals)) / n_mse

    n_ll    = jnp.maximum((~jnp.isnan(observations)).sum().astype(jnp.float32), 1.0)
    norm_ll = kf_ll / n_ll

    l2_exp = _l2_leaves(model.exposure) if model.exposure is not None else jnp.zeros(())
    l2_ssm = _l2_leaves(model.ssm)

    reg_terms = compute_regularisation_terms(
        model, delta_Lambda=delta_Lambda_for_reg,
    )

    total = (
        mse
        - kf_ll_weight        * norm_ll
        + wd_exposure          * l2_exp
        + wd_ssm               * l2_ssm
        + lambda_dev_weight    * reg_terms["lambda_dev"]
    )

    breakdown = {
        "mse"        : mse,
        "kf_ll"      : norm_ll,
        "lambda_dev" : reg_terms["lambda_dev"],
        "total"      : total,
    }
    return total, breakdown


@eqx.filter_jit
def _eval_loss(model, times, observations, covariates, coeffs,
               target_mask, kf_ll_weight, wd_exposure, wd_ssm,
               lambda_dev_weight):
    """JIT-compiled loss evaluation (inference=True, no dropout)."""
    return loss_e2e(
        model, times, observations, covariates,
        target_mask=target_mask, kf_ll_weight=kf_ll_weight,
        wd_exposure=wd_exposure, wd_ssm=wd_ssm,
        lambda_dev_weight=lambda_dev_weight,
        coeffs=coeffs,
    )


@eqx.filter_jit
def _step_e2e(model, opt_state, optim, times, observations, covariates, coeffs,
              target_mask, kf_ll_weight, wd_exposure, wd_ssm,
              lambda_dev_weight, key=None):
    """Single gradient-descent step."""
    diff_model, static_model = eqx.partition(model, eqx.is_array)

    def loss_fn(diff):
        full_model = eqx.combine(diff, static_model)
        total, _ = loss_e2e(
            full_model, times, observations, covariates,
            target_mask=target_mask, kf_ll_weight=kf_ll_weight,
            wd_exposure=wd_exposure, wd_ssm=wd_ssm,
            lambda_dev_weight=lambda_dev_weight,
            coeffs=coeffs, key=key,
        )
        return total

    loss, grads = jax.value_and_grad(loss_fn)(diff_model)
    updates, next_opt_state = optim.update(grads, opt_state, diff_model)
    next_diff = eqx.apply_updates(diff_model, updates)
    return eqx.combine(next_diff, static_model), next_opt_state, loss


# ---------------------------------------------------------------------------
# Training manager
# ---------------------------------------------------------------------------

class LTVTrainingManager:
    """
    Training manager for :class:`~tvdfm.TVDFM` with early stopping.

    Optimizer groups
    ----------------
    Weight decay is applied **inside the loss** as dynamic JAX scalars
    (``wd_exposure``, ``wd_ssm``), not inside the optimizer.  This means
    hyperparameter search over WD values never triggers JIT recompilation.

    ``"exposure"``  NCDE / GRU weights   : Adam(lr_exposure)
    ``"ssm"``       All SSM params        : Adam(lr_ssm)
    ``"other"``     Lambda_base, frozen   : set_to_zero

    Parameters
    ----------
    model : TVDFM
    lr_ssm : float
    lr_exposure : float
    """

    def __init__(
        self,
        model: TVDFM,
        lr_ssm: float = 1e-3,
        lr_exposure: float = 1e-3,
    ):
        self.model       = model
        self.lr_ssm      = lr_ssm
        self.lr_exposure = lr_exposure
        self._init_optim()

    # ------------------------------------------------------------------
    # Optimizer construction
    # ------------------------------------------------------------------

    def _init_optim(self, freeze_ssm: bool = False, freeze_exposure: bool = False):
        """
        Build the multi-transform Optax optimizer.

        No weight decay inside the optimizer — WD is a dynamic loss term.
        """
        trainable = eqx.filter(self.model, eqx.is_array)

        labels = jax.tree_util.tree_map(lambda _: "other", trainable)

        if self.model.exposure is not None:
            exp_label = "other" if freeze_exposure else "exposure"
            labels = eqx.tree_at(
                lambda m: m.exposure, labels,
                jax.tree_util.tree_map(lambda _: exp_label, trainable.exposure),
            )

        if not freeze_ssm:
            labels = eqx.tree_at(
                lambda m: m.ssm, labels,
                jax.tree_util.tree_map(lambda _: "ssm", trainable.ssm),
            )
            if self.model.ssm.fixed_Q:
                labels = eqx.tree_at(lambda m: m.ssm.log_diag_Q, labels, "other")

        if self.model.learn_Lambda and not freeze_ssm:
            labels = eqx.tree_at(lambda m: m.dLambda, labels, "ssm")

        tx_exposure = optax.adam(self.lr_exposure)
        tx_ssm      = optax.adam(self.lr_ssm)
        tx_other    = optax.set_to_zero()

        optim = optax.multi_transform(
            {"exposure": tx_exposure, "ssm": tx_ssm, "other": tx_other},
            lambda _: labels,
        )
        self.optim     = optax.chain(optax.clip_by_global_norm(1.0), optim)
        self.opt_state = self.optim.init(trainable)

    # ------------------------------------------------------------------
    # Data helpers
    # ------------------------------------------------------------------

    def _split(self, times, observations, covariates, val_n: int):
        """Chronological train / validation split using a fixed number of val steps."""
        T = times.shape[0]
        i = T - val_n
        if i <= 0:
            raise ValueError(
                f"val_n={val_n} must be smaller than T={T}."
            )
        return (
            (times[:i], observations[:i], covariates[:i]),
            (times[i:], observations[i:], covariates[i:]),
        )

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def fit(
        self,
        times, observations, covariates,
        *,
        target_mask=None,
        max_epochs: int = 1000,
        val_n: int = 20,
        patience: int = 10,
        freeze_ssm: bool = False,
        freeze_exposure: bool = False,
        kf_ll_weight: float = 0.05,
        wd_exposure: float = 0.1,
        wd_ssm: float = 0.1,
        lambda_dev_weight: float = 0.0,
        val_target_mask=None,
        horizon_augment_max: int = 0,
        val_horizon: int = 0,
        val_full_kf: bool = False,
        interpolation: str = "cubic",
        key: Optional[jax.Array] = None,
        verbose: bool = True,
    ) -> TVDFM:
        """
        Train the model end-to-end with early stopping on validation loss.

        Parameters
        ----------
        val_n : int
            Number of time steps reserved for validation (chronological tail).
        patience : int
            Early-stopping patience (epochs without improvement).
        freeze_ssm : bool
            Keep all SSM parameters fixed during this call.
        freeze_exposure : bool
            Keep the exposure model fixed during this call.
        kf_ll_weight : float
            Weight on the KF log-likelihood regulariser.
        wd_exposure : float
            L2 weight decay on exposure parameters (dynamic — no recompile).
        wd_ssm : float
            L2 weight decay on SSM parameters (dynamic — no recompile).
        lambda_dev_weight : float
            Weight on the Lambda deviation penalty used in the **training** loss.
            The validation / early-stopping loss always uses pure GDP MSE (weight=0).
        val_target_mask : ndarray [T, N] float32, optional
            When provided, the last ``val_n`` rows of this mask are used for the
            validation / early-stopping MSE instead of ``target_mask``.
            This decouples the training objective (e.g. all-series MSE for a
            richer gradient) from the early-stopping criterion (e.g. GDP-only
            MSE, the true evaluation metric).  When ``None`` the same mask is
            used for both training and validation.
        horizon_augment_max : int
            If > 0, at each training epoch a random horizon ``h ~ Uniform(0,
            horizon_augment_max)`` is sampled (Python-side) and the last ``h``
            rows of the covariate array are set to NaN before the GRU forward
            pass.  The GRU's forward-fill imputation then produces a hidden
            state identical to what it would compute at test-time for horizon
            ``h``.  This bridges the training / test distributional gap for
            long-horizon nowcasts.  Default 0 (disabled).  GRU-only feature.
        val_horizon : int
            If > 0, the last ``val_horizon`` rows of the validation covariates
            are set to NaN for the early-stopping evaluation.  This ensures the
            stopping criterion detects long-horizon degradation rather than
            optimising only for h=0 (fresh monthly data).  Default 0 (h=0 eval).
        val_full_kf : bool
            When True, the validation MSE is computed by running the Kalman
            filter on the **full** sequence (training + val rows) and reporting
            MSE only for the last ``val_n`` rows.  The KF is therefore warmed
            up from training data before it predicts the val window — matching
            exactly what happens at test time.
            When False (default), the KF runs on the val slice only (cold start
            from z₀=0), which is cheaper but produces a slightly different
            evaluation context from test time.
        interpolation : str
            Interpolation scheme for coefficient pre-computation.
        key : Optional[jax.Array]
            When provided, enables dropout during training.

        Returns
        -------
        The best model (lowest validation loss).
        """
        times, observations, covariates = map(
            jnp.asarray, [times, observations, covariates]
        )
        if target_mask is not None:
            target_mask = jnp.asarray(target_mask)
        if val_target_mask is not None:
            val_target_mask = jnp.asarray(val_target_mask)

        # Fast path: n_epochs=0 (e.g. pure EM baseline with val_n=0).
        # Skip all val setup — training loop is never entered, so we just
        # return the initial model unchanged with sentinel metrics.
        if max_epochs == 0:
            self.best_val_loss_   = float("inf")
            self.last_train_loss_ = float("nan")
            self.n_epochs_done_   = 0
            return self.model

        train, val = self._split(times, observations, covariates, val_n)
        t_t, obs_t, cov_t = train
        t_v, obs_v, cov_v = val

        # Split target_mask chronologically for training portion.
        # For validation use val_target_mask (last val_n rows) when provided,
        # otherwise fall back to the same target_mask slice.
        T = times.shape[0]
        i = T - val_n
        N = observations.shape[1]
        if target_mask is not None:
            tmask_t = target_mask[:i]
            tmask_v = target_mask[i:]
        else:
            tmask_t = tmask_v = None

        if val_target_mask is not None:
            # Override val mask — supports decoupled train/val objectives
            tmask_v = val_target_mask[i:]

        coeffs_t = _get_coeffs(t_t, cov_t, interpolation)

        # ── Warm-KF val setup ──────────────────────────────────────────────
        # When val_full_kf=True, evaluation runs the KF on all T rows so the
        # filter is warmed up from training data before predicting the val window.
        # A full-sequence target mask (zeros for training rows, tmask_v for val
        # rows) is used so that loss_e2e only counts the val portion in the MSE.
        #
        # When val_full_kf=False (legacy), only the val slice is passed (cold KF).
        if val_full_kf:
            # Full-sequence covariate array with val_horizon masking at the end.
            if val_horizon > 0 and covariates is not None:
                cov_full_eval = covariates.at[-val_horizon:, :].set(jnp.nan)
            else:
                cov_full_eval = covariates
            coeffs_full = _get_coeffs(times, cov_full_eval, interpolation)

            # Build a target mask for the full T rows: 0 for training rows, so
            # loss_e2e accumulates MSE only over the val slice.
            train_zeros = jnp.zeros((i, N), dtype=jnp.float32)
            if tmask_v is not None:
                val_mask_full = jnp.concatenate([train_zeros, tmask_v], axis=0)
            else:
                val_mask_full = jnp.concatenate(
                    [train_zeros, jnp.ones((val_n, N), dtype=jnp.float32)], axis=0
                )
        else:
            # Legacy cold-KF: val slice only.
            if val_horizon > 0 and cov_v is not None:
                cov_v_eval = cov_v.at[-val_horizon:, :].set(jnp.nan)
            else:
                cov_v_eval = cov_v
            coeffs_v = _get_coeffs(t_v, cov_v_eval, interpolation)

        self._init_optim(freeze_ssm=freeze_ssm, freeze_exposure=freeze_exposure)
        optim     = self.optim
        model     = self.model
        opt_state = self.opt_state

        kf_w   = jnp.asarray(kf_ll_weight, dtype=jnp.float32)
        wd_exp = jnp.asarray(wd_exposure,  dtype=jnp.float32)
        wd_s   = jnp.asarray(wd_ssm,       dtype=jnp.float32)

        lam_dev = jnp.asarray(lambda_dev_weight, dtype=jnp.float32)
        # Validation / early-stopping loss is always pure GDP MSE (penalties = 0).
        _zero = jnp.asarray(0.0, dtype=jnp.float32)

        def _compute_val(mdl):
            """Compute val loss with either warm-KF or cold-KF."""
            if val_full_kf:
                return _eval_loss(
                    mdl, times, observations, cov_full_eval, coeffs_full,
                    val_mask_full, kf_w, wd_exp, wd_s, _zero,
                )
            else:
                return _eval_loss(
                    mdl, t_v, obs_v, cov_v_eval, coeffs_v,
                    tmask_v, kf_w, wd_exp, wd_s, _zero,
                )

        best_val, _ = _compute_val(model)
        best_model = model
        best_state = opt_state
        no_improve = 0

        for epoch in range(max_epochs):
            # Horizon-aware training: randomly mask last h months of covariates.
            # Forces GRU to learn representations valid for all information horizons,
            # matching the test-time distribution where h ∈ {0, …, horizon_augment_max}.
            if horizon_augment_max > 0 and cov_t is not None:
                h_aug = int(np.random.randint(0, horizon_augment_max + 1))
                cov_t_epoch = cov_t.at[-h_aug:, :].set(jnp.nan) if h_aug > 0 else cov_t
            else:
                cov_t_epoch = cov_t

            step_key = jax.random.fold_in(key, epoch) if key is not None else None
            model, opt_state, train_loss = _step_e2e(
                model, opt_state, optim,
                t_t, obs_t, cov_t_epoch, coeffs_t,
                tmask_t, kf_w, wd_exp, wd_s, lam_dev, step_key,
            )
            # Validation always uses pure GDP MSE, with val_horizon masking if requested.
            val_loss, val_bd = _compute_val(model)

            if jnp.isfinite(val_loss) and val_loss < best_val:
                best_val   = val_loss
                best_model = model
                best_state = opt_state
                no_improve = 0
            else:
                no_improve += 1

            if verbose and epoch % 10 == 0:
                print(
                    f"  epoch {epoch:4d} | train {float(train_loss):.4f} "
                    f"| val total={float(val_loss):.4f}  "
                    f"mse={float(val_bd['mse']):.4f}  "
                    f"ll={float(val_bd['kf_ll']):.4f}  "
                    f"λ_dev={float(val_bd['lambda_dev']):.4f}"
                )

            if no_improve >= patience:
                break

        self.model     = best_model
        self.opt_state = best_state

        # Expose final metrics for post-training introspection
        # (train_loss / epoch are undefined when max_epochs=0 — guard them)
        self.best_val_loss_   = float(best_val)
        self.last_train_loss_ = float(train_loss) if max_epochs > 0 else float("nan")
        self.n_epochs_done_   = (epoch + 1)       if max_epochs > 0 else 0
        return self.model

