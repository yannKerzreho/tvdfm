import equinox as eqx
import jax
import jax.numpy as jnp
import diffrax
from typing import Any, Literal, Optional, Tuple

from .base import AbstractExposure


class NCDEExposure(AbstractExposure):
    """
    Exposure model based on a Neural Controlled Differential Equation (NCDE).

    Hidden state dynamics:

        dh/dt = MLP(h) @ dX/dt,   h(t0) = W_init · x(t0)

    where X(t) is a continuous interpolation of the covariate path.
    Readout layers map h(t) -> (delta_Lambda_t, delta_A_t).

    Mixed-frequency covariates
    --------------------------
    Pre-process the covariate panel **once before training** with
    :func:`~tvdfm.exposure.make_ncde_path`.  The resulting NaN-free array is
    what you pass as ``covariates`` (or pre-compute ``coeffs``).  Do **not**
    pass the raw panel with interior NaNs::

        path   = make_ncde_path(times_np, raw_covariates_np, interpolation="cubic")
        coeffs = diffrax.backward_hermite_coefficients(times_jax, jnp.array(path))
        model(times_jax, jnp.array(path), coeffs=coeffs)

    Parameters
    ----------
    lambda_indices : Optional[Tuple[int, ...]]
        - ``None``          → delta_Lambda predicted for all N series.
        - ``(i, j, ...)``   → only those rows are non-zero (scattered into [T, N, K]).
    interpolation : {"cubic", "linear", "rectilinear"}
        Interpolation scheme used by the diffrax control path.
        Must match the scheme used in ``make_ncde_path`` when pre-computing ``coeffs``.
    input_reducer : Optional[eqx.nn.Linear]
        Optional linear projection [C_in] -> [n_covariates] applied before
        the backbone. Build from PCA::

            reducer = AbstractExposure.make_pca_reducer(X_train, 10, key)
            model   = NCDEExposure(..., n_covariates=10, input_reducer=reducer)
    """

    initial_hidden: eqx.nn.Linear
    vector_field: eqx.nn.MLP
    readout: eqx.nn.Linear
    readout_A: eqx.nn.Linear

    hidden_size: int = eqx.field(static=True)
    mlp_width: int = eqx.field(static=True)
    mlp_depth: int = eqx.field(static=True)
    dropout: float = eqx.field(static=True)
    activation: Any = eqx.field(static=True)
    interpolation: str = eqx.field(static=True)

    def __init__(
        self,
        n_covariates: int,
        n_series: int,
        n_factors: int,
        key: jax.Array,
        hidden_size: int = 8,
        mlp_width: int = 64,
        mlp_depth: int = 2,
        dropout: float = 0.1,
        activation: Any = jax.nn.tanh,
        lambda_indices: Optional[Tuple[int, ...]] = None,
        interpolation: Literal["cubic", "linear", "rectilinear"] = "cubic",
        input_reducer: Optional[eqx.nn.Linear] = None,
    ):
        self.n_series = n_series
        self.n_factors = n_factors
        self.n_covariates = n_covariates
        self.lambda_indices = lambda_indices
        self.input_reducer = input_reducer

        self.hidden_size = hidden_size
        self.mlp_width = mlp_width
        self.mlp_depth = mlp_depth
        self.dropout = dropout
        self.activation = activation
        self.interpolation = interpolation

        keys = jax.random.split(key, 4)
        n_out = self.n_lambda_out

        self.initial_hidden = eqx.nn.Linear(n_covariates, hidden_size, key=keys[0])
        self.vector_field = eqx.nn.MLP(
            in_size=hidden_size,
            out_size=hidden_size * n_covariates,
            width_size=mlp_width,
            depth=mlp_depth,
            activation=activation,
            key=keys[1],
        )

        # Zero-initialize readouts so model starts identical to a static DFM.
        raw_ro = eqx.nn.Linear(hidden_size, n_out * n_factors, key=keys[2])
        self.readout = eqx.tree_at(lambda l: l.weight, raw_ro, jnp.zeros_like(raw_ro.weight))
        self.readout = eqx.tree_at(lambda l: l.bias, self.readout, jnp.zeros_like(self.readout.bias))

        raw_ro_A = eqx.nn.Linear(hidden_size, n_factors * n_factors, key=keys[3])
        self.readout_A = eqx.tree_at(lambda l: l.weight, raw_ro_A, jnp.zeros_like(raw_ro_A.weight))
        self.readout_A = eqx.tree_at(lambda l: l.bias, self.readout_A, jnp.zeros_like(self.readout_A.bias))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _vf(self, t: float, y: jnp.ndarray, args) -> jnp.ndarray:
        """CDE vector field: dy/dt = MLP(y) @ dX/dt."""
        dX_dt = args.evaluate(t)
        matrix = self.vector_field(y).reshape(self.hidden_size, self.n_covariates)
        return matrix @ dX_dt

    def _build_control(
        self, times: jnp.ndarray, cov: jnp.ndarray, coeffs: Optional[Tuple]
    ) -> Tuple[Any, Any]:
        """
        Build the diffrax control path from a clean (NaN-free) covariate array.

        ``cov`` must be free of interior NaNs — use :func:`make_ncde_path` first.
        If ``coeffs`` are pre-computed (recommended), they are used directly.

        Returns ``(control, coeffs)``.
        """
        if self.interpolation == "rectilinear":
            # rectilinear_interpolation doubles the time grid so that
            # LinearInterpolation produces a step function with impulses at
            # each observation.
            if coeffs is None:
                rt, ry = diffrax.rectilinear_interpolation(times, cov)
                coeffs = (rt, ry)
            return diffrax.LinearInterpolation(ts=coeffs[0], ys=coeffs[1]), coeffs

        elif self.interpolation == "linear":
            if coeffs is None:
                coeffs = (times, cov)
            return diffrax.LinearInterpolation(ts=coeffs[0], ys=coeffs[1]), coeffs

        else:  # "cubic" (default)
            if coeffs is None:
                coeffs = diffrax.backward_hermite_coefficients(times, cov)
            return diffrax.CubicInterpolation(times, coeffs), coeffs

    def _solve(
        self,
        times: jnp.ndarray,
        cov: jnp.ndarray,
        coeffs: Optional[Tuple],
    ) -> Tuple[jnp.ndarray, Any]:
        """Build the control path and solve the NCDE. Returns (hidden_states [T, H], coeffs)."""
        control, coeffs = self._build_control(times, cov, coeffs)
        y0 = self.initial_hidden(cov[0])

        saveat = diffrax.SaveAt(ts=times)
        stepsize_controller = diffrax.PIDController(rtol=1e-3, atol=1e-3)
        dt0 = (times[-1] - times[0]) / len(times)
        max_steps = int(len(times) * 128)

        solution = diffrax.diffeqsolve(
            diffrax.ODETerm(self._vf),
            diffrax.Dopri5(),
            t0=times[0],
            t1=times[-1],
            dt0=dt0,
            y0=y0,
            args=control,
            saveat=saveat,
            stepsize_controller=stepsize_controller,
            max_steps=max_steps,
        )
        return solution.ys, coeffs

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
        coeffs: Optional[Tuple] = None,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        cov = self._reduce(covariates)
        # If a PCA reducer is active the caller's pre-computed `coeffs` were
        # built from the raw (unreduced) covariate matrix and have the wrong
        # leading dimension for the control path.  Discard them so that
        # _build_control rebuilds them from the already-reduced `cov`.
        if self.input_reducer is not None:
            coeffs = None
        hidden_states, _ = self._solve(times, cov, coeffs)
        T = hidden_states.shape[0]

        if not inference and key is not None:
            # Variational dropout: single spatial mask shared across all T timesteps.
            mask = jax.random.bernoulli(key, 1 - self.dropout, (self.hidden_size,))
            hidden_states = hidden_states * (mask / (1 - self.dropout))

        n_out = self.n_lambda_out
        delta_lambda_active = jax.vmap(self.readout)(hidden_states).reshape(
            T, n_out, self.n_factors
        )
        delta_A = jax.vmap(self.readout_A)(hidden_states).reshape(
            T, self.n_factors, self.n_factors
        )
        return self._scatter_lambda(delta_lambda_active, T), delta_A
