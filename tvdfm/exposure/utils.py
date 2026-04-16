import numpy as np
import jax.numpy as jnp
import jax


def forward_fill_nans(x: jnp.ndarray) -> jnp.ndarray:
    """
    Forward-fill NaN values along axis 0 of a 2D array [T, C].

    Last observed value per column persists until the next observation.
    Initial NaNs (before any observation) are filled with 0.0.

    Use for RNN-based exposure models where a step-function imputation
    is the natural choice for missing covariates.
    """
    def _step(carry, x_t):
        filled = jnp.where(jnp.isnan(x_t), carry, x_t)
        return filled, filled

    init = jnp.zeros(x.shape[-1], dtype=x.dtype)
    _, result = jax.lax.scan(_step, init, x)
    return result


def make_ncde_path(
    times: np.ndarray,
    covariates: np.ndarray,
    interpolation: str = "cubic",
) -> np.ndarray:
    """
    Build a clean covariate path [T, C] for NCDE interpolation.

    This is a **numpy/scipy preprocessing function**, called once before
    training or inference.  The result should be cached and passed as the
    ``covariates`` argument (or pre-computed into ``coeffs``) to the model.

    Strategy
    --------
    For each column, only the **observed** (non-NaN) timestamps are used to
    build the interpolation.  The curve is then evaluated at all T times:

    ``"cubic"``
        Scipy ``CubicSpline`` through the observed quarterly (or other
        low-frequency) timestamps.  Naturally gives a smooth path between
        releases — this is the control path X(t) that the NCDE integrates
        against.  **Interior NaNs are never linearly pre-filled**: the cubic
        spline is defined entirely by the observed points.

    ``"linear"``
        Piecewise-linear interpolation (``scipy.interpolate.interp1d``).

    ``"rectilinear"``
        Step function: last observed value persists until the next release
        (``interp1d(kind="previous")``).  Appropriate when you model the
        covariate as a constant between releases.

    **Trailing NaNs** (after the last observation) are handled by clamping to
    the last observed value (constant extrapolation) for all three schemes.
    Leading NaNs are filled with the first observed value.

    Parameters
    ----------
    times : [T]  np.ndarray  — observation times on the fine grid
    covariates : [T, C]  np.ndarray  — raw panel; NaN = unobserved
    interpolation : {"cubic", "linear", "rectilinear"}

    Returns
    -------
    path : [T, C]  np.ndarray (float32)  — clean path, no NaNs
    """
    from scipy.interpolate import CubicSpline, interp1d

    times = np.asarray(times, dtype=np.float64)
    cov   = np.asarray(covariates, dtype=np.float64)
    T, C  = cov.shape
    result = np.empty((T, C), dtype=np.float32)

    for c in range(C):
        col      = cov[:, c]
        obs_mask = ~np.isnan(col)
        obs_t    = times[obs_mask]
        obs_v    = col[obs_mask]

        if len(obs_t) == 0:
            result[:, c] = 0.0
            continue
        if len(obs_t) == 1:
            result[:, c] = obs_v[0]
            continue

        if interpolation == "rectilinear":
            fn = interp1d(
                obs_t, obs_v, kind="previous",
                bounds_error=False, fill_value=(obs_v[0], obs_v[-1]),
            )
        elif interpolation == "linear":
            fn = interp1d(
                obs_t, obs_v, kind="linear",
                bounds_error=False, fill_value=(obs_v[0], obs_v[-1]),
            )
        else:  # "cubic"
            cs = CubicSpline(obs_t, obs_v, extrapolate=False)
            # CubicSpline(extrapolate=False) returns NaN outside [t0, t1];
            # clamp times to [obs_t[0], obs_t[-1]] for constant extrapolation.
            fn = lambda t, _cs=cs, _t0=obs_t[0], _t1=obs_t[-1]: \
                _cs(np.clip(t, _t0, _t1))

        result[:, c] = fn(times).astype(np.float32)

    return result
