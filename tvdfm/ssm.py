"""
State-space model for the Dynamic Factor Model.

State vector
    x_t = [z_t (K·p) | eps_t (N·q)]

where z_t ∈ ℝ^{K·p} follows a VAR(p) companion form and eps_t ∈ ℝ^{N·q}
follows N independent AR(q) processes.  For p = q = 1 this reduces to the
standard DFM; p = q = 0 is the static case.

Factor companion  (p lags)
    F = [[A_1, A_2, …, A_p  ],   ← K × Kp  coefficient row
         [I_{K(p-1)},  0    ]]   ← K(p-1) × Kp  shift block

    A_j = A_base[:, K(j-1):Kj] + dA[:, K(j-1):Kj]
    Spectral normalisation is applied only to A_1 (the first K columns).

Error companion  (q lags, N independent AR(q) series)
    E = [[Ψ_1, Ψ_2, …, Ψ_q  ],   ← N × Nq  diagonal AR-coefficient blocks
         [I_{N(q-1)},  0     ]]   ← N(q-1) × Nq  shift block

    Ψ_j = diag(phi_{:,j}),   phi_n = PARCOR→AR(ar_Psi[n, :])
    Stationarity is enforced by the PARCOR parameterisation.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from typing import Any, Optional, Tuple

from .utils import _spectral_norm


# ---------------------------------------------------------------------------
# PARCOR → stationary AR(q)
# ---------------------------------------------------------------------------

def _parcor_to_ar(ρ: jnp.ndarray) -> jnp.ndarray:
    """
    Map q unconstrained parameters to stationary AR(q) coefficients.

    Bijection (Barndorff-Nielsen & Schou, 1973):

        r_j = tanh(ρ_j) ∈ (−1, 1)      (partial autocorrelations)
        phi   ← Levinson-Durbin(r)         (stationary AR(q) polynomial)

    All roots of  1 − phi_1 z − ⋯ − phi_q z^q  lie strictly outside the
    unit disk, guaranteeing stationarity of the AR(q) process.

    The Python loop over q is unrolled at JAX trace time (q is a static
    Python int stored in ``error_order``).
    """
    r = jnp.tanh(ρ)           # partial autocorrelations ∈ (−1, 1)^q
    q = r.shape[0]
    phi = jnp.zeros_like(r)
    if q == 0:
        return phi
    # Step k=0: phi[0] = r[0]  (no update to previous coefficients)
    phi = phi.at[0].set(r[0])
    # Steps k=1, …, q−1: Levinson-Durbin update
    #   phi^(k+1)_j = phi^(k)_j − r_k · phi^(k)_{k−1−j}   for j < k
    #   phi^(k+1)_k = r_k
    for k in range(1, q):
        phi = phi.at[:k].set(phi[:k] - r[k] * phi[k - 1 :: -1])
        phi = phi.at[k].set(r[k])
    return phi


# ---------------------------------------------------------------------------
# DFMStateSpace
# ---------------------------------------------------------------------------

class DFMStateSpace(eqx.Module):
    """
    Differentiable state-space model for the Dynamic Factor Model.

    Parameterisation
    ----------------
    A_full  = A_base + dA                    [K, K·p]  (pre-normalisation, all lags)
    A_1     = spectral_norm(A_full[:, :K])   [K, K]    first-lag transition matrix
    Q       = diag(exp(log_diag_Q))          [K, K]    factor process noise
    R       = diag(exp(log_diag_R))          [N, N]    observation noise

    AR(q) idiosyncratic errors  (when error_order = q > 0)
    --------------------------------------------------------
    eps_{n,t} = phi_{n,1} eps_{n,t−1} + ⋯ + phi_{n,q} eps_{n,t−q} + Cov_n η_t

    phi_n is obtained from ar_Psi[n, :] via the PARCOR bijection (stationary).

    Stored parameters
    -----------------
    A_base          [K, K·p]   frozen DFM anchor  (all p lag matrices)
    dA              [K, K·p]   trainable correction
    log_diag_Q      [K]        log-diagonal of Q  (frozen when fixed_Q=True)
    log_diag_R      [N]        log-diagonal of R
    ar_Psi          [N, q]     unconstrained PARCOR params  ([N,0] when q=0)
    log_diag_Sigma  [N]        log error innovation std    ([0]   when q=0)
    P0              [K, K]     frozen initial factor covariance
    """

    # ── Frozen numpy arrays (not JAX leaves) ──────────────────────────────
    A_base: Any   # [K, K·p]
    P0:     Any   # [K, K]

    # ── Trainable JAX arrays ───────────────────────────────────────────────
    dA:             jnp.ndarray   # [K, K·p]
    log_diag_Q:     jnp.ndarray   # [K]
    log_diag_R:     jnp.ndarray   # [N]
    ar_Psi:         jnp.ndarray   # [N, q] or [N, 0]
    log_diag_Sigma: jnp.ndarray   # [N]    or [0]

    # ── Static (shape-determining) ─────────────────────────────────────────
    n_factors:             int   = eqx.field(static=True)
    n_series:              int   = eqx.field(static=True)
    spectral_radius_bound: float = eqx.field(static=True)
    fixed_Q:               bool  = eqx.field(static=True)
    factor_order:          int   = eqx.field(static=True)
    error_order:           int   = eqx.field(static=True)

    def __init__(
        self,
        n_factors: int,
        n_series:  int,
        key:       jax.Array,
        A_init:    Optional[np.ndarray] = None,
        Q_init:    Optional[np.ndarray] = None,
        R_init:    Optional[np.ndarray] = None,
        spectral_radius_bound: float = 0.98,
        fixed_Q:               bool  = True,
        factor_order:          int   = 1,
        error_order:           int   = 0,
    ):
        if factor_order < 1:
            raise ValueError("factor_order must be ≥ 1")
        if error_order < 0:
            raise ValueError("error_order must be ≥ 0")

        K, N, p, q = n_factors, n_series, factor_order, error_order
        self.n_factors             = K
        self.n_series              = N
        self.spectral_radius_bound = spectral_radius_bound
        self.fixed_Q               = fixed_Q
        self.factor_order          = p
        self.error_order           = q

        # ── A_base [K, K·p] ─────────────────────────────────────────────
        if A_init is not None:
            A_np = np.asarray(A_init, dtype=np.float32)
            if A_np.shape == (K, K):
                # backward-compat: pad higher lags with zeros
                A_np = np.concatenate(
                    [A_np, np.zeros((K, K * (p - 1)), dtype=np.float32)], axis=1
                )
            elif A_np.shape != (K, K * p):
                raise ValueError(
                    f"A_init: expected [{K},{K}] or [{K},{K*p}], got {A_np.shape}"
                )
        else:
            A_np        = np.zeros((K, K * p), dtype=np.float32)
            A_np[:, :K] = np.eye(K) * 0.5

        self.A_base = A_np
        self.dA     = jnp.zeros((K, K * p))

        # ── Process noise  Q = diag(exp(log_diag_Q)) ────────────────────
        if fixed_Q:
            self.log_diag_Q = jnp.zeros(K)
        elif Q_init is not None:
            q_arr  = np.asarray(Q_init)
            diag_Q = np.diag(q_arr) if q_arr.ndim == 2 else q_arr
            self.log_diag_Q = jnp.array(
                np.log(np.clip(diag_Q, 1e-6, 10.0)), dtype=jnp.float32
            )
        else:
            self.log_diag_Q = jnp.full(K, float(np.log(0.1)))

        # ── Observation noise  R = diag(exp(log_diag_R)) ────────────────
        if R_init is not None:
            r_arr  = np.asarray(R_init)
            diag_R = np.diag(r_arr) if r_arr.ndim == 2 else r_arr
            self.log_diag_R = jnp.array(
                np.log(np.clip(diag_R, 1e-6, 10.0)), dtype=jnp.float32
            )
        else:
            self.log_diag_R = jnp.full(N, float(np.log(0.5)))

        # ── Error AR(q) parameters ───────────────────────────────────────
        if q > 0:
            self.ar_Psi         = jnp.zeros((N, q))
            self.log_diag_Sigma = jnp.full(N, float(np.log(0.5)))
        else:
            self.ar_Psi         = jnp.zeros((N, 0))
            self.log_diag_Sigma = jnp.zeros(0)

        # ── Initial factor covariance P0 [K, K] ─────────────────────────
        # P0 solves the discrete Lyapunov equation  P = A₁ P A₁ᵀ + Q,
        # which has a unique PSD solution iff ρ(A₁) < 1.
        # When A₁ is unstable (e.g. from DFM with enforce_stationarity=False)
        # scipy's solver may not raise but return a non-PSD / divergent matrix
        # → silently invalid S_reg inside the KF scan → NaN predictions.
        # Guard: only attempt the Lyapunov solve when spectral radius < 1,
        # then validate PSD; fall back to eye(K) otherwise.
        A1 = self.A_base[:, :K]
        if A_init is not None and Q_init is not None and not fixed_Q:
            try:
                from scipy.linalg import solve_discrete_lyapunov
                A64 = A1.astype(np.float64)
                # Stability gate: spectral radius via 2-norm (≥ 1 → no Dlyap solution)
                if np.linalg.norm(A64, ord=2) < 1.0:
                    q64 = np.asarray(Q_init, dtype=np.float64)
                    Q64 = np.diag(q64) if q64.ndim == 1 else q64
                    P0_cand = solve_discrete_lyapunov(A64, Q64).astype(np.float32)
                    # Validate PSD: all eigenvalues must be strictly positive
                    if np.all(np.linalg.eigvalsh(P0_cand) > 0):
                        self.P0 = P0_cand
                    else:
                        self.P0 = np.eye(K, dtype=np.float32)
                else:
                    self.P0 = np.eye(K, dtype=np.float32)
            except Exception:
                self.P0 = np.eye(K, dtype=np.float32)
        else:
            self.P0 = np.eye(K, dtype=np.float32)

    # ──────────────────────────────────────────────────────────────────────
    # Properties
    # ──────────────────────────────────────────────────────────────────────

    @property
    def state_dim(self) -> int:
        """Full augmented state dimension  K·p + N·q."""
        return self.n_factors * self.factor_order + self.n_series * self.error_order

    @property
    def A_eff(self) -> jnp.ndarray:
        """Full [K, K·p] effective coefficient matrix (pre-normalisation)."""
        return jnp.asarray(self.A_base) + self.dA

    @property
    def A(self) -> jnp.ndarray:
        """Spectral-normalised [K, K] first-lag transition matrix."""
        K = self.n_factors
        return _spectral_norm(self.A_eff[:, :K], self.spectral_radius_bound)

    @property
    def Q(self) -> jnp.ndarray:
        return jnp.diag(jnp.exp(self.log_diag_Q))

    @property
    def R(self) -> jnp.ndarray:
        return jnp.diag(jnp.exp(self.log_diag_R))

    # ──────────────────────────────────────────────────────────────────────
    # Companion-matrix builders
    # ──────────────────────────────────────────────────────────────────────

    def _build_companion_factor(self, A_kk: jnp.ndarray, ftype) -> jnp.ndarray:
        """
        VAR(p) companion matrix  F ∈ ℝ^{Kp × Kp}.

            F = [[A_1,         A_2, …, A_p],   ← coefficient row
                 [I_{K(p−1)},  0           ]]   ← shift block

        ``A_kk [K, K]`` is the (possibly time-varying) first-lag matrix;
        lags A_2, …, A_p come from  self.A_base[:, K:] + self.dA[:, K:].
        """
        K, p = self.n_factors, self.factor_order
        if p == 1:
            return A_kk                                              # [K, K]

        A_full = jnp.asarray(self.A_base, ftype) + self.dA.astype(ftype)
        coeff  = jnp.concatenate([A_kk, A_full[:, K:]], axis=1)    # [K, K·p]
        shift  = jnp.eye(K * (p - 1), K * p, dtype=ftype)          # [K(p−1), K·p]
        return jnp.concatenate([coeff, shift], axis=0)              # [K·p, K·p]

    def _build_companion_error(self, ftype) -> jnp.ndarray:
        """
        Companion matrix for N independent AR(q) errors  E ∈ ℝ^{Nq × Nq}.

            E = [[Ψ_1,         Ψ_2, …, Ψ_q],   ← diagonal AR-coeff blocks
                 [I_{N(q−1)},  0           ]]   ← shift block

        Ψ_j = diag(phi_{:,j})  where  phi_n = PARCOR→AR(ar_Psi[n, :]).
        """
        N, q = self.n_series, self.error_order
        phi    = jax.vmap(_parcor_to_ar)(self.ar_Psi.astype(ftype))  # [N, q]
        I_N  = jnp.eye(N, dtype=ftype)
        O_N  = jnp.zeros((N, N), dtype=ftype)

        rows = [[jnp.diag(phi[:, j]) for j in range(q)]]
        for k in range(1, q):
            rows.append([I_N if j == k - 1 else O_N for j in range(q)])
        return jnp.block(rows)                                      # [N·q, N·q]

    def _build_companion_A(self, A_kk: jnp.ndarray, ftype) -> jnp.ndarray:
        """
        Full augmented transition  [state_dim, state_dim]
            = block-diag(F_factor, E_error).
        """
        F = self._build_companion_factor(A_kk, ftype)              # [K·p, K·p]
        if self.error_order == 0:
            return F

        E   = self._build_companion_error(ftype)                   # [N·q, N·q]
        Kp, Nq = F.shape[0], E.shape[0]
        return jnp.block([
            [F,                          jnp.zeros((Kp, Nq), ftype)],
            [jnp.zeros((Nq, Kp), ftype), E                         ],
        ])

    # ──────────────────────────────────────────────────────────────────────
    # Noise and loading builders
    # ──────────────────────────────────────────────────────────────────────

    def _build_QR_ext(self, ftype) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Process noise  Q_ext [sd, sd]  and observation noise  R_ext [N, N].

        When error_order > 0 observations are noise-free (R_ext ≈ 0) and
        the error innovation variance enters through the Q_ext error block.
        """
        K, p, N, q = self.n_factors, self.factor_order, self.n_series, self.error_order
        Q_f   = self.Q.astype(ftype)    # [K, K]
        R_obs = self.R.astype(ftype)    # [N, N]

        if p == 1 and q == 0:
            return Q_f, R_obs

        Kp, sd = K * p, self.state_dim
        Q_ext  = jnp.zeros((sd, sd), ftype).at[:K, :K].set(Q_f)

        if q > 0:
            Cov     = jnp.diag(jnp.exp(self.log_diag_Sigma.astype(ftype)))  # [N, N]
            Q_ext = Q_ext.at[Kp:Kp + N, Kp:Kp + N].set(Cov)
            R_ext = 1e-6 * jnp.eye(N, dtype=ftype)
        else:
            R_ext = R_obs

        return Q_ext, R_ext

    def _extend_Lambda(self, Lambda_t: jnp.ndarray, ftype) -> jnp.ndarray:
        """
        Extend loadings  [T, N, K] → [T, N, state_dim].

        Lag states get zero loadings; the current error slot gets  I_N.
        """
        K, p, N, q = self.n_factors, self.factor_order, self.n_series, self.error_order
        if p == 1 and q == 0:
            return Lambda_t

        T, sd = Lambda_t.shape[0], self.state_dim
        Lambda_ext = jnp.zeros((T, N, sd), ftype).at[:, :, :K].set(Lambda_t.astype(ftype))

        if q > 0:
            Kp    = K * p
            Lambda_ext = Lambda_ext.at[:, :, Kp:Kp + N].set(jnp.eye(N, dtype=ftype)[None])
        return Lambda_ext

    def _build_P0_ext(self, ftype) -> jnp.ndarray:
        """
        Initial state covariance  [state_dim, state_dim].

        Factor block: stored P0.
        Error block:  approximate stationary variance  var / (1 − Cov_j phi_j²).
        """
        K, p, N, q = self.n_factors, self.factor_order, self.n_series, self.error_order
        Kp, sd  = K * p, self.state_dim
        P0_f    = jnp.asarray(self.P0, ftype)

        if p == 1 and q == 0:
            return P0_f

        P0_ext = jnp.eye(sd, dtype=ftype).at[:K, :K].set(P0_f)

        if q > 0:
            phi       = jax.vmap(_parcor_to_ar)(self.ar_Psi.astype(ftype))  # [N, q]
            sigma2  = jnp.exp(self.log_diag_Sigma.astype(ftype))           # [N]
            sum_phi2 = jnp.sum(jnp.square(phi), axis=1).clip(0.0, 0.99)
            var_eps = sigma2 / (1.0 - sum_phi2 + 1e-8)
            for lag in range(q):
                sl     = slice(Kp + lag * N, Kp + (lag + 1) * N)
                P0_ext = P0_ext.at[sl, sl].set(jnp.diag(var_eps))

        return P0_ext

    # ──────────────────────────────────────────────────────────────────────
    # Kalman filter
    # ──────────────────────────────────────────────────────────────────────

    def _kalman_scan(
        self,
        Y:   jnp.ndarray,   # [T, N]
        Lambda_t: jnp.ndarray,   # [T, N, SD]
        F_t: jnp.ndarray,   # [T, SD, SD]
        Q:   jnp.ndarray,   # [SD, SD]
        R:   jnp.ndarray,   # [N, N]
        P0:  jnp.ndarray,   # [SD, SD]
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Forward Kalman filter via jax.lax.scan.

        NaN observations are handled by zeroing the Kalman gain for that
        observation.  Joseph-form covariance update for numerical stability.

        Returns
        -------
        z_pred  [T, SD]      predicted state means
        P_pred  [T, SD, SD]  predicted covariances
        z_filt  [T, SD]      filtered state means
        P_filt  [T, SD, SD]  filtered covariances
        lls     [T]          per-step log-likelihood contributions
        """
        T, N = Y.shape
        SD   = F_t.shape[-1]
        ft   = Y.dtype

        x0   = jnp.zeros(SD, dtype=ft)
        I_SD = jnp.eye(SD, dtype=ft)
        I_N  = jnp.eye(N,  dtype=ft)

        def _step(carry, xs):
            x, P = carry
            y, Lambda, F = xs

            # ── Predict ──────────────────────────────────────────────────
            x_p     = F @ x
            P_p_raw = F @ P @ F.T + Q

            # ── Numerical guard on P_p ────────────────────────────────────
            P_p = jnp.where(jnp.isfinite(P_p_raw), P_p_raw, I_SD)

            # ── NaN mask ─────────────────────────────────────────────────
            miss  = jnp.isnan(y)
            y_obs = jnp.where(miss, 0.0, y)

            # ── Innovation and Kalman gain ────────────────────────────────
            ν      = y_obs - Lambda @ x_p
            S      = Lambda @ P_p @ Lambda.T + R
            S_reg  = S + 1e-4 * I_N                             # Tikhonov
            KT     = jax.scipy.linalg.solve(S_reg, Lambda @ P_p, assume_a="pos")
            # If solve produced NaN/Inf (ill-conditioned S_reg), zero out KT so
            # this timestep is treated as fully missing: K_gain=0, x_f=x_p, P_f=P_p.
            KT     = jnp.where(jnp.isfinite(KT), KT, 0.0)
            K_gain = jnp.where(miss[None, :], 0.0, KT.T)       # [SD, N]

            # ── Update (Joseph form) ──────────────────────────────────────
            IKLambda     = I_SD - K_gain @ Lambda
            x_f     = x_p + K_gain @ ν
            P_f_raw = IKLambda @ P_p @ IKLambda.T + K_gain @ R @ K_gain.T
            P_f_sym = 0.5 * (P_f_raw + P_f_raw.T)
            # Fallback to P_p (not I_SD) so covariance stays bounded when
            # the solve fails — prevents the divergence cycle.
            P_f     = jnp.where(jnp.isfinite(P_f_sym), P_f_sym, P_p)

            # ── Log-likelihood contribution ───────────────────────────────
            n_obs  = jnp.sum(~miss).astype(ft)
            obs    = ~miss
            S_ll   = S_reg * obs[:, None] * obs[None, :] + jnp.diag(miss.astype(ft))
            sign, logdet = jnp.linalg.slogdet(S_ll)
            logdet = jnp.where(sign > 0, logdet, 0.0)
            mahal  = ν @ jax.scipy.linalg.solve(S_reg, ν, assume_a="pos")
            ll_raw = -0.5 * (n_obs * jnp.log(2.0 * jnp.pi) + logdet + mahal)
            ll     = jnp.where(jnp.isfinite(ll_raw), ll_raw, -1e6)

            return (x_f, P_f), (x_p, P_p, x_f, P_f, ll)

        _, out = jax.lax.scan(_step, (x0, P0), (Y, Lambda_t, F_t))
        z_pred, P_pred, z_filt, P_filt, lls = out
        return z_pred, P_pred, z_filt, P_filt, lls

    # ──────────────────────────────────────────────────────────────────────
    # RTS smoother
    # ──────────────────────────────────────────────────────────────────────

    def _rts_scan(
        self,
        z_filt: jnp.ndarray,   # [T, SD]
        P_filt: jnp.ndarray,   # [T, SD, SD]
        z_pred: jnp.ndarray,   # [T, SD]
        P_pred: jnp.ndarray,   # [T, SD, SD]
        F_t:    jnp.ndarray,   # [T, SD, SD]
        ftype,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Rauch-Tung-Striebel backward smoother."""
        SD = z_filt.shape[-1]
        I  = jnp.eye(SD, dtype=ftype)

        def _step(carry, xs):
            z_s, P_s = carry
            z_f, P_f, z_p, P_p, F = xs
            # Smoother gain:  J = P_f F^T P_p^{-1}
            #   ⟺  P_p J^T = F P_f   (P_f symmetric)
            # 1e-3 nugget (vs old 1e-6) guards against ill-conditioned extended
            # covariances after long all-NaN stretches in float32.
            J    = jnp.nan_to_num(
                jax.scipy.linalg.solve(
                    P_p + 1e-3 * I, F @ P_f, assume_a="pos"
                ).T
            )
            z_sm = z_f + J @ (z_s - z_p)
            P_sm = P_f + J @ (P_s - P_p) @ J.T
            P_sm = jnp.nan_to_num(0.5 * (P_sm + P_sm.T))
            # Hard fallback: if smoother diverges, return the filtered estimate.
            z_sm = jnp.where(jnp.isnan(z_sm), z_f, z_sm)
            P_sm = jnp.where(jnp.isnan(P_sm), P_f, P_sm)
            return (z_sm, P_sm), (z_sm, P_sm)

        _, (zs, Ps) = jax.lax.scan(
            _step,
            (z_filt[-1], P_filt[-1]),
            (z_filt[:-1], P_filt[:-1], z_pred[1:], P_pred[1:], F_t[1:]),
            reverse=True,
        )
        z_sm = jnp.concatenate([zs, z_filt[-1:]], axis=0)
        P_sm = jnp.concatenate([Ps, P_filt[-1:]], axis=0)
        return z_sm, P_sm

    # ──────────────────────────────────────────────────────────────────────
    # Extended-state dispatcher
    # ──────────────────────────────────────────────────────────────────────

    def _build_extended(self, Lambda_t, F_t, ftype):
        """Build (Lambda_ext, F_ext, Q_ext, R_ext, P0_ext) for the augmented state."""
        Q_ext, R_ext = self._build_QR_ext(ftype)
        Lambda_ext        = self._extend_Lambda(Lambda_t, ftype)
        F_ext        = jax.vmap(
            lambda Fkk: self._build_companion_A(Fkk, ftype)
        )(F_t)
        P0_ext       = self._build_P0_ext(ftype)
        return Lambda_ext, F_ext, Q_ext, R_ext, P0_ext

    # ──────────────────────────────────────────────────────────────────────
    # Public interface
    # ──────────────────────────────────────────────────────────────────────

    def filter(
        self,
        Y:   jnp.ndarray,   # [T, N]
        Lambda_t: jnp.ndarray,   # [T, N, K]
        F_t: jnp.ndarray,   # [T, K, K]  first-lag transition (possibly time-varying)
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Kalman filter.

        Returns filtered factor means ``z_filt [T, K]`` and per-step
        log-likelihoods ``lls [T]``.  Internally extends the state when
        p > 1 or q > 0, but always slices back to K factors on output.
        """
        ft = Y.dtype
        K  = self.n_factors

        if self.factor_order == 1 and self.error_order == 0:
            Q  = self.Q.astype(ft)
            R  = self.R.astype(ft)
            P0 = jnp.asarray(self.P0, ft)
            _, _, z_filt, _, lls = self._kalman_scan(Y, Lambda_t, F_t, Q, R, P0)
            return z_filt, lls

        Lambda_ext, F_ext, Q_ext, R_ext, P0_ext = self._build_extended(Lambda_t, F_t, ft)
        _, _, z_filt, _, lls = self._kalman_scan(
            Y, Lambda_ext, F_ext, Q_ext, R_ext, P0_ext
        )
        return z_filt[:, :K], lls

    def filter_and_smooth(
        self,
        Y:   jnp.ndarray,
        Lambda_t: jnp.ndarray,
        F_t: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Kalman filter + RTS backward smoother.

        Returns
        -------
        z_smooth  [T, K]
        P_smooth  [T, K, K]
        z_filt    [T, K]
        P_filt    [T, K, K]
        lls       [T]
        """
        ft = Y.dtype
        K  = self.n_factors

        if self.factor_order == 1 and self.error_order == 0:
            Q  = self.Q.astype(ft)
            R  = self.R.astype(ft)
            P0 = jnp.asarray(self.P0, ft)
            z_pred, P_pred, z_filt, P_filt, lls = self._kalman_scan(
                Y, Lambda_t, F_t, Q, R, P0
            )
            F_scan = F_t
        else:
            Lambda_ext, F_ext, Q_ext, R_ext, P0_ext = self._build_extended(Lambda_t, F_t, ft)
            z_pred, P_pred, z_filt, P_filt, lls = self._kalman_scan(
                Y, Lambda_ext, F_ext, Q_ext, R_ext, P0_ext
            )
            F_scan = F_ext

        z_sm, P_sm = self._rts_scan(z_filt, P_filt, z_pred, P_pred, F_scan, ft)
        return (
            z_sm[:, :K],   P_sm[:, :K, :K],
            z_filt[:, :K], P_filt[:, :K, :K],
            lls,
        )
