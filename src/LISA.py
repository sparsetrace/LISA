# LISA.py
# ============================================================
# LISA: NLSA encoder + GPLM nonlinear baseline decoder
#       + dense in-context Gaussian Process Regression (GPR)
#       on residuals in diffusion coordinate space.
#
# Baseline:
#   window W (L,D) -> ψ(W) in R^r (via NLSA Nyström OOS)
#   y_glob = GPLM(ψ) in R^D   (nonlinear stable decoder)
#
# In-context:
#   Given prefix length ℓ >= L:
#     Context pairs count: K_ctx = ℓ - L
#     For each context window:
#       ψ_i = encode(W_i)
#       e_i = y_true_i - y_glob(ψ_i)
#     Fit GP on residuals e(ψ)
#
# Rollout:
#   y = y_glob(ψ_q) + w_eff(var_q) * e_gp_mean(ψ_q)
#   (optionally sample residual using GP posterior variance)
#
# If ℓ == L (no context pairs), LISA == baseline AR (NLSA + GPLM).
#
# Dependencies:
#   numpy, scipy
#
# Repo assumptions:
#   - NLSA.py exports class NLSA with:
#        .L, .D_, .K_, .psi_ (K,r), .R_ (N,D centered if center=True), .mu_
#        .encode_window(W_LD) -> (r,)
#   - GPLM.py or gplm.py exports class GPLM with:
#        GPLM(X_train, Y_train, ...) where X_train is (N,r), Y_train is (N,D)
#        __call__(Xq) maps (B,r)->(B,D) and (r,)->(D,)
#
# ============================================================

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from scipy.linalg import cho_factor, cho_solve, solve_triangular

# ------------------------------------------------------------
# Robust imports (package or flat)
# ------------------------------------------------------------
try:
    # if you're inside a package
    from .NLSA import NLSA
except Exception:
    from NLSA import NLSA

try:
    from .GPLM import GPLM
except Exception:
    from GPLM import GPLM


# ============================================================
# Small utilities
# ============================================================

def _as_2d(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    return X


def _sliding_windows(F_tD: np.ndarray, L: int) -> np.ndarray:
    """
    Return windows W (K,L,D) from F (N,D) with K=N-L+1.
    Handles possible (K,D,L) output from sliding_window_view.
    """
    F = _as_2d(F_tD)
    W = sliding_window_view(F, window_shape=int(L), axis=0)

    a, b = W.shape[1], W.shape[2]
    if (a, b) == (L, F.shape[1]):
        return np.ascontiguousarray(W)
    if (a, b) == (F.shape[1], L):
        return np.ascontiguousarray(np.transpose(W, (0, 2, 1)))

    raise ValueError(f"Unexpected window shape {W.shape} for L={L}, D={F.shape[1]}.")


def _pairwise_sq_dists(X: np.ndarray) -> np.ndarray:
    """
    Dense pairwise squared Euclidean distances (n,n) for X (n,r).
    """
    X = np.asarray(X, dtype=np.float64)
    x2 = np.sum(X * X, axis=1, keepdims=True)
    d2 = x2 + x2.T - 2.0 * (X @ X.T)
    np.maximum(d2, 0.0, out=d2)
    return d2


def _estimate_rbf_ell_from_d2(d2_mat: np.ndarray, q: float = 0.5, eps: float = 1e-12, min_ell: float = 1e-6) -> float:
    """
    If kernel is exp(-||x-y||^2/(2 ell^2)), a good heuristic is:
      ell^2 ~= quantile(d^2)/2
    """
    iu = np.triu_indices_from(d2_mat, k=1)
    vals = d2_mat[iu]
    if vals.size == 0:
        return 1.0
    qv = float(np.quantile(vals, q))
    ell = np.sqrt(max(qv, 0.0) / 2.0 + eps)
    return float(max(ell, min_ell))


# ============================================================
# LISA main class
# ============================================================

class LISA:
    """
    LISA: NLSA encoder + GPLM baseline + dense GPR residual correction.

    - If prefix length ℓ == L: no in-context pairs -> baseline AR only.
    - If ℓ > L: residual GP is fit on the prefix windows and applied during rollout.

    Notes:
    - Residual GP is multi-output with independent output dims sharing a kernel.
      That means GP mean is vector in R^D, but variance is a scalar (same for all dims).
    """

    def __init__(
        self,
        F_tX: np.ndarray,
        *,
        L: int,
        rank: int,
        # ------------------ NLSA encoder hyperparams ------------------
        beta: float | None = None,
        alpha: float = 1.0,
        center_outputs: bool = True,
        drop_first: bool = True,
        seed: int = 0,
        nlsa_kwargs: Optional[dict] = None,
        # ------------------ GPLM baseline decoder hyperparams ----------
        gplm_kwargs: Optional[dict] = None,
        # ------------------ IC / GPR controls -------------------------
        ctx_min_windows: Optional[int] = None,
        ctx_k0: float = 10.0,                 # base mixing K_ctx/(K_ctx+ctx_k0)
        gp_noise2: float = 1e-3,              # σ_n^2 in (K + σ_n^2 I)
        gp_kernel: str = "rbf",               # "rbf" or "linear"
        gp_rbf_ell: float | None = None,      # fixed, or auto from context
        gp_rbf_q: float = 0.5,
        # ------------------ stability / trust gating -----------------
        use_var_gate: bool = True,
        gate_tau2: float = 1.0,
        gate_mode: str = "rational",          # "rational" or "exp"
    ):
        if nlsa_kwargs is None:
            nlsa_kwargs = {}
        if gplm_kwargs is None:
            gplm_kwargs = {}

        self._rng = np.random.default_rng(int(seed))

        # ---- Encoder ----
        self.base = NLSA(
            F_tX,
            L=int(L),
            rank=int(rank),
            beta=beta,
            alpha=float(alpha),
            center=bool(center_outputs),
            drop_first=bool(drop_first),
            seed=int(seed),
            **nlsa_kwargs,
        )

        base = self.base
        if base.psi_ is None or base.psi_.shape[1] == 0:
            raise ValueError("NLSA produced zero latent dims. Increase rank or set drop_first=False.")

        self.L = int(base.L)
        self.D = int(base.D_)
        self.r = int(base.psi_.shape[1])

        # mean handling: baseline GPLM will be trained in *centered output space*
        self.center_outputs = bool(base.center)
        self.mu_X = base.mu_.reshape(-1).astype(np.float64) if self.center_outputs else np.zeros((self.D,), dtype=np.float64)

        # ---- Train baseline GPLM: ψ -> next sample (centered) ----
        K = int(base.K_)
        N_pairs = K - 1
        if N_pairs < 4:
            raise ValueError("Training series too short for LISA with this L.")

        X_train = np.asarray(base.psi_[:N_pairs, :], dtype=np.float64)  # (K-1,r)
        Y_train_c = np.asarray(base.R_[self.L:self.L + N_pairs, :], dtype=np.float64)  # (K-1,D) centered if base.center

        gkw = dict(gplm_kwargs)
        gkw.setdefault("seed", int(seed))
        gkw.setdefault("center_X", False)         # IMPORTANT: Y_train is already centered
        gkw.setdefault("sigma2", 1e-5)
        gkw.setdefault("jitter", 1e-8)
        gkw.setdefault("m", min(1024, X_train.shape[0]))
        gkw.setdefault("inducing", "kmeans_medoids")
        # optional speedup at inference:
        # gkw.setdefault("pred_k", 128)

        self.gplm = GPLM(X_train, Y_train_c, **gkw)

        # ---- IC controls ----
        self.ctx_k0 = float(ctx_k0)
        self.ctx_min_windows = int(ctx_min_windows) if ctx_min_windows is not None else max(1, self.r + 1)

        # ---- GP residual controls ----
        self.gp_noise2 = float(gp_noise2)
        self.gp_kernel = str(gp_kernel).lower().strip()
        if self.gp_kernel not in ("rbf", "linear"):
            raise ValueError("gp_kernel must be 'rbf' or 'linear'")

        self.gp_rbf_ell = None if gp_rbf_ell is None else float(gp_rbf_ell)
        self.gp_rbf_q = float(gp_rbf_q)
        self.last_gp_rbf_ell_: Optional[float] = None

        # ---- variance gating ----
        self.use_var_gate = bool(use_var_gate)
        self.gate_tau2 = float(gate_tau2)
        self.gate_mode = str(gate_mode).lower().strip()
        if self.gate_mode not in ("rational", "exp"):
            raise ValueError("gate_mode must be 'rational' or 'exp'")

    # ============================================================
    # Baseline utilities
    # ============================================================

    def _encode_batch(self, W_BLD: np.ndarray) -> np.ndarray:
        """
        Encode windows (B,L,D) -> (B,r) using base.encode_window in a loop.
        """
        W = np.asarray(W_BLD, dtype=np.float64)
        if W.ndim == 2:
            W = W[None, :, :]
        B = W.shape[0]
        out = np.zeros((B, self.r), dtype=np.float64)
        for i in range(B):
            out[i] = self.base.encode_window(W[i])
        return out

    def _baseline_centered_batch(self, Psi_Br: np.ndarray) -> np.ndarray:
        """
        GPLM baseline in centered output space: (B,r) -> (B,D)
        """
        Psi = np.asarray(Psi_Br, dtype=np.float64)
        if Psi.ndim == 1:
            Psi = Psi[None, :]
        Yc = self.gplm(Psi)  # (B,D)
        return np.asarray(Yc, dtype=np.float64)

    def _baseline_centered_one(self, psi_r: np.ndarray) -> np.ndarray:
        return self._baseline_centered_batch(np.asarray(psi_r, dtype=np.float64))[0]

    # ============================================================
    # GP kernel utilities (residual GP)
    # ============================================================

    def _gp_kernel_matrix(self, Psi_ctx: np.ndarray) -> Tuple[np.ndarray, Optional[float]]:
        """
        Build dense kernel matrix K(Ψ,Ψ) on context points.
        Returns (K_mat, ell_used).
        """
        Psi_ctx = np.asarray(Psi_ctx, dtype=np.float64)

        if self.gp_kernel == "linear":
            return Psi_ctx @ Psi_ctx.T, None

        # RBF
        d2 = _pairwise_sq_dists(Psi_ctx)
        if self.gp_rbf_ell is None:
            ell_used = _estimate_rbf_ell_from_d2(d2, q=self.gp_rbf_q)
        else:
            ell_used = float(self.gp_rbf_ell)

        self.last_gp_rbf_ell_ = ell_used
        K = np.exp(-0.5 * d2 / (ell_used**2 + 1e-12))
        return K, ell_used

    def _gp_kernel_eval(self, Psi_ctx: np.ndarray, psi_q: np.ndarray, ell_used: Optional[float]) -> np.ndarray:
        """
        k(Ψ,ψ_q) for query ψ_q. Returns (K_ctx,).
        """
        Psi_ctx = np.asarray(Psi_ctx, dtype=np.float64)
        psi_q = np.asarray(psi_q, dtype=np.float64)

        if self.gp_kernel == "linear":
            return Psi_ctx @ psi_q

        assert ell_used is not None
        diff = Psi_ctx - psi_q[None, :]
        d2 = np.einsum("kr,kr->k", diff, diff, optimize=True)
        return np.exp(-0.5 * d2 / (ell_used**2 + 1e-12))

    def _k_qq(self, psi_q: np.ndarray) -> float:
        """
        Kernel self-similarity k(ψ,ψ).
        """
        psi_q = np.asarray(psi_q, dtype=np.float64)
        if self.gp_kernel == "linear":
            return float(np.dot(psi_q, psi_q))
        return 1.0  # RBF

    def _gate_from_var(self, var_f: float) -> float:
        """
        Convert predictive function variance -> trust weight in [0,1].
        """
        if not self.use_var_gate:
            return 1.0

        v = max(float(var_f), 0.0)
        tau2 = max(float(self.gate_tau2), 1e-18)

        if self.gate_mode == "exp":
            return float(np.exp(-v / tau2))
        # rational
        return float(tau2 / (tau2 + v))

    # ============================================================
    # Public API
    # ============================================================

    def predict_one_step(self, W_LD: np.ndarray) -> np.ndarray:
        """
        Predict one step from a single window (L,D) using baseline only.
        Returns (D,) in original (uncentered) scale.
        """
        W = np.asarray(W_LD, dtype=np.float64)
        if W.ndim == 1:
            W = W[:, None]
        if W.shape != (self.L, self.D):
            raise ValueError(f"Expected window shape {(self.L, self.D)}, got {W.shape}")

        psi = self.base.encode_window(W)
        y_c = self._baseline_centered_one(psi)
        return y_c + self.mu_X if self.center_outputs else y_c

    def __call__(
        self,
        prefix: np.ndarray,
        steps: int = 1,
        *,
        return_var: bool = False,
        sample: bool = False,
        rng: Optional[np.random.Generator] = None,
        include_obs_noise: bool = True,
    ):
        """
        Autoregressive forecast from prefix (ℓ,D), ℓ >= L.

        If ℓ == L (IC nullset): baseline AR only (NLSA+GPLM).
        If ℓ >  L: uses in-context dense GPR on residuals.

        Parameters
        ----------
        prefix : (ℓ,D)
        steps : horizon H
        return_var : return scalar GP function variance per step (H,)
        sample : sample GP residual instead of mean (generative mode)
        rng : RNG for sampling
        include_obs_noise : if sample=True, draw using var_y = var_f + gp_noise2

        Returns
        -------
        preds : (H,D)  (or (D,) if H==1)
        vars  : (H,) if return_var=True
        """
        prefix = _as_2d(prefix)
        ell, D = prefix.shape
        if D != self.D:
            raise ValueError(f"LISA trained with D={self.D}, got prefix D={D}.")
        if ell < self.L:
            raise ValueError(f"Need prefix length ell >= L={self.L}.")

        H = int(steps)
        if H <= 0:
            out = np.zeros((0, self.D), dtype=np.float64)
            return (out, np.zeros((0,), dtype=np.float64)) if return_var else out

        if rng is None:
            rng = self._rng

        # seed window for AR rollout
        cur = prefix[-self.L:, :].copy()

        # ---------------------------
        # If IC nullset -> baseline AR
        # ---------------------------
        K_ctx = ell - self.L
        if K_ctx <= 0 or self.r == 0:
            preds = self._rollout_baseline(cur, H)
            if return_var:
                return preds[0] if H == 1 else preds, np.zeros((H,), dtype=np.float64)
            return preds[0] if H == 1 else preds

        if K_ctx < self.ctx_min_windows:
            preds = self._rollout_baseline(cur, H)
            if return_var:
                return preds[0] if H == 1 else preds, np.zeros((H,), dtype=np.float64)
            return preds[0] if H == 1 else preds

        # ---------------------------
        # Build context windows + targets
        # ---------------------------
        W_all = _sliding_windows(prefix, self.L)    # (K_n,L,D), K_n = ell-L+1
        # context pairs correspond to windows that have next sample observed
        W_ctx = np.ascontiguousarray(W_all[:K_ctx, :, :])          # (K_ctx,L,D)
        Y_ctx = np.asarray(prefix[self.L:self.L + K_ctx, :], dtype=np.float64)  # (K_ctx,D)

        # center targets consistently with baseline training
        Y_ctx_c = (Y_ctx - self.mu_X[None, :]) if self.center_outputs else Y_ctx

        # encode ψ for context
        Psi_ctx = self._encode_batch(W_ctx)                        # (K_ctx,r)

        # baseline predictions on context
        Y_glob_ctx_c = self._baseline_centered_batch(Psi_ctx)      # (K_ctx,D)

        # residual table
        E_ctx = Y_ctx_c - Y_glob_ctx_c                              # (K_ctx,D)

        # ---------------------------
        # Fit dense GP on residuals
        # ---------------------------
        K_mat, ell_used = self._gp_kernel_matrix(Psi_ctx)          # (K_ctx,K_ctx)
        K_reg = K_mat + self.gp_noise2 * np.eye(K_ctx, dtype=np.float64)

        # Cholesky factorization (preferred)
        cf = cho_factor(K_reg, lower=True, check_finite=False)
        alpha = cho_solve(cf, E_ctx, check_finite=False)           # (K_ctx,D)
        Lfac, lower = cf

        # base context mixing weight
        w_ctx_base = float(K_ctx) / float(K_ctx + self.ctx_k0) if self.ctx_k0 > 0 else 1.0

        preds = np.zeros((H, self.D), dtype=np.float64)
        vars_out = np.zeros((H,), dtype=np.float64) if return_var else None

        # ---------------------------
        # AR rollout with GP residual
        # ---------------------------
        for h in range(H):
            psi_q = self.base.encode_window(cur)                   # (r,)

            # baseline
            y_glob_c = self._baseline_centered_one(psi_q)          # (D,)

            # GP residual mean
            k_eval = self._gp_kernel_eval(Psi_ctx, psi_q, ell_used)  # (K_ctx,)
            e_mean = k_eval @ alpha                                 # (D,)

            # GP residual variance (function variance)
            # var_f = k(qq) - k^T (K+σ^2I)^{-1} k
            u = solve_triangular(Lfac, k_eval, lower=lower, check_finite=False)
            quad = float(np.dot(u, u))
            var_f = max(0.0, self._k_qq(psi_q) - quad)

            if return_var:
                vars_out[h] = float(var_f)

            # trust gating from variance
            w_gate = self._gate_from_var(var_f)
            w_eff = w_ctx_base * w_gate

            # optionally sample residual
            e_use = e_mean
            if sample:
                var_y = var_f + (self.gp_noise2 if include_obs_noise else 0.0)
                var_y = max(0.0, float(var_y))
                if var_y > 0:
                    e_use = e_mean + np.sqrt(var_y) * rng.standard_normal(size=(self.D,))

            # combine
            y_c = y_glob_c + w_eff * e_use
            y = (y_c + self.mu_X) if self.center_outputs else y_c

            preds[h] = y

            # update rolling window
            if self.L > 1:
                cur[:-1] = cur[1:]
            cur[-1] = y

        if return_var:
            if H == 1:
                return preds[0], vars_out
            return preds, vars_out
        return preds[0] if H == 1 else preds

    def _rollout_baseline(self, seed_LD: np.ndarray, H: int) -> np.ndarray:
        """
        Baseline AR rollout only: NLSA encode + GPLM decode.
        """
        cur = np.asarray(seed_LD, dtype=np.float64).copy()
        out = np.zeros((H, self.D), dtype=np.float64)

        for h in range(int(H)):
            psi = self.base.encode_window(cur)
            y_c = self._baseline_centered_one(psi)
            y = (y_c + self.mu_X) if self.center_outputs else y_c
            out[h] = y

            if self.L > 1:
                cur[:-1] = cur[1:]
            cur[-1] = y

        return out


__all__ = ["LISA"]
