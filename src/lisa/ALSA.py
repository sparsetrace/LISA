# ALSA.py
# ============================================================
# ALSA = NLSAEncoder + GPLM baseline + in-context attention residual mixing
#
# Key idea:
#   ALSA = global predictor + analog/attention-like correction in diffusion coords ψ
#
# Behavior:
#   - if prefix length ell == L: baseline-only AR rollout (NLSA+GPLM)
#   - if ell > L: ALSA correction enabled:
#        E_ctx = Y_true - Y_glob
#        p(q,b) ∝ exp(-||ψ_q-ψ_b||^2 / (2 ell_attn^2))
#        y = y_glob + gamma_ctx * Σ_b p(q,b) E_ctx[b]
#
# This is a "Markovian" IC method because p(q,·) is a probability distribution.
#
# Repo expectations:
#   - NLSAEncoder provides:
#        .L, .K_, .psi_, .mu_ and .encode_window(window_LD)->(r,)
#   - GPLM provides:
#        GPLM(X_latent, Y_ambient, center_X=False, ...) and callable predict
#
# Dependencies:
#   numpy (scipy optional for pdist if you want it, but not required)
# ============================================================

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


from .NLSA import NLSA  # type: ignore
from .GPLM import GPLM  # type: ignore


Array = np.ndarray


# ============================================================
# Small utilities
# ============================================================

def _as_2d(X: Array) -> Array:
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    return X


def _sliding_windows(F_tD: Array, L: int) -> Array:
    """
    Return windows W (K,L,D) from F (N,D) with K=N-L+1.
    Handles numpy stride ordering differences.
    """
    F = _as_2d(F_tD)
    W = sliding_window_view(F, window_shape=int(L), axis=0)

    # W can be (K,L,D) or (K,D,L)
    a, b = W.shape[1], W.shape[2]
    if (a, b) == (L, F.shape[1]):
        return np.ascontiguousarray(W)
    if (a, b) == (F.shape[1], L):
        return np.ascontiguousarray(np.transpose(W, (0, 2, 1)))
    raise ValueError(f"Unexpected window shape {W.shape} for L={L}, D={F.shape[1]}.")


def _sample_pairwise_d2(X: Array, m: int = 1024, seed: int = 0) -> Array:
    """
    Sample pairwise squared distances for bandwidth estimation.
    Returns 1D array of sampled d^2 values.
    """
    X = np.asarray(X, dtype=np.float64)
    n = X.shape[0]
    if n <= 1:
        return np.array([1.0], dtype=np.float64)

    rng = np.random.default_rng(int(seed))
    idx = rng.choice(n, size=min(int(m), n), replace=False)
    Xs = X[idx]

    # Sample random pairs among Xs
    ns = Xs.shape[0]
    M = min(20000, ns * (ns - 1) // 2)
    ii = rng.integers(0, ns, size=M)
    jj = rng.integers(0, ns, size=M)
    mask = ii != jj
    ii, jj = ii[mask], jj[mask]
    if ii.size == 0:
        return np.array([1.0], dtype=np.float64)

    diff = Xs[ii] - Xs[jj]
    return np.sum(diff * diff, axis=1)


def _median_ell_from_d2(d2: Array, q: float = 0.5, eps: float = 1e-12) -> float:
    """
    If kernel is exp(-||x-y||^2/(2 ell^2)), heuristic:
      ell^2 ≈ quantile(d^2)/2
    """
    d2 = np.asarray(d2, dtype=np.float64)
    if d2.size == 0:
        return 1.0
    v = float(np.quantile(d2, float(q)))
    return float(np.sqrt(max(v / 2.0, eps)))


def _softmax_weights_from_d2(d2: Array, ell: float, eps: float = 1e-18) -> Array:
    """
    p_i ∝ exp(-0.5 * d2_i / ell^2)
    """
    d2 = np.asarray(d2, dtype=np.float64)
    ell2 = float(ell) * float(ell) + 1e-18
    logits = -0.5 * d2 / ell2
    logits -= float(np.max(logits))
    w = np.exp(logits)
    s = float(np.sum(w))
    return w / (s + eps)


# ============================================================
# ALSA config (sister to LISAConfig)
# ============================================================

@dataclass
class ALSAConfig:
    # Encoder (NLSA)
    L: int = 128
    rank: int = 32
    beta: Optional[float] = None
    alpha: float = 1.0
    center: bool = True
    drop_first: bool = True
    max_K_dense: int = 6000
    seed: int = 0

    # Baseline GPLM decoder psi->next sample
    gplm_kwargs: Optional[Dict[str, Any]] = None

    # psi preprocessing
    whiten_psi: bool = True

    # In-context attention correction
    min_ctx_windows: Optional[int] = None     # default r+1
    ctx_k0: float = 10.0                      # gamma_ctx = K_ctx/(K_ctx + ctx_k0)

    attn_topk: Optional[int] = 64             # None = dense over all context
    attn_ell: Optional[float] = None          # if None -> estimate from training ψ geometry
    attn_ell_q: float = 0.5                   # quantile used for attn_ell estimation


# ============================================================
# ALSA main class
# ============================================================

class ALSA:
    """
    ALSA = NLSAEncoder + GPLM baseline + attention-like (Markov) IC correction.

    Training:
      1) Fit NLSAEncoder on training series
      2) Fit GPLM baseline on diffusion coords:
           ψ_train(window) -> next sample

    Inference:
      - If prefix length ell == L: baseline-only AR rollout
      - If ell > L:
          build context windows within prefix
          residual table E_ctx = Y_true - Y_glob
          attention weights p(q,ctx) from ψ-distances
          correction = Σ p * E_ctx
          y = y_glob + gamma_ctx * correction
    """

    def __init__(self, F_train: Array, *, config: Optional[ALSAConfig] = None, **kwargs: Any):
        if config is None:
            config = ALSAConfig(**kwargs)
        self.cfg = config

        F_train = _as_2d(F_train)
        self.D = int(F_train.shape[1])
        self.L = int(self.cfg.L)

        # ---- 1) Encoder ----
        self.enc = NLSA(
            F_train,
            L=int(self.cfg.L),
            rank=int(self.cfg.rank),
            beta=self.cfg.beta,
            alpha=float(self.cfg.alpha),
            center=bool(self.cfg.center),
            drop_first=bool(self.cfg.drop_first),
            max_K_dense=int(self.cfg.max_K_dense),
            seed=int(self.cfg.seed),
        )

        self.r = 0 if self.enc.psi_ is None else int(self.enc.psi_.shape[1])
        if self.r <= 0:
            raise ValueError("NLSA produced r=0 diffusion dims. Increase rank or set drop_first=False.")

        # Output mean from encoder centering
        self.center = bool(self.cfg.center)
        self.mu_X = self.enc.mu_.reshape(-1).astype(np.float64) if self.center else np.zeros((self.D,), dtype=np.float64)

        # ---- 2) Baseline GPLM: psi -> next sample (centered space) ----
        K = int(self.enc.K_)
        n_pairs = K - 1
        Psi_tr = np.asarray(self.enc.psi_[:n_pairs, :], dtype=np.float64)  # (K-1, r)

        # Targets in centered space
        F_centered = F_train - self.mu_X[None, :] if self.center else F_train
        Y_tr = np.asarray(F_centered[self.L : self.L + n_pairs, :], dtype=np.float64)  # (K-1, D)

        # Optional whitening of psi for both:
        #   - baseline decoder geometry
        #   - attention geometry
        self.whiten_psi = bool(self.cfg.whiten_psi)
        if self.whiten_psi:
            self.psi_mu = Psi_tr.mean(axis=0, keepdims=True)
            self.psi_sd = Psi_tr.std(axis=0, keepdims=True) + 1e-12
            Psi_tr_w = (Psi_tr - self.psi_mu) / self.psi_sd
        else:
            self.psi_mu = np.zeros((1, self.r), dtype=np.float64)
            self.psi_sd = np.ones((1, self.r), dtype=np.float64)
            Psi_tr_w = Psi_tr

        gkw = {} if self.cfg.gplm_kwargs is None else dict(self.cfg.gplm_kwargs)
        # Critical: outputs already centered -> don't recenter in GPLM
        gkw.setdefault("center_X", False)
        gkw.setdefault("seed", int(self.cfg.seed))
        gkw.setdefault("sigma2", 1e-4)
        gkw.setdefault("jitter", 1e-8)
        gkw.setdefault("m", min(1024, n_pairs))
        gkw.setdefault("inducing", "kmeans_medoids")

        self.baseline = GPLM(Psi_tr_w, Y_tr, **gkw)

        # Attention hyperparams
        self.attn_topk = None if self.cfg.attn_topk is None else int(self.cfg.attn_topk)

        if self.cfg.min_ctx_windows is None:
            self.min_ctx_windows = max(1, self.r + 1)
        else:
            self.min_ctx_windows = int(self.cfg.min_ctx_windows)

        self.ctx_k0 = float(self.cfg.ctx_k0)

        # Pick attention ell if not provided (based on training psi geometry)
        if self.cfg.attn_ell is None:
            d2 = _sample_pairwise_d2(Psi_tr_w, m=1024, seed=int(self.cfg.seed) + 123)
            self.attn_ell = _median_ell_from_d2(d2, q=float(self.cfg.attn_ell_q))
        else:
            self.attn_ell = float(self.cfg.attn_ell)

    # -------------------------
    # psi processing
    # -------------------------

    def _psi_whiten(self, psi: Array) -> Array:
        psi = np.asarray(psi, dtype=np.float64)
        if psi.ndim == 1:
            psi = psi[None, :]
        if self.whiten_psi:
            return (psi - self.psi_mu) / self.psi_sd
        return psi

    # -------------------------
    # baseline step
    # -------------------------

    def predict_one_step_baseline(self, W_LD: Array) -> Array:
        """
        Baseline one-step prediction:
          window (L,D) -> psi -> GPLM -> next sample (D)
        """
        W = np.asarray(W_LD, dtype=float)
        if W.ndim == 1:
            W = W[:, None]
        if W.shape != (self.L, self.D):
            raise ValueError(f"Expected window shape {(self.L, self.D)}, got {W.shape}")

        psi = self.enc.encode_window(W)        # (r,)
        psi_w = self._psi_whiten(psi)[0]       # (r,)
        y_c = np.asarray(self.baseline(psi_w), dtype=np.float64).reshape(-1)  # centered
        return (y_c + self.mu_X) if self.center else y_c

    # -------------------------
    # main IC rollout
    # -------------------------

    def __call__(self, prefix: Array, steps: int = 1) -> Array:
        """
        Forecast from prefix.

        prefix: (ell,D), ell >= L
        steps: horizon H

        Returns:
          (H,D) (or (D,) if H==1)
        """
        prefix = _as_2d(prefix)
        ell, D = prefix.shape
        if D != self.D:
            raise ValueError(f"ALSA trained with D={self.D}, got prefix D={D}.")
        if ell < self.L:
            raise ValueError(f"Need prefix length ell >= L={self.L}.")

        H = int(steps)
        if H <= 0:
            return np.zeros((0, self.D), dtype=np.float64)

        # Seed window is always last L
        cur = prefix[-self.L :, :].copy()

        # ----------------------------------------------------
        # If no context (ell == L), run baseline-only rollout
        # ----------------------------------------------------
        if ell == self.L:
            out = self._rollout_baseline(cur, H)
            return out[0] if H == 1 else out

        # ----------------------------------------------------
        # Build context windows and residual table from prefix
        # ----------------------------------------------------
        W_all = _sliding_windows(prefix, self.L)  # (K_n, L, D)
        K_n = int(W_all.shape[0])
        K_ctx = K_n - 1

        # If too little context, baseline-only
        if K_ctx < self.min_ctx_windows:
            out = self._rollout_baseline(cur, H)
            return out[0] if H == 1 else out

        W_ctx = np.ascontiguousarray(W_all[:K_ctx, :, :])              # (K_ctx,L,D)
        Y_true = np.asarray(prefix[self.L : self.L + K_ctx, :], float) # (K_ctx,D)

        # Center targets
        Y_true_c = (Y_true - self.mu_X[None, :]) if self.center else Y_true

        # Encode context psi
        Psi_ctx = np.zeros((K_ctx, self.r), dtype=np.float64)
        for i in range(K_ctx):
            Psi_ctx[i] = self.enc.encode_window(W_ctx[i])
        Psi_ctx_w = self._psi_whiten(Psi_ctx)  # (K_ctx,r)

        # Baseline on context
        Y_glob_ctx_c = np.asarray(self.baseline(Psi_ctx_w), dtype=np.float64)  # (K_ctx,D)

        # Residual table
        E_ctx = Y_true_c - Y_glob_ctx_c  # (K_ctx,D)

        # Context gate gamma in [0,1]
        gamma_ctx = float(K_ctx) / float(K_ctx + self.ctx_k0) if self.ctx_k0 > 0 else 1.0

        # ----------------------------------------------------
        # Rollout with attention residual correction
        # ----------------------------------------------------
        out = np.zeros((H, self.D), dtype=np.float64)

        for h in range(H):
            psi_q = self.enc.encode_window(cur)        # (r,)
            psi_q_w = self._psi_whiten(psi_q)[0]       # (r,)

            # Baseline prediction
            y_glob_c = np.asarray(self.baseline(psi_q_w), dtype=np.float64).reshape(-1)  # (D,)

            # Attention weights on psi distances (Markov weights)
            diff = Psi_ctx_w - psi_q_w[None, :]  # (K_ctx,r)
            d2 = np.einsum("kr,kr->k", diff, diff, optimize=True)

            if self.attn_topk is not None and self.attn_topk < K_ctx:
                k = int(self.attn_topk)
                idx = np.argpartition(d2, kth=k - 1)[:k]
                w = _softmax_weights_from_d2(d2[idx], self.attn_ell)
                e_hat = w @ E_ctx[idx]  # (D,)
            else:
                w = _softmax_weights_from_d2(d2, self.attn_ell)
                e_hat = w @ E_ctx       # (D,)

            # Combine
            y_c = y_glob_c + gamma_ctx * e_hat
            y = (y_c + self.mu_X) if self.center else y_c

            out[h] = y

            # Advance window
            if self.L > 1:
                cur[:-1] = cur[1:]
            cur[-1] = y

        return out[0] if H == 1 else out

    def _rollout_baseline(self, seed_LD: Array, H: int) -> Array:
        """
        Baseline-only AR rollout (no IC correction).
        """
        cur = np.asarray(seed_LD, dtype=float).copy()
        out = np.zeros((int(H), self.D), dtype=np.float64)

        for h in range(int(H)):
            y = self.predict_one_step_baseline(cur)
            out[h] = y
            if self.L > 1:
                cur[:-1] = cur[1:]
            cur[-1] = y

        return out

    def __repr__(self) -> str:
        return (
            f"ALSA(L={self.L}, D={self.D}, r={self.r}, "
            f"center={self.center}, whiten_psi={self.whiten_psi}, "
            f"attn_topk={self.attn_topk}, attn_ell={self.attn_ell:.4g})"
        )


__all__ = ["ALSA", "ALSAConfig"]
