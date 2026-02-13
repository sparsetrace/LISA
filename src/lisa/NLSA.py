# nlsa_encoder.py
# ============================================================
# NLSA Encoder (Diffusion Maps on Hankel windows)
#
# - Learns diffusion coordinates ψ_T for each training Hankel window
# - Supports Nyström out-of-sample embedding for new windows
#
# This is intentionally an "encoder only":
#   - no decoder
#   - no forecasting
#   - just diffusion maps / NLSA coordinates + Nyström extension
#
# ============================================================

from __future__ import annotations

import numpy as np
from scipy.signal import fftconvolve
from scipy.sparse.linalg import eigsh


# ============================================================
# FFT-safe helpers (NO feature mixing)
# ============================================================

def window_norms_sq(R_tX: np.ndarray, L: int) -> np.ndarray:
    """
    r2_T = ||window_T||^2 for all Hankel windows.
    R_tX: (N,D), returns (K,) with K=N-L+1.
    """
    R_tX = np.asarray(R_tX, dtype=float)
    if R_tX.ndim == 1:
        R_tX = R_tX[:, None]
    s_t = np.sum(R_tX * R_tX, axis=1)  # (N,)
    return fftconvolve(s_t, np.ones(L, dtype=float), mode="valid")  # (K,)


def window_dot_all(R_tX: np.ndarray, W_cX: np.ndarray) -> np.ndarray:
    """
    Dot products between every training window of R_tX and a query window W_cX:

      col_T = <window_T, W> = sum_{c,X} R_{T+c,X} * W_{c,X}

    Returns col_T shape (K,), K=N-L+1.

    IMPORTANT: Channel-safe (no feature mixing) by summing per-channel convolutions.
    """
    R_tX = np.asarray(R_tX, dtype=float)
    W_cX = np.asarray(W_cX, dtype=float)
    if R_tX.ndim == 1:
        R_tX = R_tX[:, None]
    if W_cX.ndim == 1:
        W_cX = W_cX[:, None]

    N, D = R_tX.shape
    L, Dw = W_cX.shape
    if Dw != D:
        raise ValueError(f"W has D={Dw} but R has D={D}")

    K = N - L + 1
    if K <= 0:
        raise ValueError(f"Need N={N} >= L={L}")

    col = np.zeros(K, dtype=float)
    W_rev = W_cX[::-1, :]  # flip in time
    for x in range(D):
        col += fftconvolve(R_tX[:, x], W_rev[:, x], mode="valid")
    return col


def build_dense_gram(R_tX: np.ndarray, L: int) -> np.ndarray:
    """
    Dense Gram matrix G_{TT'} = <window_T, window_T'>.

    Complexity: O(K^2 * D * log N) due to looping over T' and FFTing each channel.
    """
    R_tX = np.asarray(R_tX, dtype=float)
    if R_tX.ndim == 1:
        R_tX = R_tX[:, None]
    N, D = R_tX.shape
    K = N - L + 1
    if K <= 0:
        raise ValueError(f"Need N={N} >= L={L}")

    G = np.zeros((K, K), dtype=float)
    for Tprime in range(K):
        W = R_tX[Tprime : Tprime + L, :]  # (L,D)
        G[:, Tprime] = window_dot_all(R_tX, W)

    # symmetrize for numerical cleanliness
    return 0.5 * (G + G.T)


# ============================================================
# NLSA Encoder only
# ============================================================

class NLSA:
    """
    Dense NLSA / Diffusion Maps encoder on Hankel windows.

    Training series:
      F_tX : (N,D)
      windows: W_T = [F_T, ..., F_{T+L-1}]  -> T=0..K-1, K=N-L+1

    Kernel:
      K(T,T') = exp(-beta * ||W_T - W_T'||^2)

    Diffusion normalization (alpha):
      K_alpha = K / (q(T)^alpha q(T')^alpha),  q(T) = sum_{T'} K(T,T')
      d(T) = sum_{T'} K_alpha(T,T')
      P_sym = d^{-1/2} K_alpha d^{-1/2}   (symmetric)

    Embedding:
      Compute top eigenpairs of P_sym:
        P_sym φ_j = λ_j φ_j

      Diffusion coordinates on training windows:
        ψ_j(T) = d(T)^{-1/2} φ_j(T)

    Out-of-sample (Nyström):
      Given query window W_q:
        k(q,T) = exp(-beta * ||W_q - W_T||^2)
      Normalize like training:
        k_alpha(q,T) = k(q,T) / (q(q)^alpha q(T)^alpha)
        P(q,T) = k_alpha(q,T) / d(q)

      Nyström extension:
        ψ_q(j) = sum_T P(q,T) * ψ_j(T) / λ_j

    Notes:
      - This is encoder-only: no decoder, no forecasting.
      - Dense KxK matrices => O(K^2) memory.
    """

    def __init__(
        self,
        F_tX: np.ndarray,
        L: int,
        rank: int = 20,
        beta: float | None = None,
        alpha: float = 1.0,
        center: bool = True,
        drop_first: bool = True,
        max_K_dense: int = 60000,
        beta_sample_pairs: int = 20000,
        seed: int = 0,
    ):
        R = np.asarray(F_tX, dtype=float)
        if R.ndim == 1:
            R = R[:, None]

        self.center = bool(center)
        self.mu_ = R.mean(axis=0, keepdims=True) if self.center else np.zeros((1, R.shape[1]))
        self.R_ = R - self.mu_ if self.center else R

        self.N_, self.D_ = self.R_.shape
        self.L = int(L)
        if self.N_ < self.L:
            raise ValueError(f"N={self.N_} must be >= L={self.L}")
        self.K_ = self.N_ - self.L + 1

        self.rank_req_ = int(rank)
        self.beta_in_ = beta
        self.alpha_ = float(alpha)
        self.drop_first_ = bool(drop_first)

        self.beta_sample_pairs_ = int(beta_sample_pairs)
        self.rng_ = np.random.default_rng(int(seed))

        # learned artifacts
        self.r2_T_ = None            # (K,)
        self.G_ = None               # (K,K) Gram
        self.beta_ = None
        self.K_T_ = None             # raw kernel row-sums q(T)
        self.d_T_ = None             # alpha-normalized degree d(T)
        self.inv_sqrt_d_ = None      # d(T)^(-1/2)
        self.lam_ = None             # (r,) eigenvalues
        self.phi_ = None             # (K,r) symmetric eigvecs
        self.psi_ = None             # (K,r) diffusion coords

        self.fit()

    # -------------------------
    # training
    # -------------------------

    def _choose_beta(self, D2: np.ndarray) -> float:
        """
        Pick beta via median heuristic on sampled off-diagonal distances,
        unless beta was provided.
        """
        if self.beta_in_ is not None:
            return float(self.beta_in_)

        K = self.K_
        M_max = K * (K - 1) // 2
        M = min(self.beta_sample_pairs_, M_max)
        if M <= 0:
            return 1.0

        ii = self.rng_.integers(0, K, size=M)
        jj = self.rng_.integers(0, K, size=M)
        mask = ii != jj
        ii, jj = ii[mask], jj[mask]
        if ii.size == 0:
            return 1.0

        med = np.median(D2[ii, jj])
        return 1.0 / (med + 1e-12)

    def fit(self) -> "NLSAEncoder":
        # window norms
        self.r2_T_ = window_norms_sq(self.R_, self.L)  # (K,)

        # dense Gram and distances (FFT-safe)
        self.G_ = build_dense_gram(self.R_, self.L)
        D2 = self.r2_T_[:, None] + self.r2_T_[None, :] - 2.0 * self.G_
        np.maximum(D2, 0.0, out=D2)

        # beta
        self.beta_ = self._choose_beta(D2)

        # Gaussian kernel on windows
        Kmat = np.exp(-self.beta_ * D2)  # (K,K)

        # diffusion maps normalization
        K_T = Kmat.sum(axis=1) + 1e-18  # q(T)
        KTa = K_T ** self.alpha_
        Kalpha = Kmat / (KTa[:, None] * KTa[None, :])

        d_T = Kalpha.sum(axis=1) + 1e-18
        inv_sqrt_d = 1.0 / np.sqrt(d_T)

        Psym = (inv_sqrt_d[:, None] * Kalpha) * inv_sqrt_d[None, :]

        # eigendecomp of symmetric operator: largest eigenvalues
        k = min(self.rank_req_ + (1 if self.drop_first_ else 0), self.K_ - 1)
        if k <= 0:
            self.lam_ = np.zeros((0,), dtype=float)
            self.phi_ = np.zeros((self.K_, 0), dtype=float)
            self.psi_ = np.zeros((self.K_, 0), dtype=float)
            self.K_T_ = K_T
            self.d_T_ = d_T
            self.inv_sqrt_d_ = inv_sqrt_d
            return self

        w, V = eigsh(Psym, k=k, which="LA")

        # sort descending
        idx = np.argsort(w)[::-1]
        w = w[idx]
        V = V[:, idx]

        # drop trivial mode (lambda ~ 1)
        if self.drop_first_ and w.size > 0:
            w = w[1:]
            V = V[:, 1:]

        self.lam_ = w
        self.phi_ = V
        self.K_T_ = K_T
        self.d_T_ = d_T
        self.inv_sqrt_d_ = inv_sqrt_d

        # diffusion coordinates ψ(T,j) = d(T)^(-1/2) φ(T,j)
        self.psi_ = inv_sqrt_d[:, None] * V  # (K,r)
        return self

    # -------------------------
    # encoding (Nyström)
    # -------------------------

    def encode_window(self, W_cX: np.ndarray) -> np.ndarray:
        """
        Encode ONE query window (L,D) into diffusion coordinates ψ_q (r,).

        Nyström:
          ψ_q = P(q,T) @ (ψ_T / λ)
        """
        if self.psi_ is None or self.lam_ is None or self.psi_.shape[1] == 0:
            return np.zeros((0,), dtype=float)

        W = np.asarray(W_cX, dtype=float)
        if W.ndim == 1:
            W = W[:, None]
        if W.shape != (self.L, self.D_):
            raise ValueError(f"Expected window shape {(self.L, self.D_)}, got {W.shape}")

        # center consistently
        Wc = W - self.mu_ if self.center else W

        # dot products with all training windows (FFT-safe)
        col = window_dot_all(self.R_, Wc)  # (K,)

        r2_q = float(np.sum(Wc * Wc))
        D2 = r2_q + self.r2_T_ - 2.0 * col
        np.maximum(D2, 0.0, out=D2)

        k_qT = np.exp(-self.beta_ * D2)  # (K,)

        # alpha normalization query->train
        Kq = float(np.sum(k_qT)) + 1e-18
        KTa = self.K_T_ ** self.alpha_
        Kqa = (Kq ** self.alpha_)
        k_qT_alpha = k_qT / (Kqa * KTa)

        dq = float(np.sum(k_qT_alpha)) + 1e-18
        P_qT = k_qT_alpha / dq  # (K,)

        # Nyström extension: ψ_q = Σ_T P(q,T) ψ(T)/λ
        lam_safe = np.maximum(self.lam_, 1e-12)
        scale = self.psi_ / lam_safe[None, :]  # (K,r)
        psi_q = P_qT @ scale  # (r,)
        return psi_q

    def encode_windows(self, W_BLX: np.ndarray) -> np.ndarray:
        """
        Encode a batch of windows.

        Input:
          W_BLX: (B,L,D)

        Output:
          Psi_Br: (B,r)
        """
        W = np.asarray(W_BLX, dtype=float)
        if W.ndim != 3:
            raise ValueError("encode_windows expects shape (B,L,D).")
        B = W.shape[0]
        r = 0 if self.psi_ is None else int(self.psi_.shape[1])
        out = np.zeros((B, r), dtype=float)
        for b in range(B):
            out[b] = self.encode_window(W[b])
        return out

    def encode_series(self, F_aX: np.ndarray) -> np.ndarray:
        """
        Encode all Hankel windows of a NOVEL series F_aX.

        F_aX: (N,D) with N >= L

        Returns:
          Psi: (K_n, r) where K_n = N-L+1
        """
        F = np.asarray(F_aX, dtype=float)
        if F.ndim == 1:
            F = F[:, None]
        N, D = F.shape
        if D != self.D_:
            raise ValueError(f"Expected D={self.D_}, got {D}.")
        if N < self.L:
            raise ValueError(f"Need N >= L={self.L}.")

        # build windows naively
        K_n = N - self.L + 1
        r = 0 if self.psi_ is None else int(self.psi_.shape[1])
        out = np.zeros((K_n, r), dtype=float)
        for t0 in range(K_n):
            out[t0] = self.encode_window(F[t0 : t0 + self.L, :])
        return out


__all__ = ["NLSAEncoder"]
