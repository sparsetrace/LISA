# Minimal usage
# from metrics import compare_all, mse_by_horizon
# 
# metrics = compare_all(truth, pred, burn_in=0, acf_max_lag=200, mmd_sample=1024, seed=0)
# print(metrics)
# 
# mseh = mse_by_horizon(truth, pred)  # per-step MSE curve


# metrics.py
from __future__ import annotations

import numpy as np
from typing import Dict, Optional, Tuple


def _as_2d(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a)
    if a.ndim == 1:
        return a[:, None]
    if a.ndim != 2:
        raise ValueError(f"Expected 1D or 2D array, got shape {a.shape}")
    return a


def _slice_time(
    R: np.ndarray,
    Q: np.ndarray,
    *,
    burn_in: int = 0,
    T_max: Optional[int] = None,
    stride: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    R = _as_2d(R)
    Q = _as_2d(Q)
    if R.shape != Q.shape:
        raise ValueError(f"R and Q must have same shape, got {R.shape} vs {Q.shape}")

    b = int(burn_in)
    s = int(stride)
    if b < 0 or s <= 0:
        raise ValueError("burn_in must be >=0 and stride must be >=1")

    T = R.shape[0]
    t0 = min(b, T)
    t1 = T if T_max is None else min(T, t0 + int(T_max))
    R2 = R[t0:t1:s]
    Q2 = Q[t0:t1:s]
    return R2, Q2


# -----------------------------
# 1) MSE
# -----------------------------
def mse(R_tX: np.ndarray, Q_tX: np.ndarray, *, burn_in: int = 0, T_max: Optional[int] = None, stride: int = 1) -> float:
    R, Q = _slice_time(R_tX, Q_tX, burn_in=burn_in, T_max=T_max, stride=stride)
    return float(np.mean((R - Q) ** 2))


def mse_by_horizon(R_tX: np.ndarray, Q_tX: np.ndarray, *, burn_in: int = 0, T_max: Optional[int] = None, stride: int = 1) -> np.ndarray:
    """
    Per-step MSE across features: returns shape (T',)
    """
    R, Q = _slice_time(R_tX, Q_tX, burn_in=burn_in, T_max=T_max, stride=stride)
    return np.mean((R - Q) ** 2, axis=1)


def mse_per_feature(R_tX: np.ndarray, Q_tX: np.ndarray, *, burn_in: int = 0, T_max: Optional[int] = None, stride: int = 1) -> np.ndarray:
    """
    Per-feature MSE: returns shape (X,)
    """
    R, Q = _slice_time(R_tX, Q_tX, burn_in=burn_in, T_max=T_max, stride=stride)
    return np.mean((R - Q) ** 2, axis=0)


# -----------------------------
# 2) PSD log-MSE (spectrum mismatch)
# -----------------------------
def _hann(n: int) -> np.ndarray:
    if n <= 1:
        return np.ones((n,), dtype=float)
    i = np.arange(n, dtype=float)
    return 0.5 - 0.5 * np.cos(2.0 * np.pi * i / (n - 1))


def psd_logmse(
    R_tX: np.ndarray,
    Q_tX: np.ndarray,
    *,
    burn_in: int = 0,
    T_max: Optional[int] = None,
    stride: int = 1,
    detrend: bool = True,
    window: str = "hann",
    eps: float = 1e-12,
    normalize_per_feature: bool = True,
) -> float:
    """
    Compare log power spectra of R and Q.
    - computes one-sided PSD via rFFT along time for each feature
    - returns mean squared error of log(PSD + eps) over (freq, feature)

    normalize_per_feature=True rescales PSD of each feature to sum to 1
    (so mismatch reflects shape, not total variance).
    """
    R, Q = _slice_time(R_tX, Q_tX, burn_in=burn_in, T_max=T_max, stride=stride)
    T, X = R.shape
    if T < 8:
        return float("nan")

    Rw = R.astype(float)
    Qw = Q.astype(float)

    if detrend:
        Rw = Rw - np.mean(Rw, axis=0, keepdims=True)
        Qw = Qw - np.mean(Qw, axis=0, keepdims=True)

    if window is None or window.lower() == "none":
        w = np.ones((T,), dtype=float)
    elif window.lower() == "hann":
        w = _hann(T)
    else:
        raise ValueError("window must be 'hann' or None/'none'")

    # apply window (broadcast over features)
    Rw = Rw * w[:, None]
    Qw = Qw * w[:, None]

    # rFFT
    FR = np.fft.rfft(Rw, axis=0)
    FQ = np.fft.rfft(Qw, axis=0)

    # raw power
    PR = (np.abs(FR) ** 2) / max(T, 1)
    PQ = (np.abs(FQ) ** 2) / max(T, 1)

    if normalize_per_feature:
        PR = PR / (np.sum(PR, axis=0, keepdims=True) + eps)
        PQ = PQ / (np.sum(PQ, axis=0, keepdims=True) + eps)

    LR = np.log(PR + eps)
    LQ = np.log(PQ + eps)
    return float(np.mean((LR - LQ) ** 2))


# -----------------------------
# 3) ACF-MSE (autocorrelation mismatch)
# -----------------------------
def _acf_1d(x: np.ndarray, max_lag: int) -> np.ndarray:
    """
    Normalized autocorrelation via FFT, lags 0..max_lag.
    """
    x = np.asarray(x, float)
    n = x.size
    if n == 0:
        return np.zeros((max_lag + 1,), dtype=float)
    x = x - np.mean(x)
    # next power of 2 for speed
    nfft = 1 << int(np.ceil(np.log2(max(1, 2 * n - 1))))
    fx = np.fft.rfft(x, n=nfft)
    acov = np.fft.irfft(fx * np.conj(fx), n=nfft)[: max_lag + 1]
    acov = acov / max(n, 1)
    if acov[0] <= 0:
        return np.zeros((max_lag + 1,), dtype=float)
    return acov / acov[0]


def acf_mse(
    R_tX: np.ndarray,
    Q_tX: np.ndarray,
    *,
    max_lag: int = 200,
    burn_in: int = 0,
    T_max: Optional[int] = None,
    stride: int = 1,
) -> float:
    """
    Mean squared error between autocorrelation functions of R and Q,
    averaged over features and lags 1..max_lag (excluding lag 0).
    """
    R, Q = _slice_time(R_tX, Q_tX, burn_in=burn_in, T_max=T_max, stride=stride)
    T, X = R.shape
    if T < max_lag + 2:
        max_lag = max(1, min(max_lag, T - 2))

    errs = []
    for j in range(X):
        aR = _acf_1d(R[:, j], max_lag=max_lag)
        aQ = _acf_1d(Q[:, j], max_lag=max_lag)
        # exclude lag 0 (always 1)
        errs.append(np.mean((aR[1:] - aQ[1:]) ** 2))
    return float(np.mean(errs)) if errs else float("nan")


# -----------------------------
# 4) MMD² (distribution/attractor fidelity)
# -----------------------------
def _sq_dists(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    # ||a-b||^2 = ||a||^2 + ||b||^2 - 2a·b
    AA = np.sum(A * A, axis=1, keepdims=True)
    BB = np.sum(B * B, axis=1, keepdims=True).T
    D2 = AA + BB - 2.0 * (A @ B.T)
    return np.maximum(D2, 0.0)


def _median_bandwidth(Z: np.ndarray, max_pairs: int = 4096, seed: int = 0, eps: float = 1e-12) -> float:
    rng = np.random.default_rng(seed)
    n = Z.shape[0]
    if n < 2:
        return 1.0
    m = min(max_pairs, n * (n - 1) // 2)
    i = rng.integers(0, n, size=m)
    j = rng.integers(0, n, size=m)
    mask = i != j
    i, j = i[mask], j[mask]
    if i.size == 0:
        return 1.0
    d2 = np.sum((Z[i] - Z[j]) ** 2, axis=1)
    med = np.median(d2)
    # kernel uses exp(-||x-y||^2/(2*sigma^2))
    sigma = np.sqrt(max(med, 0.0) / 2.0 + eps)
    return float(max(sigma, 1e-6))


def mmd_rbf(
    R_tX: np.ndarray,
    Q_tX: np.ndarray,
    *,
    burn_in: int = 0,
    T_max: Optional[int] = None,
    stride: int = 1,
    sample: int = 2048,
    sigma: Optional[float] = None,
    sigma_seed: int = 0,
    unbiased: bool = True,
    seed: int = 0,
) -> float:
    """
    MMD^2 between the empirical distributions of state samples from R and Q.

    Uses an RBF kernel: k(x,y) = exp(-||x-y||^2/(2*sigma^2))

    Notes:
      - O(sample^2) time/memory, so keep sample modest (e.g. 512–4096).
      - If sigma is None, uses median heuristic on pooled samples.
    """
    R, Q = _slice_time(R_tX, Q_tX, burn_in=burn_in, T_max=T_max, stride=stride)
    rng = np.random.default_rng(seed)

    T = R.shape[0]
    if T == 0:
        return float("nan")

    n = min(int(sample), T)
    idxR = rng.choice(T, size=n, replace=False) if n < T else np.arange(T)
    idxQ = rng.choice(T, size=n, replace=False) if n < T else np.arange(T)

    Xs = R[idxR].astype(float)
    Ys = Q[idxQ].astype(float)

    if sigma is None:
        Z = np.vstack([Xs, Ys])
        sigma = _median_bandwidth(Z, seed=sigma_seed)

    sig2 = float(sigma) ** 2
    if sig2 <= 0:
        raise ValueError("sigma must be > 0")

    Kxx = np.exp(-_sq_dists(Xs, Xs) / (2.0 * sig2))
    Kyy = np.exp(-_sq_dists(Ys, Ys) / (2.0 * sig2))
    Kxy = np.exp(-_sq_dists(Xs, Ys) / (2.0 * sig2))

    if unbiased and n > 1:
        np.fill_diagonal(Kxx, 0.0)
        np.fill_diagonal(Kyy, 0.0)
        mmd2 = (Kxx.sum() / (n * (n - 1))) + (Kyy.sum() / (n * (n - 1))) - 2.0 * (Kxy.mean())
    else:
        mmd2 = Kxx.mean() + Kyy.mean() - 2.0 * Kxy.mean()

    return float(max(mmd2, 0.0))






from typing import Optional, Dict, Tuple
import numpy as np

def compare_all_fast(
    R_tX: np.ndarray,
    Q_tX: np.ndarray,
    *,
    dt: float,
    burn_in: int = 0,
    T_max: Optional[int] = None,
    stride: int = 1,
    acf_max_lag: int = 200,
    include_mmd_rff: bool = False,
    mmd_sample: int = 1024,
    mmd_rff_dim: int = 512,
    mmd_sigma: Optional[float] = None,
    seed: int = 0,
    # spectral_kl knobs
    spectral_nperseg: int = 256,
    spectral_mode: str = "js",   # "kl_pq", "kl_qp", "js"
    spectral_eps: float = 1e-12,
    ) -> Dict[str, float]:
    out: Dict[str, float] = {}

    out["mse"] = mse(R_tX, Q_tX, burn_in=burn_in, T_max=T_max, stride=stride)

    # swapped in for psd_logmse:
    out["spectral_kl"] = spectral_kl(
        R_tX, Q_tX,
        dt=dt,
        nperseg=spectral_nperseg,
        eps=spectral_eps,
        mode=spectral_mode,
        per_dim=False,
    )

    out["acf_mse"] = acf_mse_fast(
        R_tX, Q_tX,
        max_lag=acf_max_lag,
        burn_in=burn_in, T_max=T_max, stride=stride,
    )

    if include_mmd_rff:
        out["mmd2_rff"] = mmd_rff(
            R_tX, Q_tX,
            burn_in=burn_in, T_max=T_max, stride=stride,
            sample=mmd_sample, rff_dim=mmd_rff_dim,
            sigma=mmd_sigma,
            seed=seed,
        )

    return out

def compare_all_fast_with_curves(
    R_tX: np.ndarray,
    Q_tX: np.ndarray,
    *,
    dt: float,
    burn_in: int = 0,
    T_max: Optional[int] = None,
    stride: int = 1,
    acf_max_lag: int = 200,
    include_mmd_rff: bool = False,
    mmd_sample: int = 1024,
    mmd_rff_dim: int = 512,
    mmd_sigma: Optional[float] = None,
    seed: int = 0,
    # spectral_kl knobs
    spectral_nperseg: int = 256,
    spectral_mode: str = "js",
    spectral_eps: float = 1e-12,
) -> Tuple[Dict[str, float], Dict[str, np.ndarray]]:
    scalars = compare_all_fast(
        R_tX, Q_tX,
        dt=dt,
        burn_in=burn_in, T_max=T_max, stride=stride,
        acf_max_lag=acf_max_lag,
        include_mmd_rff=include_mmd_rff,
        mmd_sample=mmd_sample, mmd_rff_dim=mmd_rff_dim,
        mmd_sigma=mmd_sigma,
        seed=seed,
        spectral_nperseg=spectral_nperseg,
        spectral_mode=spectral_mode,
        spectral_eps=spectral_eps,
    )

    curves = {
        "mse_by_horizon": mse_by_horizon(R_tX, Q_tX, burn_in=burn_in, T_max=T_max, stride=stride),
        "mse_per_feature": mse_per_feature(R_tX, Q_tX, burn_in=burn_in, T_max=T_max, stride=stride),
    }
    return scalars, curves

from typing import Optional, Dict, Tuple

def acf_mse_fast(
    R_tX: np.ndarray,
    Q_tX: np.ndarray,
    *,
    max_lag: int = 200,
    burn_in: int = 0,
    T_max: Optional[int] = None,
    stride: int = 1,
    eps: float = 1e-12,
    ) -> float:
    R, Q = _slice_time(R_tX, Q_tX, burn_in=burn_in, T_max=T_max, stride=stride)
    T, X = R.shape
    if T < 8:
        return float("nan")
    if T < max_lag + 2:
        max_lag = max(1, min(max_lag, T - 2))

    def acf_all(Z: np.ndarray) -> np.ndarray:
        Z = Z.astype(np.float32)
        Z = Z - Z.mean(axis=0, keepdims=True)
        n = Z.shape[0]
        nfft = 1 << int(np.ceil(np.log2(max(1, 2*n - 1))))
        F = np.fft.rfft(Z, n=nfft, axis=0)
        acov = np.fft.irfft(F * np.conj(F), n=nfft, axis=0)[: max_lag + 1]
        acov = acov / max(n, 1)
        ac0 = acov[0:1, :]  # (1,X)
        return np.where(ac0 > eps, acov / ac0, 0.0)

    aR = acf_all(R)
    aQ = acf_all(Q)
    return float(np.mean((aR[1:] - aQ[1:]) ** 2))

def mmd_rff(
    R_tX: np.ndarray,
    Q_tX: np.ndarray,
    *,
    burn_in: int = 0,
    T_max: Optional[int] = None,
    stride: int = 1,
    sample: int = 2048,
    rff_dim: int = 512,
    sigma: Optional[float] = None,
    seed: int = 0,
    ) -> float:
    R, Q = _slice_time(R_tX, Q_tX, burn_in=burn_in, T_max=T_max, stride=stride)
    rng = np.random.default_rng(seed)
    T, X = R.shape
    if T == 0:
        return float("nan")

    n = min(int(sample), T)
    idxR = rng.choice(T, size=n, replace=False) if n < T else np.arange(T)
    idxQ = rng.choice(T, size=n, replace=False) if n < T else np.arange(T)

    Xs = R[idxR].astype(np.float32)
    Ys = Q[idxQ].astype(np.float32)

    if sigma is None:
        Z = np.vstack([Xs, Ys])
        sigma = _median_bandwidth(Z, seed=seed)

    sigma = float(sigma)
    W = rng.normal(0.0, 1.0 / sigma, size=(X, rff_dim)).astype(np.float32)
    b = rng.uniform(0.0, 2*np.pi, size=(rff_dim,)).astype(np.float32)

    def phi(A):
        return np.sqrt(2.0 / rff_dim) * np.cos(A @ W + b)

    mx = phi(Xs).mean(axis=0)
    my = phi(Ys).mean(axis=0)
    return float(np.mean((mx - my) ** 2))

from scipy.signal import welch

def _psd_prob(x: np.ndarray, fs: float, nperseg: int = 256, eps: float = 1e-12):
    """
    x: (T,) 1D signal
    returns: p(f) normalized PSD over frequency bins (excluding DC)
    """
    f, P = welch(x, fs=fs, nperseg=min(nperseg, x.shape[0]), detrend="constant")
    # drop DC
    f, P = f[1:], P[1:]
    P = np.maximum(P, 0.0)
    P = P + eps
    p = P / np.sum(P)
    return p

def spectral_kl(
    truth: np.ndarray,
    pred: np.ndarray,
    *,
    dt: float,
    nperseg: int = 256,
    eps: float = 1e-12,
    mode: str = "js",   # "kl_pq", "kl_qp", "js"
    per_dim: bool = False,
    ):
    """
    truth, pred: (T, D)
    Returns scalar divergence (averaged over dims), or (D,) if per_dim=True.
    """
    truth = np.asarray(truth, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    assert truth.shape == pred.shape
    T, D = truth.shape
    fs = 1.0 / float(dt)

    out = np.zeros((D,), dtype=np.float64)
    for d in range(D):
        p = _psd_prob(truth[:, d], fs=fs, nperseg=nperseg, eps=eps)
        q = _psd_prob(pred[:, d],  fs=fs, nperseg=nperseg, eps=eps)

        if mode == "kl_pq":
            out[d] = float(np.sum(p * (np.log(p) - np.log(q))))
        elif mode == "kl_qp":
            out[d] = float(np.sum(q * (np.log(q) - np.log(p))))
        elif mode == "js":
            m = 0.5 * (p + q)
            out[d] = 0.5 * float(np.sum(p * (np.log(p) - np.log(m)))) + 0.5 * float(np.sum(q * (np.log(q) - np.log(m))))
        else:
            raise ValueError("mode must be 'kl_pq', 'kl_qp', or 'js'")

    return out if per_dim else float(np.mean(out))

