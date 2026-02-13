# src/utils.py
from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any, Dict, Iterable, Iterator, Optional, Sequence, Tuple, Union

import numpy as np

ArrayLike = Union[np.ndarray, Sequence[float]]


# -------------------------
# Basic math helpers
# -------------------------
def ensure_2d(X: np.ndarray) -> np.ndarray:
    """Ensure X is 2D: (D,) -> (1,D)."""
    X = np.asarray(X)
    return X[None, :] if X.ndim == 1 else X


def as_contig_f32(X: np.ndarray) -> np.ndarray:
    """Contiguous float32 array (good default for ANN + kernels)."""
    return np.ascontiguousarray(np.asarray(X, dtype=np.float32))


def sqdist_ab(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Squared Euclidean distances between rows:
      A: (a,d), B: (b,d) -> D2: (a,b)
    """
    A = np.asarray(A)
    B = np.asarray(B)
    A2 = np.sum(A * A, axis=1, keepdims=True)          # (a,1)
    B2 = np.sum(B * B, axis=1, keepdims=True).T        # (1,b)
    G = A @ B.T                                        # (a,b)
    return np.maximum(A2 + B2 - 2.0 * G, 0.0)


def rbf_from_D2(D2: np.ndarray, *, beta: float, eps: float) -> np.ndarray:
    """RBF kernel weights from squared distances."""
    eps = float(eps)
    if eps <= 0:
        raise ValueError("eps must be > 0")
    return np.exp(-float(beta) * (np.asarray(D2) / eps))


# -------------------------
# ε heuristics
# -------------------------
def median_eps_from_knn_d2(D2_iK: np.ndarray, *, use_kth: bool = True) -> float:
    """
    Median bandwidth from kNN squared distances.
    D2_iK: (N,K) squared distances to K nearest neighbors (excluding self).

    - use_kth=True: use the Kth neighbor distance per point, then median over points
    - use_kth=False: use all distances, then median
    """
    D2_iK = np.asarray(D2_iK)
    if D2_iK.size == 0:
        return 1.0
    v = D2_iK[:, -1] if use_kth else D2_iK.reshape(-1)
    eps = float(np.median(v))
    return max(eps, 1e-12)


def median_eps_from_pairs(X: np.ndarray, *, max_pairs: int = 200_000, seed: int = 0) -> float:
    """
    Median of random-pair squared distances (rough fallback if you don't have kNN distances).
    """
    X = np.asarray(X)
    N = X.shape[0]
    if N < 2:
        return 1.0

    rng = np.random.default_rng(seed)
    p = int(min(max_pairs, N * (N - 1) // 2))

    i = rng.integers(0, N, size=p, endpoint=False)
    j = rng.integers(0, N, size=p, endpoint=False)
    mask = (i != j)
    i = i[mask]
    j = j[mask]
    if i.size == 0:
        return 1.0

    D2 = np.sum((X[i] - X[j]) ** 2, axis=1)
    eps = float(np.median(D2))
    return max(eps, 1e-12)


# -------------------------
# Inducing / landmark selection
# -------------------------
def fps_indices(X: np.ndarray, m: int, *, seed: int = 0) -> np.ndarray:
    """
    Farthest Point Sampling indices (O(N*m)).
    Good for space-filling inducing points / landmarks.

    X: (N,d)
    Returns idx: (m,)
    """
    X = np.asarray(X)
    N = X.shape[0]
    m = int(min(max(1, m), N))

    rng = np.random.default_rng(seed)
    idx = np.empty(m, dtype=np.int64)

    idx[0] = int(rng.integers(0, N))
    d2 = np.sum((X - X[idx[0]]) ** 2, axis=1)

    for t in range(1, m):
        idx[t] = int(np.argmax(d2))
        new_d2 = np.sum((X - X[idx[t]]) ** 2, axis=1)
        d2 = np.minimum(d2, new_d2)

    return idx


# -------------------------
# Batching utilities
# -------------------------
def batched_range(n: int, batch_size: int) -> Iterator[Tuple[int, int]]:
    """Yield (start, end) slices covering [0, n) in batches."""
    bs = int(batch_size)
    if bs <= 0:
        raise ValueError("batch_size must be > 0")
    for s in range(0, int(n), bs):
        yield s, min(int(n), s + bs)


def batch_iter(X: np.ndarray, batch_size: int) -> Iterator[np.ndarray]:
    """Yield contiguous batches from X."""
    X = np.asarray(X)
    for s, e in batched_range(X.shape[0], batch_size):
        yield X[s:e]


# -------------------------
# Metrics
# -------------------------
def rmse(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a)
    b = np.asarray(b)
    return float(np.sqrt(np.mean((a - b) ** 2)))


def mse(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a)
    b = np.asarray(b)
    return float(np.mean((a - b) ** 2))


# -------------------------
# Device helpers (JAX optional)
# -------------------------
def get_jax_device(prefer: str = "auto"):
    """
    Safe JAX device selection.
    prefer: "auto" | "gpu" | "cpu"
    - returns a jax Device if jax is installed, else None
    """
    prefer = (prefer or "auto").lower()
    try:
        import jax  # local import
    except Exception:
        return None

    devs = jax.devices()
    gpu = [d for d in devs if d.platform == "gpu"]
    cpu = [d for d in devs if d.platform == "cpu"]

    if prefer in ("auto", "gpu"):
        return gpu[0] if gpu else cpu[0] if cpu else devs[0]
    if prefer == "cpu":
        return cpu[0] if cpu else devs[0]
    # fallback
    return gpu[0] if gpu else cpu[0] if cpu else devs[0]


# -------------------------
# JSON helpers (for configs)
# -------------------------
def to_jsonable(x: Any) -> Any:
    """
    Convert common objects (numpy scalars/arrays, dataclasses) into JSON-serializable types.
    """
    if is_dataclass(x):
        return {k: to_jsonable(v) for k, v in asdict(x).items()}

    if isinstance(x, (np.floating, np.integer)):
        return x.item()

    if isinstance(x, np.ndarray):
        # prefer list for small arrays; for large arrays you typically store separately
        return x.tolist()

    if isinstance(x, dict):
        return {str(k): to_jsonable(v) for k, v in x.items()}

    if isinstance(x, (list, tuple)):
        return [to_jsonable(v) for v in x]

    return x
