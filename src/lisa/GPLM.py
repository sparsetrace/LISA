# GPLM.py
# ============================================================
# GPLM: Gaussian-Process-Like Manifold decoder (Nyström KRR / GP)
#
# Core idea:
#   Given latent coordinates R_ix (N,d) and corresponding outputs R_iX (N,D),
#   learn a smooth decoder f: R^d -> R^D using inducing points (Nyström).
#
# This file is designed to be a *building block* for:
#   - DIMA decoding (latent -> ambient)
#   - LISA / ELSA global heads (psi -> next sample)
#   - any kernel-smooth regression w/ optional geometric flow
#
# Main API:
#   - __init__(...) trains automatically
#   - __call__(R_ax, batch_size=None) returns mean predictions
#   - predict(R_ax, return_var=True) returns mean + scalar uncertainty
#   - kernel_mass(R_ax) returns a cheap "support" diagnostic
#   - flow(...) integrates geodesic-ish latent motion under pullback metric
#
# Flow extensions:
#   flow(..., k_tangent=K)           -> projects to top-K metric eigendirections
#   flow(..., force_fn=callable)     -> adds an external force to momentum update
#
# NOTE on geometry:
#   The pullback metric g(r) = J(r)^T J(r) is PSD and can be singular when d > D.
#   This implementation uses robust eigendecomposition fallback when Cholesky fails.
#
# ============================================================

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Literal, Optional, Tuple, Union

import numpy as np
import scipy.linalg as la

from .ann import ANNBackend, make_ann
from .utils import fps_indices, median_eps_from_knn_d2, sqdist_ab

InducingMode = Literal["random_subset", "fps", "kmeans_medoids", "given"]
MetricSolver = Literal["auto", "chol", "eigh"]


def _kmeans2_safe(Z: np.ndarray, m: int, seed: int = 0) -> np.ndarray:
    """
    KMeans centers with a safe fallback.
    Uses scipy.cluster.vq.kmeans2 if available; otherwise samples points.
    """
    Z = np.asarray(Z)
    m = int(min(max(1, m), Z.shape[0]))
    try:
        from scipy.cluster.vq import kmeans2  # type: ignore

        # minit="points" picks initial centers from data -> stable for medoids snapping
        C, _ = kmeans2(Z.astype(np.float64, copy=False), m, minit="points", seed=seed)
        return C.astype(Z.dtype, copy=False)
    except Exception:
        rng = np.random.default_rng(seed)
        idx = rng.choice(Z.shape[0], size=m, replace=False)
        return Z[idx]


@dataclass
class _MetricFact:
    """A small wrapper that can solve g x = b robustly."""

    method: MetricSolver
    g: np.ndarray  # (d,d)
    chol: Optional[Tuple[np.ndarray, bool]] = None
    eigvals: Optional[np.ndarray] = None
    eigvecs: Optional[np.ndarray] = None
    # for pseudo-inverse safety
    eig_clip: float = 1e-12

    def solve(self, b: np.ndarray) -> np.ndarray:
        """Return x = g^{-1} b (vector)."""
        b = np.asarray(b, dtype=np.float64)
        if self.method == "chol" and self.chol is not None:
            return la.cho_solve(self.chol, b, check_finite=False)
        # eigen fallback
        if self.eigvals is None or self.eigvecs is None:
            # last-resort: direct solve
            return np.linalg.solve(self.g, b)
        lam = np.maximum(self.eigvals, float(self.eig_clip))
        V = self.eigvecs
        return V @ ((V.T @ b) / lam)

    def project_topk(self, v: np.ndarray, k: int) -> np.ndarray:
        """
        Project v onto the span of the top-k eigenvectors of g.
        """
        v = np.asarray(v, dtype=np.float64)
        k = int(k)
        if k <= 0:
            return np.zeros_like(v)
        d = v.shape[0]

        # Ensure eigendecomposition exists
        if self.eigvals is None or self.eigvecs is None:
            w, V = np.linalg.eigh(self.g)
            idx = np.argsort(w)[::-1]
            self.eigvals = w[idx]
            self.eigvecs = V[:, idx]

        k = min(k, d)
        Vk = self.eigvecs[:, :k]
        return Vk @ (Vk.T @ v)


class GPLM:
    """
    GPLM = Inducing-point / Nyström GP-like decoder on latents.

    Training:
      R_ix: (N,d) latents
      R_iX: (N,D) ambients

      Choose inducing Z_mx (m << N), typically subset (medoids) of R_ix.

      Latent kernel (unnormalized Gaussian affinity):
        C_im = exp(-β * ||R_ix - Z_mx||^2 / ε)      (N,m)
        W_mn = exp(-β * ||Z_mx - Z_nx||^2 / ε)      (m,m)

      Reduced KRR/GP mean solve:
        M_mX = (C^T C + σ2 W + jitter I)^-1 (C^T (R_iX - mean_X))

      Predict:
        For novel R_ax:
          C_am = exp(-β * ||R_ax - Z_mx||^2 / ε)
          R_aX = C_am M_mX + mean_X

    Uncertainty:
      We provide a cheap scalar "leverage" uncertainty:
        u(r) = C(r)^T A^{-1} C(r)
      (A = C^T C + σ2 W + jitter I)
      and return  σ2 * u(r)  as a proxy variance (scalar per query).

    Geometry:
      flow() integrates geodesic-ish motion on the pullback metric induced by decoder,
      with options:
        - k_tangent projection to avoid drift in nullspace when d > D
        - force_fn to add "prior forces" (e.g. DDPM score, density barrier)
    """

    def __init__(
        self,
        R_ix: np.ndarray,
        R_iX: np.ndarray,
        *,
        # ASCII names (preferred for library APIs)
        beta: float = 1.0,
        # ε estimation
        eps: Optional[float] = None,
        k_eps: int = 256,
        eps_use_kth: bool = True,
        eps_mul: float = 1.0,
        # regularization
        sigma2: float = 1e-5,
        jitter: float = 1e-8,
        # inducing
        m: int = 1024,
        inducing: InducingMode = "kmeans_medoids",
        Z_mx: Optional[np.ndarray] = None,
        seed: int = 0,
        # preprocess
        center_X: bool = True,
        whiten_latent: bool = False,
        dtype: Any = np.float32,
        # compute/memory
        fit_block: int = 8192,
        # inference speed
        pred_k: Optional[int] = None,
        ann_backend: ANNBackend = "auto",
        ann_params: Optional[Dict[str, Any]] = None,
        n_jobs: int = -1,
        # accept unicode kwargs (β, ε, κ_eps, σ2, pred_κ, ...)
        **kwargs: Any,
    ):
        # ---- map unicode kwargs -> ascii ----
        if "β" in kwargs:
            beta = kwargs.pop("β")
        if "ε" in kwargs:
            eps = kwargs.pop("ε")
        if "κ_eps" in kwargs:
            k_eps = kwargs.pop("κ_eps")
        if "ε_use_kth" in kwargs:
            eps_use_kth = kwargs.pop("ε_use_kth")
        if "ε_mul" in kwargs:
            eps_mul = kwargs.pop("ε_mul")
        if "σ2" in kwargs:
            sigma2 = kwargs.pop("σ2")
        if "pred_κ" in kwargs:
            pred_k = kwargs.pop("pred_κ")

        if kwargs:
            raise TypeError(f"Unexpected kwargs: {sorted(kwargs.keys())}")

        # ---- store params (provide both spellings) ----
        self.beta = float(beta)
        self.β = self.beta

        self.sigma2 = float(sigma2)
        self.σ2 = self.sigma2

        self.jitter = float(jitter)
        self.seed = int(seed)
        self.dtype = dtype
        self.fit_block = int(fit_block)

        # ---- validate / cast ----
        R_ix = np.ascontiguousarray(np.asarray(R_ix).astype(self.dtype, copy=False))
        R_iX = np.ascontiguousarray(np.asarray(R_iX).astype(self.dtype, copy=False))
        if R_ix.ndim != 2 or R_iX.ndim != 2 or R_ix.shape[0] != R_iX.shape[0]:
            raise ValueError("R_ix must be (N,d) and R_iX must be (N,D) with same N.")

        self.R_ix = R_ix
        self.R_iX = R_iX
        self.N, self.d_lat = R_ix.shape
        _, self.D = R_iX.shape

        # ---- center output ----
        self.center_X = bool(center_X)
        if self.center_X:
            self.mean_X = R_iX.mean(axis=0).astype(np.float64)
            Y = (R_iX.astype(np.float64) - self.mean_X[None, :])
        else:
            self.mean_X = np.zeros((self.D,), dtype=np.float64)
            Y = R_iX.astype(np.float64)

        # ---- latent whitening (optional) ----
        self.whiten_latent = bool(whiten_latent)
        Ztrain = R_ix.astype(np.float64)
        if self.whiten_latent:
            self.lat_mean_x = Ztrain.mean(axis=0)
            self.lat_std_x = np.maximum(Ztrain.std(axis=0), 1e-12)
            Ztrain_w = (Ztrain - self.lat_mean_x) / self.lat_std_x
        else:
            self.lat_mean_x = np.zeros((self.d_lat,), dtype=np.float64)
            self.lat_std_x = np.ones((self.d_lat,), dtype=np.float64)
            Ztrain_w = Ztrain

        self.R_ix_w = Ztrain_w  # (N,d) float64

        # ---- ANN on training latents (for eps + medoids snapping) ----
        self.ann_train, self.ann_backend = make_ann(ann_backend, ann_params=ann_params, n_jobs=n_jobs)
        self.ann_train.build(self.R_ix_w.astype(self.dtype, copy=False))

        # ---- eps via kNN distances on latents ----
        if eps is None:
            k_eps = int(min(max(8, int(k_eps)), self.N - 1))
            # ask for k_eps+1 to try to include self
            j_iK1, D2_iK1 = self.ann_train.search(self.R_ix_w.astype(self.dtype, copy=False), k_eps + 1)

            i = np.arange(self.N)[:, None]
            is_self = (j_iK1 == i)

            if np.any(is_self):
                D2_iK = np.empty((self.N, k_eps), dtype=np.float64)
                for ii in range(self.N):
                    keep = (j_iK1[ii] != ii)
                    D2_iK[ii] = D2_iK1[ii][keep][:k_eps]
            else:
                D2_iK = D2_iK1[:, :k_eps].astype(np.float64, copy=False)

            eps_hat = median_eps_from_knn_d2(D2_iK, use_kth=bool(eps_use_kth))
        else:
            eps_hat = float(eps)

        eps_hat *= float(eps_mul)
        if eps_hat <= 0:
            raise ValueError("eps must be > 0.")
        self.eps = float(eps_hat)
        self.ε = self.eps

        # ---- choose inducing points (in whitened latent space) ----
        rng = np.random.default_rng(self.seed)
        m = int(min(max(1, int(m)), self.N))

        if Z_mx is not None:
            Zm = np.asarray(Z_mx, dtype=np.float64)
            if Zm.ndim != 2 or Zm.shape[1] != self.d_lat:
                raise ValueError("Z_mx must be (m, d_lat).")
            Zm_w = (Zm - self.lat_mean_x) / self.lat_std_x
        else:
            if inducing == "random_subset":
                idx = rng.choice(self.N, size=m, replace=False)
                Zm_w = self.R_ix_w[idx]
            elif inducing == "fps":
                idx = fps_indices(self.R_ix_w, m=m, seed=self.seed)
                Zm_w = self.R_ix_w[idx]
            elif inducing == "kmeans_medoids":
                C = _kmeans2_safe(self.R_ix_w, m, seed=self.seed).astype(np.float64, copy=False)
                j_cm, _ = self.ann_train.search(C.astype(self.dtype, copy=False), 1)
                idx = j_cm.reshape(-1).astype(np.int64)

                # de-duplicate and refill if needed
                idx_u = np.unique(idx)
                if idx_u.size < m:
                    needed = m - idx_u.size
                    pool = np.setdiff1d(np.arange(self.N), idx_u, assume_unique=False)
                    if pool.size >= needed:
                        extra = rng.choice(pool, size=needed, replace=False)
                    else:
                        extra = rng.choice(self.N, size=needed, replace=True)
                    idx = np.concatenate([idx_u, extra])
                else:
                    idx = idx_u[:m]

                Zm_w = self.R_ix_w[idx]
            elif inducing == "given":
                raise ValueError("Provide Z_mx when inducing='given'.")
            else:
                raise ValueError(f"Unknown inducing mode: {inducing!r}")

        self.Z_mx_w = np.ascontiguousarray(Zm_w.astype(np.float64, copy=False))
        self.m = int(self.Z_mx_w.shape[0])

        # also store raw inducing points (unwhitened) for convenience
        self.Z_mx = (self.Z_mx_w * self.lat_std_x[None, :]) + self.lat_mean_x[None, :]

        # ---- ANN on inducing points for fast prediction ----
        self.ann_Z, _ = make_ann(ann_backend, ann_params=ann_params, n_jobs=n_jobs)
        self.ann_Z.build(self.Z_mx_w.astype(self.dtype, copy=False))

        # pred_k
        if pred_k is None:
            self.pred_k = None
        else:
            self.pred_k = int(min(max(1, int(pred_k)), self.m))
        self.pred_κ = self.pred_k  # unicode alias

        # ---- W_mm ----
        D2_mm = sqdist_ab(self.Z_mx_w, self.Z_mx_w)  # (m,m)
        W_mm = np.exp(-self.beta * (D2_mm.astype(np.float64) / self.eps))
        W_mm.flat[:: self.m + 1] += self.jitter
        self.W_mm = W_mm  # (m,m)

        # ---- accumulate G=C^T C and B=C^T Y ----
        G_mm = np.zeros((self.m, self.m), dtype=np.float64)
        B_mX = np.zeros((self.m, self.D), dtype=np.float64)

        bs = int(self.fit_block)
        for i0 in range(0, self.N, bs):
            i1 = min(self.N, i0 + bs)
            Zi = self.R_ix_w[i0:i1]  # (b,d) float64
            D2_im = sqdist_ab(Zi, self.Z_mx_w)  # (b,m)
            C_im = np.exp(-self.beta * (D2_im.astype(np.float64) / self.eps))
            G_mm += C_im.T @ C_im
            B_mX += C_im.T @ Y[i0:i1]

        A_mm = G_mm + self.sigma2 * W_mm
        A_mm.flat[:: self.m + 1] += self.jitter

        # Store factorization for mean *and* uncertainty
        self._A_mm = A_mm
        self._A_cF = la.cho_factor(A_mm, lower=True, check_finite=False)

        self.M_mX = la.cho_solve(self._A_cF, B_mX, check_finite=False)  # (m,D)

    # ------------------------------------------------------------
    # Convenience alternate constructor
    # ------------------------------------------------------------
    @classmethod
    def fit(cls, R_ix: np.ndarray, R_iX: np.ndarray, **kwargs: Any) -> "GPLM":
        """Alias constructor (kept for convenience)."""
        return cls(R_ix, R_iX, **kwargs)

    # ------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------
    def __call__(self, R_ax: Union[np.ndarray, list], *, batch_size: Optional[int] = None) -> np.ndarray:
        """
        Mean prediction only. For uncertainty, use predict(..., return_var=True).
        """
        R_ax = np.asarray(R_ax)
        single = (R_ax.ndim == 1)
        if single:
            R_ax = R_ax[None, :]
        R_ax = np.ascontiguousarray(R_ax.astype(self.dtype, copy=False))

        if batch_size is None:
            Y = self._decode_mean(R_ax)
        else:
            bs = int(batch_size)
            out = []
            for s in range(0, R_ax.shape[0], bs):
                out.append(self._decode_mean(R_ax[s : s + bs]))
            Y = np.vstack(out)

        return Y[0] if single else Y

    def predict(
        self,
        R_ax: Union[np.ndarray, list],
        *,
        return_var: bool = False,
        batch_size: Optional[int] = None,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Predict mean and (optional) scalar variance proxy per query.

        Returns:
          mean: (A,D)
          var_scalar: (A,)  approximately sigma2 * C A^{-1} C^T
        """
        R_ax = np.asarray(R_ax)
        single = (R_ax.ndim == 1)
        if single:
            R_ax = R_ax[None, :]
        R_ax = np.ascontiguousarray(R_ax.astype(self.dtype, copy=False))

        if not return_var:
            mean = self.__call__(R_ax, batch_size=batch_size)
            return mean[None, :] if single and mean.ndim == 1 else mean

        if batch_size is None:
            mean, var = self._decode_mean_var(R_ax)
        else:
            bs = int(batch_size)
            ms = []
            vs = []
            for s in range(0, R_ax.shape[0], bs):
                m, v = self._decode_mean_var(R_ax[s : s + bs])
                ms.append(m)
                vs.append(v)
            mean = np.vstack(ms)
            var = np.concatenate(vs, axis=0)

        if single:
            return mean[0], var[0]
        return mean, var

    def _decode_mean(self, R_ax: np.ndarray) -> np.ndarray:
        """Mean decoder: (A,d)->(A,D)"""
        Za = R_ax.astype(np.float64)
        Za_w = (Za - self.lat_mean_x) / self.lat_std_x  # (A,d)

        if self.pred_k is None or self.pred_k == self.m:
            D2_am = sqdist_ab(Za_w, self.Z_mx_w)  # (A,m)
            C_am = np.exp(-self.beta * (D2_am.astype(np.float64) / self.eps))
            Yc = C_am @ self.M_mX  # centered outputs (A,D)
        else:
            j_aK, D2_aK = self.ann_Z.search(Za_w.astype(self.dtype, copy=False), self.pred_k)
            W = np.exp(-self.beta * (D2_aK.astype(np.float64) / self.eps))  # (A,k)
            M = self.M_mX[j_aK]  # (A,k,D)
            Yc = np.sum(W[:, :, None] * M, axis=1)  # (A,D)

        return Yc + self.mean_X[None, :]

    def _decode_mean_var(self, R_ax: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Mean + scalar uncertainty proxy.
        var_scalar[a] = sigma2 * C(a)^T A^{-1} C(a)
        """
        Za = R_ax.astype(np.float64)
        Za_w = (Za - self.lat_mean_x) / self.lat_std_x  # (A,d)

        # Use full inducing for uncertainty (simple + stable)
        D2_am = sqdist_ab(Za_w, self.Z_mx_w)  # (A,m)
        C_am = np.exp(-self.beta * (D2_am.astype(np.float64) / self.eps))  # (A,m)

        # mean
        Yc = C_am @ self.M_mX  # (A,D)
        mean = Yc + self.mean_X[None, :]

        # scalar variance proxy: sigma2 * diag(C A^{-1} C^T)
        # Compute each row's quadratic form via cho_solve
        # tmp = A^{-1} C^T -> shape (m,A)
        tmp = la.cho_solve(self._A_cF, C_am.T, check_finite=False)  # (m,A)
        quad = np.sum(C_am.T * tmp, axis=0)  # (A,)
        var = self.sigma2 * quad  # scalar variance proxy
        return mean, var

    # ------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------
    def kernel_mass(self, R_ax: Union[np.ndarray, list]) -> np.ndarray:
        """
        S(r) = sum_m exp(-beta ||r - z_m||^2 / eps)
        Useful as a quick "support" indicator:
          - large mass: near inducing cloud
          - tiny mass: off-support (decoder tends to mean)
        """
        R_ax = np.asarray(R_ax, dtype=np.float64)
        single = (R_ax.ndim == 1)
        if single:
            R_ax = R_ax[None, :]
        Za_w = (R_ax - self.lat_mean_x[None, :]) / self.lat_std_x[None, :]
        D2 = sqdist_ab(Za_w, self.Z_mx_w)
        C = np.exp(-self.beta * (D2 / self.eps))
        mass = np.sum(C, axis=1)
        return mass[0] if single else mass

    # ============================================================
    # Geodesic flow on pullback manifold (no C_mm storage)
    # ============================================================
    #
    # Key math:
    #   Decoder: X = f(r)
    #   Metric:  g(r) = J(r)^T J(r)
    #
    # This integrator uses the same "generalized leapfrog" structure you had,
    # but:
    #   - "eps" is renamed to dt
    #   - a robust metric solver is used (Cholesky fallback to eig)
    #   - optional top-k tangent projection is supported
    #   - optional external force can be added to momentum updates
    #
    # External force API:
    #   force_fn(r, v) -> f   shape (d,)
    # Interpreted as "dp/dt = f" in coordinates (same object you add to momentum).
    #
    # ============================================================

    def _rbf_cache_single(
        self, r_x: np.ndarray, *, idx_m: Optional[np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Cache kernel terms at position r (single point):

          rw = (r - mean)/std
          Dw = rw - Zw  = (r - z)/std
          Dw_over = Dw / std = (r - z)/std^2   (chain-rule for unwhitened coordinates)

          k_m:     (k,)
          Dw_over: (k,d)
          grad_k:  (k,d) where grad_k[m,x] = d/d r_x k_m

        If idx_m is None -> use all inducing points (k=m). Otherwise use subset (k=pred_k).
        """
        c = self.beta / self.eps
        inv_std = 1.0 / self.lat_std_x  # (d,)

        r = r_x.astype(np.float64, copy=False)
        rw = (r - self.lat_mean_x) / self.lat_std_x  # (d,)

        Zw = self.Z_mx_w if idx_m is None else self.Z_mx_w[idx_m]  # (k,d)
        Dw = rw[None, :] - Zw  # (k,d) = (r-z)/std
        D2 = np.sum(Dw * Dw, axis=1)  # (k,)
        k_m = np.exp(-c * D2)  # (k,)

        Dw_over = Dw * inv_std[None, :]  # (k,d) = (r-z)/std^2

        # grad_k[m,x] = d/d r_x exp(-c ||Dw||^2) = -(2c) k_m * (Dw_x/std_x)
        grad_k = -(2.0 * c) * (k_m[:, None] * Dw_over)  # (k,d)
        return k_m, Dw_over, grad_k

    def _metric_from_gradM(
        self,
        grad_k: np.ndarray,  # (k,d)
        M_kX: np.ndarray,  # (k,D)
        *,
        D_block: int = 8192,
        lam: float = 1e-8,
        metric_solver: MetricSolver = "auto",
        eig_clip: float = 1e-12,
    ) -> _MetricFact:
        """
        Compute pullback metric g = J^T J without forming C_mm:

          J_{Xx} = sum_m grad_k[m,x] M_{mX}
          g_{xy} = sum_X J_{Xx} J_{Xy}

        Implemented by blocking over ambient dimension D:

          Jb = M_block^T @ grad_k    (block,d)
          g += Jb^T @ Jb

        Returns a robust solver wrapper for g.
        """
        k, d = grad_k.shape
        D = M_kX.shape[1]
        g = np.zeros((d, d), dtype=np.float64)

        for j0 in range(0, D, int(D_block)):
            j1 = min(D, j0 + int(D_block))
            Mb = M_kX[:, j0:j1]  # (k,block)
            Jb = Mb.T @ grad_k  # (block,d)
            g += Jb.T @ Jb  # (d,d)

        # sym + diagonal floor
        g = 0.5 * (g + g.T)
        g.flat[:: d + 1] += float(lam)

        # Robust factorization
        if metric_solver == "eigh":
            w, V = np.linalg.eigh(g)
            idx = np.argsort(w)[::-1]
            w = w[idx]
            V = V[:, idx]
            return _MetricFact(method="eigh", g=g, eigvals=w, eigvecs=V, eig_clip=float(eig_clip))

        if metric_solver in ("chol", "auto"):
            try:
                cF = la.cho_factor(g, lower=True, check_finite=False)
                return _MetricFact(method="chol", g=g, chol=cF, eig_clip=float(eig_clip))
            except la.LinAlgError:
                # fallback to eigen
                w, V = np.linalg.eigh(g)
                idx = np.argsort(w)[::-1]
                w = w[idx]
                V = V[:, idx]
                return _MetricFact(method="eigh", g=g, eigvals=w, eigvecs=V, eig_clip=float(eig_clip))

        # final fallback
        w, V = np.linalg.eigh(g)
        idx = np.argsort(w)[::-1]
        w = w[idx]
        V = V[:, idx]
        return _MetricFact(method="eigh", g=g, eigvals=w, eigvecs=V, eig_clip=float(eig_clip))

    def _force_from_cache_noC(
        self,
        k_m: np.ndarray,  # (k,)
        Dw_over: np.ndarray,  # (k,d) = (r-z)/std^2
        v_x: np.ndarray,  # (d,)
        M_kX: np.ndarray,  # (k,D)
        *,
        D_block: int = 8192,
    ) -> np.ndarray:
        """
        Geodesic momentum force (in your notation f_x = 0.5 v^y v^z ∂_x g_yz)
        computed without C_mm using ambient contraction:

          Jv_X      = sum_m S_m M_{mX}
          Hv_{xX}   = sum_m T_{xm} M_{mX}
          f_x       = sum_X Hv_{xX} Jv_X

        where for RBF kernel with whitened latent (chain rule included):
          s_m = sum_y Dw_over[m,y] v_y
          S_m = -(2c) k_m s_m
          T_{xm} = k_m [ 4c^2 Dw_over[m,x] s_m - 2c v_x / std_x^2 ]
          c = beta/eps
        """
        c = self.beta / self.eps
        v = v_x.astype(np.float64, copy=False)

        inv_std2 = (1.0 / self.lat_std_x) ** 2  # (d,)

        # s_m = sum_y Dw_over[m,y] v_y
        s_m = Dw_over @ v  # (k,)
        S = -(2.0 * c) * (k_m * s_m)  # (k,)

        # T as (d,k)
        term1 = (4.0 * c * c) * (k_m * s_m)[:, None] * Dw_over  # (k,d)
        term2 = (2.0 * c) * k_m[:, None] * (v[None, :] * inv_std2[None, :])  # (k,d)
        T = (term1 - term2).T  # (d,k)

        d = v.shape[0]
        D = M_kX.shape[1]
        f = np.zeros((d,), dtype=np.float64)

        for j0 in range(0, D, int(D_block)):
            j1 = min(D, j0 + int(D_block))
            Mb = M_kX[:, j0:j1]  # (k,block)
            Jv_b = S @ Mb  # (block,)
            Hv_b = T @ Mb  # (d,block)
            f += Hv_b @ Jv_b  # (d,)

        return f

    def flow(
        self,
        R_ax: Union[np.ndarray, list],
        v_ax: Union[np.ndarray, list],
        *,
        dt: float = 1e-2,
        # fixed-point iterations for symplectic-ish generalized leapfrog
        K_p: int = 5,
        K_q: int = 5,
        # geometry compute
        D_block: int = 8192,
        lam: float = 1e-8,
        metric_solver: MetricSolver = "auto",
        eig_clip: float = 1e-12,
        # tangent projection (optional)
        k_tangent: Optional[int] = None,
        # external force on momentum: dp/dt += force_fn(r, v)
        force_fn: Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        One generalized-leapfrog step for *geodesic flow* on the pullback manifold,
        with optional tangent projection and external force.

        Inputs:
          R_ax: (A,d) or (d,)   latent positions (unwhitened coordinates)
          v_ax: (A,d) or (d,)   latent velocities (unwhitened coordinates)

        Outputs:
          (R_next, v_next) with same shapes as inputs.

        Notes:
          - If k_tangent is set, we project velocities onto top-k eigendirections
            of the metric g(r) at each evaluation point.
          - If force_fn is provided, it is added to momentum update as dp/dt.
          - Robust metric solver avoids Cholesky failures when g is PSD/singular.
        """
        R = np.asarray(R_ax, dtype=np.float64)
        v = np.asarray(v_ax, dtype=np.float64)

        single = (R.ndim == 1)
        if single:
            R = R[None, :]
            v = v[None, :]
        if R.ndim != 2 or v.ndim != 2 or R.shape != v.shape or R.shape[1] != self.d_lat:
            raise ValueError(f"Expected R_ax and v_ax shape (A,{self.d_lat}) (or ({self.d_lat},)).")

        A, d = R.shape
        dt = float(dt)

        # Inducing subset indices per point (optional)
        if self.pred_k is None or self.pred_k == self.m:
            idx_aK = None
        else:
            Rw = (R - self.lat_mean_x[None, :]) / self.lat_std_x[None, :]
            idx_aK, _ = self.ann_Z.search(Rw.astype(self.dtype, copy=False), self.pred_k)
            idx_aK = idx_aK.astype(np.int64, copy=False)

        R_next = np.empty_like(R)
        v_next = np.empty_like(v)

        for a in range(A):
            idx = None if idx_aK is None else idx_aK[a]
            M_kX = self.M_mX if idx is None else self.M_mX[idx]  # (k,D)

            r_n = R[a]
            v_n = v[a]

            # ---- geometry at r_n ----
            k_m_n, Dw_over_n, grad_k_n = self._rbf_cache_single(r_n, idx_m=idx)
            mf_n = self._metric_from_gradM(
                grad_k_n, M_kX, D_block=D_block, lam=lam,
                metric_solver=metric_solver, eig_clip=eig_clip
            )

            # Optional tangent projection at start
            if k_tangent is not None:
                v_n = mf_n.project_topk(v_n, int(k_tangent))

            # momentum p_n = g(r_n) v_n
            p_n = mf_n.g @ v_n

            # ---- (1) implicit half-step in momentum (fixed point) ----
            p = p_n.copy()
            for _ in range(int(K_p)):
                v_k = mf_n.solve(p)  # v = g_n^{-1} p
                if k_tangent is not None:
                    v_k = mf_n.project_topk(v_k, int(k_tangent))

                f_geo = self._force_from_cache_noC(k_m_n, Dw_over_n, v_k, M_kX, D_block=D_block)

                f_ext = 0.0
                if force_fn is not None:
                    f_ext = np.asarray(force_fn(r_n, v_k), dtype=np.float64)
                    if f_ext.shape != (d,):
                        raise ValueError("force_fn must return shape (d,)")

                p = p_n + 0.5 * dt * (f_geo + f_ext)

            p_half = p

            # ---- (2) implicit position update (fixed point) ----
            v_half_n = mf_n.solve(p_half)
            if k_tangent is not None:
                v_half_n = mf_n.project_topk(v_half_n, int(k_tangent))

            r = r_n + dt * v_half_n  # init guess

            for _ in range(int(K_q)):
                k_m_r, Dw_over_r, grad_k_r = self._rbf_cache_single(r, idx_m=idx)
                mf_r = self._metric_from_gradM(
                    grad_k_r, M_kX, D_block=D_block, lam=lam,
                    metric_solver=metric_solver, eig_clip=eig_clip
                )
                v_half_r = mf_r.solve(p_half)
                if k_tangent is not None:
                    v_half_r = mf_r.project_topk(v_half_r, int(k_tangent))
                r = r_n + 0.5 * dt * (v_half_n + v_half_r)

            r_np1 = r

            # ---- (3) explicit half-step in momentum at r_{n+1} ----
            k_m_np1, Dw_over_np1, grad_k_np1 = self._rbf_cache_single(r_np1, idx_m=idx)
            mf_np1 = self._metric_from_gradM(
                grad_k_np1, M_kX, D_block=D_block, lam=lam,
                metric_solver=metric_solver, eig_clip=eig_clip
            )

            v_mid = mf_np1.solve(p_half)
            if k_tangent is not None:
                v_mid = mf_np1.project_topk(v_mid, int(k_tangent))

            f_geo_np1 = self._force_from_cache_noC(k_m_np1, Dw_over_np1, v_mid, M_kX, D_block=D_block)

            f_ext_np1 = 0.0
            if force_fn is not None:
                f_ext_np1 = np.asarray(force_fn(r_np1, v_mid), dtype=np.float64)
                if f_ext_np1.shape != (d,):
                    raise ValueError("force_fn must return shape (d,)")

            p_np1 = p_half + 0.5 * dt * (f_geo_np1 + f_ext_np1)

            v_np1 = mf_np1.solve(p_np1)
            if k_tangent is not None:
                v_np1 = mf_np1.project_topk(v_np1, int(k_tangent))

            R_next[a] = r_np1
            v_next[a] = v_np1

        if single:
            return R_next[0], v_next[0]
        return R_next, v_next


__all__ = ["GPLM", "InducingMode"]
