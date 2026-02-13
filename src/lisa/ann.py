# src/ann.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

from .utils import as_contig_f32, sqdist_ab


# -------------------------
# Public API
# -------------------------
ANNBackend = str  # "auto" | "faiss" | "pynndescent" | "sklearn" | "brute"


class ANNBase:
    """Minimal ANN interface used by DMAP/GPLM."""
    def build(self, X: np.ndarray) -> "ANNBase":
        raise NotImplementedError

    def search(self, Q: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns:
          idx: (B,k) int64
          D2 : (B,k) float32  (squared Euclidean distances)
        """
        raise NotImplementedError


def make_ann(
    backend: ANNBackend = "auto",
    ann_params: Optional[Dict[str, Any]] = None,
    n_jobs: int = -1,
) -> Tuple[ANNBase, str]:
    """
    Create an ANN implementation.

    backend:
      - "auto": prefers faiss, then pynndescent, then sklearn, else brute
      - "faiss": FAISS (if installed)
      - "pynndescent": NNDescent (if installed)
      - "sklearn": sklearn NearestNeighbors (if installed)
      - "brute": exact brute force

    ann_params:
      - for faiss:
          index: "flat" | "hnsw" | "ivf_flat"
          hnsw_M: int (default 32)
          ef_search: int (default 64)
          ef_construction: int (default 200)
          ivf_nlist: int (default 1024)
          ivf_nprobe: int (default 16)
          use_float16: bool (default False; GPU only typically)
      - for pynndescent:
          n_trees: int
          n_iters: int
          metric: str (default "euclidean")
      - for sklearn:
          algorithm: str (default "auto")
          leaf_size: int (default 40)
          metric: str (default "euclidean")
    """
    ann_params = {} if ann_params is None else dict(ann_params)
    b = (backend or "auto").lower()

    if b == "auto":
        for cand in ("faiss", "pynndescent", "sklearn", "brute"):
            ann, used = make_ann(cand, ann_params=ann_params, n_jobs=n_jobs)
            if used != "brute" or cand == "brute":
                return ann, used
        return BruteANN(), "brute"

    if b == "faiss":
        try:
            return FaissANN(ann_params=ann_params), "faiss"
        except Exception as e:
            raise ImportError(
                "FAISS backend requested but faiss is not available or failed to initialize. "
                "Install with: pip install dima[faiss]"
            ) from e

    if b == "pynndescent":
        try:
            return PyNNDescentANN(ann_params=ann_params), "pynndescent"
        except Exception as e:
            raise ImportError(
                "pynndescent backend requested but pynndescent is not available. "
                "Install with: pip install pynndescent"
            ) from e

    if b == "sklearn":
        try:
            return SklearnANN(n_jobs=n_jobs, ann_params=ann_params), "sklearn"
        except Exception as e:
            raise ImportError(
                "sklearn backend requested but scikit-learn is not available. "
                "Install with: pip install scikit-learn"
            ) from e

    if b == "brute":
        return BruteANN(), "brute"

    raise ValueError(f"Unknown ANN backend: {backend!r}")


# -------------------------
# Brute-force (no deps)
# -------------------------
class BruteANN(ANNBase):
    def __init__(self):
        self.X = None

    def build(self, X: np.ndarray) -> "BruteANN":
        self.X = as_contig_f32(X)
        return self

    def search(self, Q: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        if self.X is None:
            raise RuntimeError("BruteANN.search called before build().")
        X = self.X
        Q = as_contig_f32(Q)
        k = int(k)
        if k <= 0:
            raise ValueError("k must be >= 1")
        if k > X.shape[0]:
            k = X.shape[0]

        D2 = sqdist_ab(Q, X)  # (B,N)
        idx = np.argpartition(D2, kth=k - 1, axis=1)[:, :k]
        rows = np.arange(Q.shape[0])[:, None]
        d2 = D2[rows, idx]

        # sort within k
        ordk = np.argsort(d2, axis=1)
        idx = idx[rows, ordk].astype(np.int64)
        d2 = d2[rows, ordk].astype(np.float32)
        return idx, d2


# -------------------------
# FAISS
# -------------------------
class FaissANN(ANNBase):
    def __init__(self, ann_params: Optional[Dict[str, Any]] = None):
        self.ann_params = {} if ann_params is None else dict(ann_params)
        self.index = None
        self.X = None  # keep reference for possible rebuild

        # delayed import
        import faiss  # type: ignore
        self.faiss = faiss

    def _build_index(self, d: int):
        p = self.ann_params
        faiss = self.faiss

        index_kind = str(p.get("index", "flat")).lower()

        if index_kind == "flat":
            index = faiss.IndexFlatL2(d)

        elif index_kind == "hnsw":
            M = int(p.get("hnsw_M", 32))
            index = faiss.IndexHNSWFlat(d, M)
            # optional tuning
            ef_search = int(p.get("ef_search", 64))
            ef_constr = int(p.get("ef_construction", 200))
            index.hnsw.efSearch = ef_search
            index.hnsw.efConstruction = ef_constr

        elif index_kind == "ivf_flat":
            nlist = int(p.get("ivf_nlist", 1024))
            quantizer = faiss.IndexFlatL2(d)
            index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)
            nprobe = int(p.get("ivf_nprobe", 16))
            index.nprobe = nprobe

        else:
            raise ValueError(f"Unknown faiss index kind: {index_kind!r}")

        return index

    def build(self, X: np.ndarray) -> "FaissANN":
        X = as_contig_f32(X)
        self.X = X
        faiss = self.faiss
        d = int(X.shape[1])

        index = self._build_index(d)

        # IVF needs training
        if hasattr(index, "is_trained") and not index.is_trained:
            index.train(X)

        index.add(X)
        self.index = index
        return self

    def search(self, Q: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        if self.index is None:
            raise RuntimeError("FaissANN.search called before build().")
        Q = as_contig_f32(Q)
        k = int(k)
        if k <= 0:
            raise ValueError("k must be >= 1")

        # FAISS returns (distances, indices); for L2 these are squared distances
        D2, I = self.index.search(Q, k)
        return I.astype(np.int64), D2.astype(np.float32)


# -------------------------
# PyNNDescent
# -------------------------
class PyNNDescentANN(ANNBase):
    def __init__(self, ann_params: Optional[Dict[str, Any]] = None):
        self.ann_params = {} if ann_params is None else dict(ann_params)
        self.index = None
        self.X = None

        from pynndescent import NNDescent  # type: ignore
        self.NNDescent = NNDescent

    def build(self, X: np.ndarray) -> "PyNNDescentANN":
        X = as_contig_f32(X)
        self.X = X
        p = self.ann_params

        metric = p.get("metric", "euclidean")
        n_trees = p.get("n_trees", None)
        n_iters = p.get("n_iters", None)

        kwargs: Dict[str, Any] = {"metric": metric}
        if n_trees is not None:
            kwargs["n_trees"] = int(n_trees)
        if n_iters is not None:
            kwargs["n_iters"] = int(n_iters)

        self.index = self.NNDescent(X, **kwargs)
        return self

    def search(self, Q: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        if self.index is None:
            raise RuntimeError("PyNNDescentANN.search called before build().")
        Q = as_contig_f32(Q)
        k = int(k)
        if k <= 0:
            raise ValueError("k must be >= 1")

        # NNDescent returns (indices, distances) with euclidean distances (not squared)
        I, d = self.index.query(Q, k=k)
        D2 = (d.astype(np.float32) ** 2)
        return I.astype(np.int64), D2


# -------------------------
# scikit-learn NearestNeighbors
# -------------------------
class SklearnANN(ANNBase):
    def __init__(self, n_jobs: int = -1, ann_params: Optional[Dict[str, Any]] = None):
        self.ann_params = {} if ann_params is None else dict(ann_params)
        self.n_jobs = int(n_jobs)
        self.nn = None
        self.X = None

        from sklearn.neighbors import NearestNeighbors  # type: ignore
        self.NearestNeighbors = NearestNeighbors

    def build(self, X: np.ndarray) -> "SklearnANN":
        X = as_contig_f32(X)
        self.X = X
        p = self.ann_params

        algorithm = p.get("algorithm", "auto")
        leaf_size = int(p.get("leaf_size", 40))
        metric = p.get("metric", "euclidean")

        self.nn = self.NearestNeighbors(
            n_neighbors=1,  # set later in search
            algorithm=algorithm,
            leaf_size=leaf_size,
            metric=metric,
            n_jobs=self.n_jobs,
        )
        self.nn.fit(X)
        return self

    def search(self, Q: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        if self.nn is None:
            raise RuntimeError("SklearnANN.search called before build().")
        Q = as_contig_f32(Q)
        k = int(k)
        if k <= 0:
            raise ValueError("k must be >= 1")

        self.nn.set_params(n_neighbors=k)
        d, I = self.nn.kneighbors(Q, return_distance=True)
        # sklearn distances are euclidean; square them
        D2 = (d.astype(np.float32) ** 2)
        return I.astype(np.int64), D2
