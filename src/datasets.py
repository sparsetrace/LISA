# datasets.py
from __future__ import annotations

import os
import sys
import time
import urllib.request
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Tuple, Union

import numpy as np

try:
    import pandas as pd
except Exception as e:  # pragma: no cover
    raise ImportError("datasets.py requires pandas. Install with: pip install pandas") from e


# -----------------------------
# Config
# -----------------------------
ELECTRICITY_URL = "https://data.openei.org/files/8562/historic_load_hourly_2016_2023_county.h5"
ELECTRICITY_H5 = "historic_load_hourly_2016_2023_county.h5"


# -----------------------------
# Helpers
# -----------------------------
def download_with_progress(url: str, out_path: str, chunk_size: int = 1024 * 1024) -> None:
    """
    Download a large file with a simple text progress bar (no extra deps).
    Skips if file already exists.
    """
    if os.path.exists(out_path):
        return

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    print(f"[datasets] Downloading:\n  {url}\n  -> {out_path}")

    with urllib.request.urlopen(url) as response, open(out_path, "wb") as out_file:
        total = getattr(response, "length", None)
        downloaded = 0
        start_time = time.time()

        while True:
            chunk = response.read(chunk_size)
            if not chunk:
                break
            out_file.write(chunk)
            downloaded += len(chunk)

            if total:
                frac = downloaded / total
                pct = frac * 100
                bar_len = 30
                filled = int(bar_len * frac)
                bar = "#" * filled + "-" * (bar_len - filled)
                rate = downloaded / (time.time() - start_time + 1e-9)
                sys.stdout.write(
                    f"\r[{bar}] {pct:6.2f}%  "
                    f"{downloaded / 1e6:8.1f} / {total / 1e6:8.1f} MB  "
                    f"{rate / 1e6:4.1f} MB/s"
                )
                sys.stdout.flush()

    if total:
        print("\n[datasets] Download complete.")
    else:
        print("[datasets] Download complete.")


def _safe_cache_name(
    *,
    start: Optional[str],
    stop: Optional[str],
    resample: Optional[str],
    agg: str,
    n_features: Optional[int],
    feature_method: str,
    seed: int,
    downcast_float32: bool,
) -> str:
    def clean(x: Optional[str]) -> str:
        if x is None:
            return "none"
        return str(x).replace(":", "-").replace("/", "-").replace(" ", "_")

    return (
        "electricity"
        f"_start-{clean(start)}"
        f"_stop-{clean(stop)}"
        f"_res-{clean(resample)}"
        f"_agg-{agg}"
        f"_nf-{n_features if n_features is not None else 'all'}"
        f"_fsel-{feature_method}"
        f"_seed-{int(seed)}"
        f"_f32-{int(bool(downcast_float32))}"
        ".npz"
    )


def _require_tables_hint(err: Exception) -> None:
    msg = (
        "\n[datasets] Failed to read .h5. This usually means the optional dependency `tables` is missing.\n"
        "Install once to build a cache:\n"
        "  pip install tables\n"
        "or, if using extras:\n"
        "  pip install .[electricity]\n"
    )
    raise ImportError(msg) from err


def _coerce_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True)
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    return df.sort_index()


def _fill_missing(df: pd.DataFrame) -> pd.DataFrame:
    # Conservative: forward-fill then back-fill; if still NaN, fill 0.
    df = df.ffill().bfill()
    if df.isna().values.any():
        df = df.fillna(0.0)
    return df


def _select_features(
    df: pd.DataFrame,
    *,
    features: Optional[Sequence[str]] = None,
    n_features: Optional[int] = None,
    method: str = "variance",
    seed: int = 0,
) -> pd.DataFrame:
    """
    Feature selection on columns.
    - features: explicit list of column labels
    - n_features: choose subset
      method: "variance" (top variance) or "random"
    """
    if features is not None:
        missing = [c for c in features if c not in df.columns]
        if missing:
            raise KeyError(f"Requested features not found in columns: {missing[:10]}{'...' if len(missing)>10 else ''}")
        return df.loc[:, list(features)]

    if n_features is None:
        return df

    n_features = int(n_features)
    if n_features <= 0:
        raise ValueError("n_features must be positive")

    if n_features >= df.shape[1]:
        return df

    method = method.lower()
    if method == "random":
        rng = np.random.default_rng(int(seed))
        cols = list(df.columns)
        idx = rng.choice(len(cols), size=n_features, replace=False)
        sel = [cols[i] for i in idx]
        return df.loc[:, sel]

    if method == "variance":
        # top variance across time (after missing fill)
        v = df.var(axis=0, ddof=0)
        sel = list(v.sort_values(ascending=False).index[:n_features])
        return df.loc[:, sel]

    raise ValueError("feature_method must be one of: {'variance','random'}")


# -----------------------------
# Main API: Electricity
# -----------------------------
@dataclass(frozen=True)
class ElectricityData:
    R_tX: np.ndarray
    t_index: np.ndarray          # int64 UTC ns
    X_index: np.ndarray          # dtype=object strings


def load_electricity(
    *,
    data_dir: str = "data",
    url: str = ELECTRICITY_URL,
    h5_name: str = ELECTRICITY_H5,
    # slicing
    start: Optional[str] = None,      # e.g. "2019-01-01"
    stop: Optional[str] = None,       # e.g. "2020-01-01"
    # resampling
    resample: Optional[str] = None,   # e.g. "D" for daily, "W" weekly; None keeps hourly
    agg: str = "sum",                 # "sum" or "mean"
    # feature downselect
    features: Optional[Sequence[str]] = None,
    n_features: Optional[int] = None,
    feature_method: str = "variance", # "variance" or "random"
    seed: int = 0,
    # dtype / missing
    downcast_float32: bool = True,
    # caching
    cache: bool = True,
    cache_dir: Optional[str] = None,
    force_rebuild_cache: bool = False,
) -> ElectricityData:
    """
    Returns a manageable (T, X) array from the OpenEI historic county load dataset.

    Behavior:
      - If a matching .npz cache exists -> loads it (NO `tables` required).
      - Else downloads .h5 and reads via pandas (requires `tables` installed once),
        then applies slicing/resampling/feature-selection, and optionally writes cache.

    Notes:
      - index is returned as int64 UTC nanoseconds (portable, no pandas dependency later).
      - columns are returned as strings (county IDs like 'p36041').
    """
    data_dir = str(data_dir)
    os.makedirs(data_dir, exist_ok=True)

    if cache_dir is None:
        cache_dir = os.path.join(data_dir, "cache")
    os.makedirs(cache_dir, exist_ok=True)

    agg = agg.lower()
    if agg not in ("sum", "mean"):
        raise ValueError("agg must be 'sum' or 'mean'")

    cache_name = _safe_cache_name(
        start=start, stop=stop, resample=resample, agg=agg,
        n_features=n_features, feature_method=feature_method,
        seed=seed, downcast_float32=downcast_float32,
    )
    cache_path = os.path.join(cache_dir, cache_name)

    # 1) Load cache if present
    if cache and (not force_rebuild_cache) and os.path.exists(cache_path):
        z = np.load(cache_path, allow_pickle=True)
        R_tX = z["R_tX"]
        t_index = z["t_index"]
        X_index = z["X_index"]
        return ElectricityData(R_tX=R_tX, t_index=t_index, X_index=X_index)

    # 2) Ensure H5 present
    h5_path = os.path.join(data_dir, h5_name)
    download_with_progress(url, h5_path)

    # 3) Read H5 (requires tables)
    try:
        df = pd.read_hdf(h5_path)
    except Exception as e:
        _require_tables_hint(e)

    # 4) Basic preprocess
    df = _coerce_datetime_index(df)
    df = _fill_missing(df)

    # 5) Time slice
    if start is not None or stop is not None:
        df = df.loc[start:stop]

    # 6) Resample
    if resample is not None:
        if agg == "sum":
            df = df.resample(resample).sum()
        else:
            df = df.resample(resample).mean()

    # 7) Feature selection
    df = _select_features(df, features=features, n_features=n_features, method=feature_method, seed=seed)

    # 8) Downcast
    if downcast_float32:
        df = df.astype("float32", copy=False)

    # 9) Build outputs
    R_tX = df.to_numpy()
    t_index = df.index.view("int64")          # UTC ns
    X_index = np.asarray(df.columns.astype(str), dtype=object)

    # 10) Cache
    if cache:
        np.savez_compressed(cache_path, R_tX=R_tX, t_index=t_index, X_index=X_index)
        print(f"[datasets] Wrote cache: {cache_path}")

    return ElectricityData(R_tX=R_tX, t_index=t_index, X_index=X_index)
