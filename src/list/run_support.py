run_support.py
import numpy as np
import matplotlib.pyplot as plt

from metrics import compare_all_fast_with_curves

COLORS = {
    "ALSA": "C0",   # blue
    "LISA": "C3",   # red
}

METRICS = {
    "MSE":               ["mse", "MSE", "mse_mean"],
    "Spectral (JS/KL)":  ["spectral_kl", "spectral_js", "spectral_div", "spec_kl", "spectral"],
    "ACF-MSE":           ["acf_mse", "acf_l2", "acf_err", "acf"],
    "MMD$^2$ (RFF)":     ["mmd2_rff", "mmd_rff", "mmd2", "mmd"],
}

styles = {
    "Truth": dict(color="k",  lw=2.3, ls="-"),
    "NLSA":  dict(color="0.2", lw=2.0, ls="--"),
    "ALSA":  dict(color=COLORS["ALSA"], lw=2.0, ls="-.", alpha=0.95),
    "LISA":  dict(color=COLORS["LISA"], lw=2.1, ls=":", alpha=0.95),
}

def standardize_global(F_tX: np.ndarray, eps: float = 1e-12):
    mu = F_tX.mean(axis=0, keepdims=True)
    sd = F_tX.std(axis=0, keepdims=True) + eps
    return (F_tX - mu) / sd, mu, sd

def plot_3d_phase_multi(
    bg: np.ndarray,
    truth: np.ndarray,
    preds: dict[str, np.ndarray],
    *,
    title: str,
    elev: float = 20,
    azim: float = 35,
    bg_stride: int = 5,
    traj_stride: int = 1,
    max_bg_points: int = 20000,
    styles: dict[str, dict] | None = None,
    bg_style: dict | None = None,
    axis_names: tuple[str, str, str] = ("x", "y", "z"),
    ):
    """
    bg:    (Tbg, 3) background attractor (e.g. train)
    truth: (T, 3)
    preds: name -> (T, 3)

    styles: dict mapping curve name -> matplotlib kwargs
      e.g. styles["Truth"] = {...}, styles["NLSA"] = {...}
      keys should match preds keys.
    """

    bg = np.asarray(bg)
    truth = np.asarray(truth)
    if bg.ndim != 2 or truth.ndim != 2 or bg.shape[1] != 3 or truth.shape[1] != 3:
        raise ValueError(f"bg and truth must be (T,3). Got bg={bg.shape}, truth={truth.shape}")

    for k, v in preds.items():
        v = np.asarray(v)
        if v.ndim != 2 or v.shape[1] != 3:
            raise ValueError(f"pred '{k}' must be (T,3). Got {v.shape}")
        preds[k] = v

    if styles is None:
        styles = {}

    if bg_style is None:
        bg_style = dict(color="0.75", lw=0.8, alpha=0.35)

    # subsample background for speed
    if bg.shape[0] > max_bg_points:
        idx = np.linspace(0, bg.shape[0] - 1, max_bg_points).astype(int)
        bgp = bg[idx]
    else:
        bgp = bg
    bgp = bgp[::max(1, int(bg_stride))]

    t = truth[::max(1, int(traj_stride))]

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(bgp[:, 0], bgp[:, 1], bgp[:, 2], label="Background (train)", **bg_style)

    # Truth styling
    truth_style = dict(color="k", lw=2.2)
    truth_style.update(styles.get("Truth", {}))
    ax.plot(t[:, 0], t[:, 1], t[:, 2], label="Truth", **truth_style)

    # Predictions styling (fallback to cycle if not provided)
    cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["C0", "C1", "C2", "C3", "C4"])
    for i, (name, P) in enumerate(preds.items()):
        p = P[::max(1, int(traj_stride))]
        style = dict(color=cycle[i % len(cycle)], lw=1.8, alpha=0.95)
        style.update(styles.get(name, {}))
        ax.plot(p[:, 0], p[:, 1], p[:, 2], label=name, **style)

    # mark start point
    ax.scatter(t[0, 0], t[0, 1], t[0, 2], color=truth_style.get("color", "k"), s=60)

    ax.set_title(title)
    ax.set_xlabel(axis_names[0])
    ax.set_ylabel(axis_names[1])
    ax.set_zlabel(axis_names[2])
    ax.view_init(elev=elev, azim=azim)
    ax.legend(frameon=False, loc="upper left")
    plt.tight_layout()
    plt.show()
    return None

def make_task(F_test: np.ndarray, a_start: int, ell_ctx: int, steps: int):
    """
    Returns:
      prefix: (ell_ctx, D)
      truth : (steps, D)
    """
    assert a_start >= ell_ctx
    assert a_start + steps <= F_test.shape[0]
    prefix = F_test[a_start - ell_ctx: a_start, :]
    truth  = F_test[a_start: a_start + steps, :]
    return prefix, truth


def pick_metric_key(available_keys: list[str], candidates: list[str]) -> str:
    """Pick the first existing key from a list of candidate names."""
    for c in candidates:
        if c in available_keys:
            return c
    # fallback: substring match
    for c in candidates:
        for k in available_keys:
            if c.lower() in k.lower():
                return k
    raise KeyError(f"None of candidates {candidates} found in keys: {available_keys}")


def eval_multistart(
    model_name: str,
    predictor_fn,                # takes prefix -> (steps,D)
    F_test: np.ndarray,
    *,
    starts: np.ndarray,
    ell_ctx: int,
    steps: int,
    burn_in_metrics: int = 0,
    mmd_sample: int = 1024,
    seed: int = 0,
    dt=0.01,
    ):
    """
    Returns:
      scalars_mean: dict[str,float]
      scalars_std : dict[str,float]
      curves_mean : dict[str,np.ndarray]
    """
    scalar_keys = None
    scalars_acc = {}
    curves_acc = {"mse_by_horizon": [], "mse_per_feature": []}

    for i, a_start in enumerate(starts):
        prefix, truth = make_task(F_test, int(a_start), ell_ctx, steps)
        pred = predictor_fn(prefix)

        s, c = compare_all_fast_with_curves(
            truth, pred,
            burn_in=burn_in_metrics,
            acf_max_lag=min(200, steps - 2),
            mmd_sample=min(mmd_sample, steps),
            seed=seed + i,
            include_mmd_rff=True,
            dt=dt,
        )

        if scalar_keys is None:
            scalar_keys = list(s.keys())
            for k in scalar_keys:
                scalars_acc[k] = []

        for k in scalar_keys:
            scalars_acc[k].append(s[k])

        curves_acc["mse_by_horizon"].append(c["mse_by_horizon"])
        curves_acc["mse_per_feature"].append(c["mse_per_feature"])

    scalars_mean = {k: float(np.nanmean(v)) for k, v in scalars_acc.items()}
    scalars_std  = {k: float(np.nanstd(v))  for k, v in scalars_acc.items()}

    curves_mean = {
        "mse_by_horizon": np.nanmean(np.stack(curves_acc["mse_by_horizon"], axis=0), axis=0),
        "mse_per_feature": np.nanmean(np.stack(curves_acc["mse_per_feature"], axis=0), axis=0),
    }

    print(f"\n[{model_name}] ℓ={ell_ctx} mean±std over {len(starts)} starts:")
    for k in scalars_mean.keys():
        print(f"  {k:16s}: {scalars_mean[k]:.6g} ± {scalars_std[k]:.3g}")

    return scalars_mean, scalars_std, curves_mean
