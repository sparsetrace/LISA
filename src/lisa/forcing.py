# forcing.py
"""
Forced / nonstationary variants of chaotic attractors.

Standard output is always a single long trajectory:
  R_tX : (T-burn_in, D)

For ODEs, forcing MUST be applied inside the integrator (nonautonomous RK4),
not post-hoc. This file provides three forcing mechanisms for Lorenz-63:

  1) Additive forcing to the dynamics (adds kappa*u(t) to dx/dt, dy/dt, dz/dt)
  2) Parameter drift (e.g., rho(t) varies with time / forcing)
  3) Regime switching (piecewise-constant params over time)

All functions accept (T, dt) as primary inputs, with reasonable defaults.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Sequence, Tuple, Union

import numpy as np

Array = np.ndarray


# -----------------------------
# small utils
# -----------------------------
def _rng_from_seed(seed: Optional[int]) -> np.random.Generator:
    return np.random.default_rng(None if seed is None else int(seed))


def _discard_burn_in(traj: Array, burn_in: int) -> Array:
    burn_in = int(burn_in)
    if burn_in <= 0:
        return traj
    if burn_in >= traj.shape[0]:
        raise ValueError(f"burn_in={burn_in} must be < T={traj.shape[0]}")
    return traj[burn_in:]


def _as_1d(u: Union[None, float, Array]) -> Optional[Array]:
    if u is None:
        return None
    if np.isscalar(u):
        return np.asarray([float(u)], dtype=float)
    u = np.asarray(u, dtype=float)
    if u.ndim != 1:
        raise ValueError(f"Expected 1D forcing array, got shape {u.shape}")
    return u


def _build_time_grid(T: int, dt: float) -> Array:
    return np.arange(int(T), dtype=float) * float(dt)


def _interp_u(u_t: Array, t_grid: Array, tcur: float) -> float:
    """
    Forcing evaluated at arbitrary time using linear interpolation.
    Uses endpoint values outside the grid.
    """
    if u_t.size == 1:
        return float(u_t[0])
    return float(np.interp(tcur, t_grid, u_t, left=u_t[0], right=u_t[-1]))


def _kappa_vec(kappa: Union[float, Sequence[float], Array], axis: Optional[int] = None) -> Array:
    """
    Convert kappa specification into a length-3 vector for Lorenz63.
    - If axis is provided, interpret kappa as scalar magnitude applied to that axis.
    - Else if kappa is scalar -> applied to x-axis by default.
    - Else if kappa has length 3 -> used directly.
    """
    if axis is not None:
        kv = np.zeros(3, dtype=float)
        kv[int(axis)] = float(kappa)  # type: ignore[arg-type]
        return kv

    if np.isscalar(kappa):
        # default: force x-equation
        return np.array([float(kappa), 0.0, 0.0], dtype=float)

    kv = np.asarray(kappa, dtype=float).reshape(-1)
    if kv.size != 3:
        raise ValueError("kappa must be scalar, length-3, or scalar+axis")
    return kv


# -----------------------------
# forcing signal helpers (optional)
# -----------------------------
def forcing_sine(T: int, dt: float, *, amp: float = 1.0, freq_hz: float = 0.1, phase: float = 0.0, bias: float = 0.0) -> Array:
    """
    u(t) = bias + amp * sin(2*pi*freq_hz*t + phase)
    Returns u_t of length T aligned with sample times n*dt.
    """
    t = _build_time_grid(T, dt)
    return (bias + amp * np.sin(2.0 * np.pi * float(freq_hz) * t + float(phase))).astype(float)


def forcing_piecewise_constant(T: int, *, values: Sequence[float], lengths: Sequence[int]) -> Array:
    """
    Build a piecewise-constant forcing u_t of length T.
    Example: values=[0,1,0.5], lengths=[2000,3000,5000]
    """
    vals = list(values)
    lens = list(map(int, lengths))
    if len(vals) != len(lens):
        raise ValueError("values and lengths must have same length")
    u = np.concatenate([np.full(L, v, dtype=float) for v, L in zip(vals, lens)], axis=0)
    if u.size < T:
        u = np.pad(u, (0, T - u.size), mode="edge")
    return u[:T]


# -----------------------------
# core integrator: nonautonomous RK4 for Lorenz63
# -----------------------------
def _lorenz63_step_rk4(
    x: Array,
    t: float,
    dt: float,
    *,
    sigma_fn: Callable[[float], float],
    rho_fn: Callable[[float], float],
    beta_fn: Callable[[float], float],
    force_fn: Callable[[float], Array],   # returns (3,) additive term to dx/dt
) -> Array:
    """
    One RK4 step for x' = f(t,x) with Lorenz63 + forcing.
    """
    x = np.asarray(x, dtype=float)

    def f(tcur: float, s: Array) -> Array:
        sigma = float(sigma_fn(tcur))
        rho = float(rho_fn(tcur))
        beta = float(beta_fn(tcur))

        fx = np.empty(3, dtype=float)
        fx[0] = sigma * (s[1] - s[0])
        fx[1] = s[0] * (rho - s[2]) - s[1]
        fx[2] = s[0] * s[1] - beta * s[2]
        fx += force_fn(tcur)
        return fx

    k1 = f(t, x)
    k2 = f(t + 0.5 * dt, x + 0.5 * dt * k1)
    k3 = f(t + 0.5 * dt, x + 0.5 * dt * k2)
    k4 = f(t + dt, x + dt * k3)
    return x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def _simulate_lorenz63_nonauto(
    T: int,
    dt: float,
    *,
    x0: Array,
    sigma_fn: Callable[[float], float],
    rho_fn: Callable[[float], float],
    beta_fn: Callable[[float], float],
    force_fn: Callable[[float], Array],
) -> Array:
    T = int(T)
    dt = float(dt)
    x = np.asarray(x0, dtype=float).reshape(3).copy()
    traj = np.zeros((T, 3), dtype=float)
    t = 0.0
    for n in range(T):
        traj[n] = x
        x = _lorenz63_step_rk4(
            x, t, dt,
            sigma_fn=sigma_fn,
            rho_fn=rho_fn,
            beta_fn=beta_fn,
            force_fn=force_fn,
        )
        t += dt
    return traj


# -----------------------------
# 1) Additive forcing: x' = f(x) + kappa * u(t)
# -----------------------------
def lorenz63_forced_additive(
    T: int,
    dt: float = 0.01,
    *,
    sigma: float = 10.0,
    rho: float = 28.0,
    beta: float = 8.0 / 3.0,
    u_t: Optional[Array] = None,
    u_fn: Optional[Callable[[float], float]] = None,
    kappa: Union[float, Sequence[float], Array] = 1.0,
    axis: Optional[int] = 0,
    x0: Optional[Array] = None,
    seed: Optional[int] = 0,
    burn_in: int = 0,
    return_u: bool = False,
) -> Union[Array, Tuple[Array, Array]]:
    """
    Additive forcing applied to dx/dt,dy/dt,dz/dt.

    You may provide either:
      - u_t: array of length T (sampled at n*dt), or
      - u_fn: callable u(t)

    kappa:
      - scalar + axis (default) => applies to one coordinate derivative
      - length-3 vector => applies to all derivatives
      - scalar with axis=None => defaults to x-axis

    Returns:
      traj : (T-burn_in, 3)
      and optionally u_used : (T,) if return_u=True
    """
    rng = _rng_from_seed(seed)
    if x0 is None:
        x0 = np.array([1.0, 1.0, 1.0], dtype=float) + 0.01 * rng.standard_normal(3)
    else:
        x0 = np.asarray(x0, dtype=float).reshape(3)

    kv = _kappa_vec(kappa, axis=axis)

    # build forcing evaluator
    if u_fn is None:
        ut = _as_1d(u_t)
        if ut is None:
            ut = np.zeros(int(T), dtype=float)
        # ensure length T
        if ut.size != int(T):
            if ut.size < int(T):
                ut = np.pad(ut, (0, int(T) - ut.size), mode="edge")
            ut = ut[: int(T)]
        t_grid = _build_time_grid(T, dt)

        def u_eval(tcur: float) -> float:
            return _interp_u(ut, t_grid, tcur)

        u_used = ut
    else:
        def u_eval(tcur: float) -> float:
            return float(u_fn(tcur))

        # produce a sampled u for logging/return
        t_grid = _build_time_grid(T, dt)
        u_used = np.array([u_eval(ti) for ti in t_grid], dtype=float)

    def sigma_fn(_t: float) -> float:
        return float(sigma)

    def rho_fn(_t: float) -> float:
        return float(rho)

    def beta_fn(_t: float) -> float:
        return float(beta)

    def force_fn(tcur: float) -> Array:
        return kv * u_eval(tcur)

    traj = _simulate_lorenz63_nonauto(
        T=T, dt=dt, x0=x0,
        sigma_fn=sigma_fn, rho_fn=rho_fn, beta_fn=beta_fn,
        force_fn=force_fn
    )
    traj = _discard_burn_in(traj, burn_in)
    u_used2 = _discard_burn_in(u_used[:, None], burn_in).reshape(-1)  # align
    return (traj, u_used2) if return_u else traj


# -----------------------------
# 2) Parameter drift: rho(t) varies (optionally driven by u(t))
# -----------------------------
def lorenz63_param_drift(
    T: int,
    dt: float = 0.01,
    *,
    sigma: float = 10.0,
    rho0: float = 28.0,
    beta: float = 8.0 / 3.0,
    # specify rho(t) either directly or via forcing
    rho_t: Optional[Array] = None,                   # length T at n*dt
    rho_fn: Optional[Callable[[float], float]] = None,
    u_t: Optional[Array] = None,                     # used only if rho_t/rho_fn not provided
    u_fn: Optional[Callable[[float], float]] = None,
    kappa_rho: float = 1.0,                          # rho(t) = rho0 + kappa_rho * u(t)
    # optional additive forcing simultaneously (set to 0 to disable)
    kappa_force: Union[float, Sequence[float], Array] = 0.0,
    axis_force: Optional[int] = 0,
    x0: Optional[Array] = None,
    seed: Optional[int] = 0,
    burn_in: int = 0,
    return_rho: bool = False,
) -> Union[Array, Tuple[Array, Array]]:
    """
    Nonstationary Lorenz63 where rho(t) drifts.

    Priority for defining rho(t):
      1) rho_fn(t)
      2) rho_t sampled at n*dt
      3) rho0 + kappa_rho * u(t)  (u from u_fn or u_t; defaults to 0)

    You can also add *simultaneous additive forcing* via kappa_force/axis_force.

    Returns:
      traj : (T-burn_in, 3)
      and optionally rho_used : (T-burn_in,) if return_rho=True
    """
    rng = _rng_from_seed(seed)
    if x0 is None:
        x0 = np.array([1.0, 1.0, 1.0], dtype=float) + 0.01 * rng.standard_normal(3)
    else:
        x0 = np.asarray(x0, dtype=float).reshape(3)

    # build u evaluator (only if needed)
    ut = _as_1d(u_t)
    if ut is None:
        ut = np.zeros(int(T), dtype=float)
    if ut.size != int(T):
        if ut.size < int(T):
            ut = np.pad(ut, (0, int(T) - ut.size), mode="edge")
        ut = ut[: int(T)]
    t_grid = _build_time_grid(T, dt)

    if u_fn is None:
        def u_eval(tcur: float) -> float:
            return _interp_u(ut, t_grid, tcur)
        u_used = ut
    else:
        def u_eval(tcur: float) -> float:
            return float(u_fn(tcur))
        u_used = np.array([u_eval(ti) for ti in t_grid], dtype=float)

    # build rho evaluator
    if rho_fn is not None:
        def rho_eval(tcur: float) -> float:
            return float(rho_fn(tcur))
        rho_used = np.array([rho_eval(ti) for ti in t_grid], dtype=float)
    elif rho_t is not None:
        rt = _as_1d(rho_t)
        if rt is None:
            raise ValueError("rho_t must be a 1D array if provided")
        if rt.size != int(T):
            if rt.size < int(T):
                rt = np.pad(rt, (0, int(T) - rt.size), mode="edge")
            rt = rt[: int(T)]

        def rho_eval(tcur: float) -> float:
            return float(np.interp(tcur, t_grid, rt, left=rt[0], right=rt[-1]))
        rho_used = rt
    else:
        def rho_eval(tcur: float) -> float:
            return float(rho0 + kappa_rho * u_eval(tcur))
        rho_used = np.array([rho_eval(ti) for ti in t_grid], dtype=float)

    kv_force = _kappa_vec(kappa_force, axis=axis_force)

    def sigma_fn(_t: float) -> float:
        return float(sigma)

    def rho_fn2(tcur: float) -> float:
        return float(rho_eval(tcur))

    def beta_fn(_t: float) -> float:
        return float(beta)

    def force_fn(tcur: float) -> Array:
        # optional additive forcing in dynamics
        return kv_force * u_eval(tcur) if np.any(kv_force != 0.0) else np.zeros(3, dtype=float)

    traj = _simulate_lorenz63_nonauto(
        T=T, dt=dt, x0=x0,
        sigma_fn=sigma_fn, rho_fn=rho_fn2, beta_fn=beta_fn,
        force_fn=force_fn
    )

    traj = _discard_burn_in(traj, burn_in)
    rho_used2 = _discard_burn_in(rho_used[:, None], burn_in).reshape(-1)
    return (traj, rho_used2) if return_rho else traj


# -----------------------------
# 3) Regime switching: piecewise-constant params (sigma,rho,beta)
# -----------------------------
@dataclass(frozen=True)
class Lorenz63Params:
    sigma: float = 10.0
    rho: float = 28.0
    beta: float = 8.0 / 3.0


def lorenz63_regime_switch(
    T: int,
    dt: float = 0.01,
    *,
    # a schedule of regimes: (start_step, params)
    schedule: Sequence[Tuple[int, Lorenz63Params]] = ((0, Lorenz63Params()),),
    # optional additive forcing simultaneously
    u_t: Optional[Array] = None,
    u_fn: Optional[Callable[[float], float]] = None,
    kappa: Union[float, Sequence[float], Array] = 0.0,
    axis: Optional[int] = 0,
    x0: Optional[Array] = None,
    seed: Optional[int] = 0,
    burn_in: int = 0,
    return_params: bool = False,
) -> Union[Array, Tuple[Array, Dict[str, Array]]]:
    """
    Piecewise-constant parameter switching. `schedule` is a list of
    (start_step, Lorenz63Params). Example:

      schedule = [
        (0,    Lorenz63Params(rho=28.0)),
        (8000, Lorenz63Params(rho=35.0)),
        (14000,Lorenz63Params(rho=25.0)),
      ]

    Regime selection at time t uses idx = floor(t/dt), then picks the last
    schedule entry with start_step <= idx.

    Returns:
      traj : (T-burn_in, 3)
      and optionally a dict with sigma_t, rho_t, beta_t arrays (aligned).
    """
    rng = _rng_from_seed(seed)
    if x0 is None:
        x0 = np.array([1.0, 1.0, 1.0], dtype=float) + 0.01 * rng.standard_normal(3)
    else:
        x0 = np.asarray(x0, dtype=float).reshape(3)

    # build regime arrays (per step)
    T = int(T)
    sigma_t = np.empty(T, dtype=float)
    rho_t = np.empty(T, dtype=float)
    beta_t = np.empty(T, dtype=float)

    sched = sorted([(int(s), p) for s, p in schedule], key=lambda z: z[0])
    if not sched or sched[0][0] != 0:
        raise ValueError("schedule must start at step 0")

    # fill by segments
    for i, (s0, p0) in enumerate(sched):
        s1 = sched[i + 1][0] if i + 1 < len(sched) else T
        s0 = max(0, min(T, s0))
        s1 = max(s0, min(T, s1))
        sigma_t[s0:s1] = float(p0.sigma)
        rho_t[s0:s1] = float(p0.rho)
        beta_t[s0:s1] = float(p0.beta)

    # forcing evaluator (optional)
    kv = _kappa_vec(kappa, axis=axis)

    if u_fn is None:
        ut = _as_1d(u_t)
        if ut is None:
            ut = np.zeros(T, dtype=float)
        if ut.size != T:
            if ut.size < T:
                ut = np.pad(ut, (0, T - ut.size), mode="edge")
            ut = ut[:T]
        t_grid = _build_time_grid(T, dt)

        def u_eval(tcur: float) -> float:
            return _interp_u(ut, t_grid, tcur)

        u_used = ut
    else:
        t_grid = _build_time_grid(T, dt)

        def u_eval(tcur: float) -> float:
            return float(u_fn(tcur))

        u_used = np.array([u_eval(ti) for ti in t_grid], dtype=float)

    # parameter evaluators: stepwise constant
    def _param_at(arr: Array, tcur: float) -> float:
        idx = int(tcur / float(dt))
        if idx < 0:
            idx = 0
        elif idx >= T:
            idx = T - 1
        return float(arr[idx])

    def sigma_fn(tcur: float) -> float:
        return _param_at(sigma_t, tcur)

    def rho_fn(tcur: float) -> float:
        return _param_at(rho_t, tcur)

    def beta_fn(tcur: float) -> float:
        return _param_at(beta_t, tcur)

    def force_fn(tcur: float) -> Array:
        if np.all(kv == 0.0):
            return np.zeros(3, dtype=float)
        return kv * u_eval(tcur)

    traj = _simulate_lorenz63_nonauto(
        T=T, dt=dt, x0=x0,
        sigma_fn=sigma_fn, rho_fn=rho_fn, beta_fn=beta_fn,
        force_fn=force_fn
    )

    traj = _discard_burn_in(traj, burn_in)
    if not return_params:
        return traj

    # align parameter arrays with burn_in
    sigma_used = _discard_burn_in(sigma_t[:, None], burn_in).reshape(-1)
    rho_used = _discard_burn_in(rho_t[:, None], burn_in).reshape(-1)
    beta_used = _discard_burn_in(beta_t[:, None], burn_in).reshape(-1)
    u_used2 = _discard_burn_in(u_used[:, None], burn_in).reshape(-1)

    info = {
        "sigma_t": sigma_used,
        "rho_t": rho_used,
        "beta_t": beta_used,
        "u_t": u_used2,
    }
    return traj, info
