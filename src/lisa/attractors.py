# attractors.py
"""
A small zoo of chaotic attractors with a standardized API.

All generators return a trajectory array of shape (T, D), where:
  - T is the number of time points / steps.
  - D is the state dimension (e.g., 3 for Lorenz-63).

For continuous-time systems (ODEs), dt is the integration step (RK4).
For discrete-time maps (Hénon, Ikeda), dt is accepted for API consistency but ignored.

You can later standardize/split/chop trajectories for train/test externally.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

import numpy as np


Array = np.ndarray


# -----------------------------
# generic RK4 integrator
# -----------------------------
def rk4(f: Callable[[Array], Array], x0: Array, T: int, dt: float) -> Array:
    """
    Integrate x' = f(x) using fixed-step RK4.

    Parameters
    ----------
    f  : function x -> dx/dt
    x0 : (D,)
    T  : number of time points (trajectory length)
    dt : step size

    Returns
    -------
    traj : (T, D)
    """
    x = np.asarray(x0, dtype=float).copy()
    D = x.size
    traj = np.zeros((int(T), D), dtype=float)
    for t in range(int(T)):
        traj[t] = x
        k1 = f(x)
        k2 = f(x + 0.5 * dt * k1)
        k3 = f(x + 0.5 * dt * k2)
        k4 = f(x + dt * k3)
        x = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return traj


def _rng_from_seed(seed: Optional[int]) -> np.random.Generator:
    return np.random.default_rng(None if seed is None else int(seed))


def _discard_burn_in(traj: Array, burn_in: int) -> Array:
    burn_in = int(burn_in)
    if burn_in <= 0:
        return traj
    if burn_in >= traj.shape[0]:
        raise ValueError(f"burn_in={burn_in} must be < T={traj.shape[0]}")
    return traj[burn_in:]


# -----------------------------
# 1) Lorenz-63 (3D ODE)
# -----------------------------
def lorenz63(
    T: int,
    dt: float = 0.01,
    *,
    sigma: float = 10.0,
    rho: float = 28.0,
    beta: float = 8.0 / 3.0,
    x0: Optional[Array] = None,
    seed: Optional[int] = 0,
    burn_in: int = 0,
    ) -> Array:
    """
    Lorenz-63 system (classic chaotic regime).
    Returns (T-burn_in, 3).
    """
    rng = _rng_from_seed(seed)
    if x0 is None:
        x0 = np.array([1.0, 1.0, 1.0], dtype=float) + 0.01 * rng.standard_normal(3)
    else:
        x0 = np.asarray(x0, dtype=float).reshape(3)

    def f(x: Array) -> Array:
        return np.array(
            [
                sigma * (x[1] - x[0]),
                x[0] * (rho - x[2]) - x[1],
                x[0] * x[1] - beta * x[2],
            ],
            dtype=float,
        )

    traj = rk4(f, x0, T=T, dt=float(dt))
    return _discard_burn_in(traj, burn_in)


# -----------------------------
# 2) Lorenz-96 (dD ODE)
# -----------------------------
def lorenz96(
    T: int,
    dt: float = 0.01,
    *,
    dim: int = 5,
    F: float = 8.0,
    x0: Optional[Array] = None,
    seed: Optional[int] = 0,
    burn_in: int = 0,
    ) -> Array:
    """
    Lorenz-96 in the standard chaotic regime (F ~ 8, dim ~ 40).
    Returns (T-burn_in, dim).
    """
    rng = _rng_from_seed(seed)
    d = int(dim)
    if x0 is None:
        # common init: near-constant with small noise
        x0 = F * np.ones(d, dtype=float)
        x0 += 0.01 * rng.standard_normal(d)
        # add a localized bump
        x0[0] += 0.1
    else:
        x0 = np.asarray(x0, dtype=float).reshape(d)

    def f(x: Array) -> Array:
        # cyclic indices: x_{i+1}, x_{i-1}, x_{i-2}
        xp1 = np.roll(x, -1)
        xm1 = np.roll(x, 1)
        xm2 = np.roll(x, 2)
        return (xp1 - xm2) * xm1 - x + F

    traj = rk4(f, x0, T=T, dt=float(dt))
    return _discard_burn_in(traj, burn_in)


# -----------------------------
# 3) Rössler (3D ODE)
# -----------------------------
def rossler(
    T: int,
    dt: float = 0.02,
    *,
    a: float = 0.2,
    b: float = 0.2,
    c: float = 5.7,
    x0: Optional[Array] = None,
    seed: Optional[int] = 0,
    burn_in: int = 0,
    ) -> Array:
    """
    Rössler attractor (classic chaotic parameters).
    Returns (T-burn_in, 3).
    """
    rng = _rng_from_seed(seed)
    if x0 is None:
        x0 = np.array([0.1, 0.0, 0.0], dtype=float) + 0.01 * rng.standard_normal(3)
    else:
        x0 = np.asarray(x0, dtype=float).reshape(3)

    def f(x: Array) -> Array:
        x1, y1, z1 = x
        return np.array(
            [
                -y1 - z1,
                x1 + a * y1,
                b + z1 * (x1 - c),
            ],
            dtype=float,
        )

    traj = rk4(f, x0, T=T, dt=float(dt))
    return _discard_burn_in(traj, burn_in)

# -----------------------------
# 4) Duffing oscillator (forced, 2D ODE)
# -----------------------------
def duffing(
    T: int,
    dt: float = 0.01,
    *,
    delta: float = 0.2,
    gamma: float = 0.3,
    omega: float = 1.2,
    # equation: x'' + delta x' - x + x^3 = gamma cos(omega t)
    x0: Optional[Array] = None,   # (x, v)
    seed: Optional[int] = 0,
    burn_in: int = 0,
    ) -> Array:
    """
    Forced Duffing oscillator in a commonly chaotic regime.
    State is (x, v) where v = x'.
    Returns (T-burn_in, 2).

    Note: This is already nonautonomous due to cos(omega t).
    """
    rng = _rng_from_seed(seed)
    if x0 is None:
        x = np.array([0.1, 0.0], dtype=float) + 0.01 * rng.standard_normal(2)
    else:
        x = np.asarray(x0, dtype=float).reshape(2)

    traj = np.zeros((int(T), 2), dtype=float)

    # Nonautonomous RK4 (depends on time)
    t = 0.0
    for n in range(int(T)):
        traj[n] = x

        def f(tcur: float, s: Array) -> Array:
            xx, vv = s
            # x' = v
            # v' = x - x^3 - delta v + gamma cos(omega t)
            return np.array(
                [vv, xx - xx**3 - delta * vv + gamma * np.cos(omega * tcur)],
                dtype=float,
            )

        k1 = f(t, x)
        k2 = f(t + 0.5 * dt, x + 0.5 * dt * k1)
        k3 = f(t + 0.5 * dt, x + 0.5 * dt * k2)
        k4 = f(t + dt, x + dt * k3)
        x = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        t += dt

    return _discard_burn_in(traj, burn_in)

def generate(name: str, T: int, dt: float = 0.01, **kwargs) -> Array:
    """
    Convenience wrapper:
      traj = generate("lorenz63", T=20000, dt=0.01, burn_in=2000)
    """
    key = name.lower()
    if key not in ATTRACTORS:
        raise KeyError(f"Unknown attractor '{name}'. Options: {sorted(ATTRACTORS.keys())}")
    return ATTRACTORS[key](T=T, dt=dt, **kwargs)

def _chua_nonlinearity(x, m0, m1):
    # h(x) = m1*x + 0.5*(m0-m1)(|x+1|-|x-1|)
    return m1 * x + 0.5 * (m0 - m1) * (np.abs(x + 1.0) - np.abs(x - 1.0))

def chua(
    T: int,
    *,
    dt: float = 0.01,
    burn_in: int = 2000,
    alpha: float = 15.6,
    beta: float = 28.0,
    m0: float = -1.15,
    m1: float = -0.70,
    seed: int = 0,
    x0: np.ndarray | None = None,
    ):
    """
    Chua double-scroll attractor.

    Equations:
      xdot = alpha*(y - x - h(x))
      ydot = x - y + z
      zdot = -beta*y
    """
    rng = np.random.default_rng(seed)

    if x0 is None:
        x = 0.1 * rng.standard_normal(3)
    else:
        x = np.array(x0, dtype=np.float64).reshape(3)

    def f(x):
        xx, yy, zz = x
        h = _chua_nonlinearity(xx, m0=m0, m1=m1)
        dx = alpha * (yy - xx - h)
        dy = xx - yy + zz
        dz = -beta * yy
        return np.array([dx, dy, dz], dtype=np.float64)

    # RK4
    T = int(T_total)
    out = np.zeros((T, 3), dtype=np.float64)
    for n in range(T):
        out[n] = x
        k1 = f(x)
        k2 = f(x + 0.5 * dt * k1)
        k3 = f(x + 0.5 * dt * k2)
        k4 = f(x + dt * k3)
        x = x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

    if burn_in > 0:
        out = out[burn_in:]
    return out

def torus_quasiperiodic(
    T: int,
    *,
    dt: float = 0.01,
    burn_in: int = 0,
    R: float = 2.0,
    r: float = 0.7,
    omega1: float = 1.0,
    omega2: float = np.sqrt(2.0),  # irrational ratio -> quasi-periodic
    seed: int = 0,
    noise: float = 0.0,            # set small (e.g. 1e-3) for "almost but not really"
    ):
    """
    Quasiperiodic orbit on a torus embedded in R^3.

    Returns:
      (T_total - burn_in, 3)
    """
    rng = np.random.default_rng(seed)
    T = int(T_total)
    t = np.arange(T, dtype=np.float64) * float(dt)

    th1 = omega1 * t
    th2 = omega2 * t

    x = (R + r * np.cos(th2)) * np.cos(th1)
    y = (R + r * np.cos(th2)) * np.sin(th1)
    z = r * np.sin(th2)

    X = np.stack([x, y, z], axis=1)

    if noise > 0:
        X = X + noise * rng.standard_normal(size=X.shape)

    if burn_in > 0:
        X = X[burn_in:]
    return X


# -----------------------------
# Dadras (Dadras–Momeni) attractor (3D autonomous ODE)
# -----------------------------
def dadras(
    T: int,
    dt: float = 0.01,
    *,
    a: float = 3.0,
    b: float = 2.7,
    c: float = 1.7,
    d: float = 2.0,
    e: float = 9.0,
    x0: Optional[Array] = None,   # (x, y, z)
    seed: Optional[int] = 0,
    burn_in: int = 0,
    ) -> Array:
    """
    Dadras (Dadras–Momeni) chaotic attractor.

    Equations (common form):
      x' = y - a*x + b*y*z
      y' = c + z*(1 - x)
      z' = d*x*y - e*z

    Returns: (T - burn_in, 3) as [x, y, z].

    Typical chaotic params:
      a=3, b=2.7, c=1.7, d=2, e=9

    Notes:
      - Autonomous & stationary on attractor after burn-in.
      - RK4 integrator (stable for dt ~ 1e-3..1e-2 typically).
    """
    rng = _rng_from_seed(seed)

    if x0 is None:
        x = np.array([1.0, 1.0, 1.0], dtype=float) + 0.01 * rng.standard_normal(3)
    else:
        x = np.asarray(x0, dtype=float).reshape(3)

    traj = np.zeros((int(T), 3), dtype=float)

    t = 0.0
    for n in range(int(T)):
        traj[n] = x

        def f(_t: float, s: Array) -> Array:
            xx, yy, zz = s
            dx = yy - a * xx + b * yy * zz
            dy = c + zz * (1.0 - xx)
            dz = d * xx * yy - e * zz
            return np.array([dx, dy, dz], dtype=float)

        # RK4
        k1 = f(t, x)
        k2 = f(t + 0.5 * dt, x + 0.5 * dt * k1)
        k3 = f(t + 0.5 * dt, x + 0.5 * dt * k2)
        k4 = f(t + dt, x + dt * k3)
        x = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        t += dt

    return _discard_burn_in(traj, burn_in)

# -----------------------------
# Halvorsen attractor (3D autonomous ODE)
# -----------------------------
def halvorsen(
    T: int,
    dt: float = 0.005,
    *,
    alpha: float = 1.4,
    x0: Optional[Array] = None,   # (x, y, z)
    seed: Optional[int] = 0,
    burn_in: int = 0,
    ) -> Array:
    """
    Halvorsen attractor (cyclically symmetric chaotic system).

    Equations:
      x' = -α x - 4y - 4z - y^2
      y' = -α y - 4z - 4x - z^2
      z' = -α z - 4x - 4y - x^2

    Typical chaotic parameter:
      α = 1.4

    Returns: (T - burn_in, 3) as [x, y, z].

    Notes:
      - dt=0.005 is a common stable choice for this system.
      - RK4 integrator.
    """
    rng = _rng_from_seed(seed)

    if x0 is None:
        # classic-ish small start
        x = np.array([0.1, 0.0, 0.0], dtype=float) + 0.01 * rng.standard_normal(3)
    else:
        x = np.asarray(x0, dtype=float).reshape(3)

    traj = np.zeros((int(T), 3), dtype=float)

    t = 0.0
    a = float(alpha)
    h = float(dt)

    def f(_t: float, s: Array) -> Array:
        xx, yy, zz = s
        dx = -a * xx - 4.0 * yy - 4.0 * zz - (yy * yy)
        dy = -a * yy - 4.0 * zz - 4.0 * xx - (zz * zz)
        dz = -a * zz - 4.0 * xx - 4.0 * yy - (xx * xx)
        return np.array([dx, dy, dz], dtype=float)

    for n in range(int(T)):
        traj[n] = x

        # RK4
        k1 = f(t, x)
        k2 = f(t + 0.5 * h, x + 0.5 * h * k1)
        k3 = f(t + 0.5 * h, x + 0.5 * h * k2)
        k4 = f(t + h, x + h * k3)
        x = x + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        t += h

    return _discard_burn_in(traj, burn_in)

def _rng_from_seed(seed: Optional[int]) -> np.random.Generator:
    if seed is None:
        return np.random.default_rng()
    return np.random.default_rng(int(seed))

def _discard_burn_in(traj: Array, burn_in: int) -> Array:
    b = int(max(0, burn_in))
    return traj[b:] if b > 0 else traj

# -----------------------------
# Duffing oscillator (forced)
# -----------------------------
def duffing_NESS(
    T: int,
    dt: float = 0.01,
    *,
    # equation: x'' + delta x' - x + x^3 = gamma cos(omega t)
    delta: float = 0.2,
    gamma: float = 0.30,
    omega: float = 1.0,
    # IC
    x0: Optional[Array] = None,   # (x, v) OR (x, v, theta)
    seed: Optional[int] = 0,
    burn_in: int = 0,
    # extra outputs
    include_phase: bool = True,   # return (x,v,cosθ,sinθ) if True else (x,v)
) -> Array:
    """
    Forced Duffing oscillator in a commonly chaotic regime.

    Base 2D state: (x, v) where v = x'
    If include_phase=True, we *augment* the observed state with forcing phase:
      theta' = omega
      return (x, v, cos(theta), sin(theta))  -> 4D autonomous observation
    This tends to improve forecastability vs a purely 2D nonautonomous observation.

    References for typical forced Duffing form and chaos behavior:
    - Duffing equation definition and chaos discussion :contentReference[oaicite:1]{index=1}
    - Example parameter sets (alpha=-1, beta=1, delta~0.2, gamma~0.3, omega~1) :contentReference[oaicite:2]{index=2}
    """
    rng = _rng_from_seed(seed)

    # init
    if x0 is None:
        x = np.array([0.1, 0.0], dtype=float) + 0.01 * rng.standard_normal(2)
        theta = 0.0
    else:
        x0 = np.asarray(x0, dtype=float).reshape(-1)
        if x0.size >= 2:
            x = x0[:2].copy()
        else:
            raise ValueError("x0 must have at least (x,v).")
        theta = float(x0[2]) if x0.size >= 3 else 0.0

    if include_phase:
        traj = np.zeros((int(T), 4), dtype=float)
    else:
        traj = np.zeros((int(T), 2), dtype=float)

    # RK4 integration
    t = 0.0
    for n in range(int(T)):
        if include_phase:
            traj[n, 0] = x[0]
            traj[n, 1] = x[1]
            traj[n, 2] = np.cos(theta)
            traj[n, 3] = np.sin(theta)
        else:
            traj[n] = x

        def f(tcur: float, s: Array, th: float) -> Array:
            xx, vv = s
            # x' = v
            # v' = x - x^3 - delta v + gamma cos(omega t)
            return np.array([vv, xx - xx**3 - delta * vv + gamma * np.cos(th)], dtype=float)

        # RK4 for (x,v) with theta advanced explicitly
        k1 = f(t, x, theta)
        k2 = f(t + 0.5 * dt, x + 0.5 * dt * k1, theta + 0.5 * dt * omega)
        k3 = f(t + 0.5 * dt, x + 0.5 * dt * k2, theta + 0.5 * dt * omega)
        k4 = f(t + dt, x + dt * k3, theta + dt * omega)

        x = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        theta = theta + dt * omega
        t += dt

    return _discard_burn_in(traj, burn_in)



# -----------------------------
# registry / convenience
# -----------------------------
ATTRACTORS: Dict[str, Callable[..., Array]] = {
    "lorenz63": lorenz63,
    "lorenz96": lorenz96,
    "rossler": rossler,
    "halvorsen": halvorsen,
    "chua": chua,
    "dadras": dadras,
    "torus_quasiperiodic": torus_quasiperiodic,
    "duffing": duffing,
    "duffing_NESS": duffing_NESS
}
