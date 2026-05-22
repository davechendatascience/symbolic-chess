"""Spatial operators for PDE identification — extension of expression_layer/core.

The core expression_layer treats inputs as 1D time series. For PDE identification
we need spatial derivatives of fields u(x, t), so we extend with operators that
act along a spatial axis of a (T_time, N_space) field, then flatten back to a
1D vector that the existing PySR pipeline consumes.

Workflow
--------
  1. Generate or load a field u with shape (T, N_space). Each row is a time
     snapshot of the spatial state.
  2. Pre-compute spatial-derivative features via the operators below.
  3. Use `pde_feature_bank` to construct a flat dict of (T*N,) candidate
     features that PySR can search over.
  4. Run PySR / expression_layer on this bank; PySR doesn't see the 2D
     structure, just a long vector of (t, x) row-wise observations.

This module does NOT modify Variable or core operators — they remain 1D.
Spatial operators are pre-computations producing 1D Variables.

Boundary handling
-----------------
Central differences are used internally. At the spatial boundaries we use
one-sided (forward / backward) differences of the same order so the output
shape matches the input. NaN padding would propagate through PySR's eval.

References
----------
SINDy (Brunton, Proctor, Kutz 2016) — sparse regression on a candidate library
of derivative terms. PDE-FIND (Rudy et al 2017) — same idea applied to PDEs.
This module implements the candidate-library construction; PySR replaces the
sparse linear regressor with full symbolic search.
"""
from __future__ import annotations
import numpy as np


# ----------------- Spatial derivatives -----------------

def diff_x(u: np.ndarray, dx: float = 1.0, order: int = 1, axis: int = -1) -> np.ndarray:
    """Finite-difference partial derivative along a spatial axis.

    Central differences in the interior, one-sided at the boundaries (same
    order of accuracy as central where possible). Output shape matches input.

    Parameters
    ----------
    u : ndarray, e.g. (T, N_space) field
    dx : grid spacing
    order : derivative order (1 or 2 supported directly; higher via iteration)
    axis : spatial axis (default -1, the last)

    Returns
    -------
    du : same shape as u
    """
    if order == 1:
        return _diff_x_first(u, dx, axis)
    if order == 2:
        return _diff_x_second(u, dx, axis)
    # Higher orders by iteration
    result = u
    for _ in range(order):
        result = _diff_x_first(result, dx, axis)
    return result


def _diff_x_first(u: np.ndarray, dx: float, axis: int) -> np.ndarray:
    """First derivative; central interior + one-sided boundaries."""
    n = u.shape[axis]
    if n < 2:
        return np.zeros_like(u)
    out = np.empty_like(u, dtype=np.float64)
    # Build slicers for the chosen axis
    slc_all = [slice(None)] * u.ndim
    def s(i): a = list(slc_all); a[axis] = i; return tuple(a)
    # Interior: central difference (u[i+1] - u[i-1]) / (2 dx)
    interior = (np.take(u, np.arange(2, n), axis=axis) -
                np.take(u, np.arange(0, n-2), axis=axis)) / (2.0 * dx)
    out[s(slice(1, n-1))] = interior
    # Boundaries: forward / backward second-order
    out[s(0)] = (-3.0 * np.take(u, 0, axis=axis)
                  + 4.0 * np.take(u, 1, axis=axis)
                  - 1.0 * np.take(u, 2, axis=axis)) / (2.0 * dx)
    out[s(n-1)] = ( 3.0 * np.take(u, n-1, axis=axis)
                    - 4.0 * np.take(u, n-2, axis=axis)
                    + 1.0 * np.take(u, n-3, axis=axis)) / (2.0 * dx)
    return out


def _diff_x_second(u: np.ndarray, dx: float, axis: int) -> np.ndarray:
    """Second derivative; central interior + one-sided boundaries."""
    n = u.shape[axis]
    if n < 3:
        return np.zeros_like(u)
    out = np.empty_like(u, dtype=np.float64)
    slc_all = [slice(None)] * u.ndim
    def s(i): a = list(slc_all); a[axis] = i; return tuple(a)
    # Interior: (u[i+1] - 2 u[i] + u[i-1]) / dx^2
    interior = (np.take(u, np.arange(2, n), axis=axis)
                 - 2.0 * np.take(u, np.arange(1, n-1), axis=axis)
                 + np.take(u, np.arange(0, n-2), axis=axis)) / (dx * dx)
    out[s(slice(1, n-1))] = interior
    # Boundaries: one-sided second derivative
    out[s(0)] = (2.0 * np.take(u, 0, axis=axis)
                  - 5.0 * np.take(u, 1, axis=axis)
                  + 4.0 * np.take(u, 2, axis=axis)
                  - 1.0 * np.take(u, 3, axis=axis)) / (dx * dx) if n >= 4 else np.zeros_like(np.take(u, 0, axis=axis))
    out[s(n-1)] = (2.0 * np.take(u, n-1, axis=axis)
                    - 5.0 * np.take(u, n-2, axis=axis)
                    + 4.0 * np.take(u, n-3, axis=axis)
                    - 1.0 * np.take(u, n-4, axis=axis)) / (dx * dx) if n >= 4 else np.zeros_like(np.take(u, 0, axis=axis))
    return out


def diff_t(u: np.ndarray, dt: float = 1.0, axis: int = 0) -> np.ndarray:
    """Time derivative — same construction as spatial, different axis convention.

    Default axis=0 (time as the first dimension of u). Returns same shape.
    """
    return _diff_x_first(u, dt, axis)


def laplacian_1d(u: np.ndarray, dx: float = 1.0, axis: int = -1) -> np.ndarray:
    """1D Laplacian = second spatial derivative."""
    return diff_x(u, dx=dx, order=2, axis=axis)


def laplacian_2d(u: np.ndarray, dx: float = 1.0, dy: float = 1.0,
                   axes: tuple[int, int] = (-2, -1)) -> np.ndarray:
    """2D Laplacian = sum of second derivatives along two axes."""
    return (diff_x(u, dx=dx, order=2, axis=axes[0])
             + diff_x(u, dx=dy, order=2, axis=axes[1]))


# ----------------- Bank builder for PDE feature library -----------------

def pde_feature_bank_1d(u: np.ndarray, dx: float, *,
                           max_spatial_order: int = 4,
                           include_products: bool = True,
                           include_powers: tuple[int, ...] = (2,),
                           ) -> dict[str, np.ndarray]:
    """Construct a candidate feature library for 1D PDE identification.

    Returns a dict from feature name to a (T, N) array. Use `flatten_bank()`
    to convert to PySR-compatible 1D form.

    Includes:
      - u, u_x, u_xx, ..., u_{x...} up to `max_spatial_order`
      - Powers u^k for k in `include_powers`
      - Pairwise products u^k · u_{x...} if `include_products`

    Parameters
    ----------
    u : (T, N_space) field — time × space
    dx : grid spacing
    max_spatial_order : up to which derivative order to compute
    include_products : if True, include products u·u_x, u²·u_x, u·u_xx, etc.
    include_powers : tuple of integer powers of u to include (e.g. (2,) → u^2)

    Returns
    -------
    bank : dict[str, ndarray of shape (T, N)]
    """
    bank: dict[str, np.ndarray] = {"u": u.astype(np.float64)}
    derivs = {1: diff_x(u, dx=dx, order=1)}
    bank["u_x"] = derivs[1]
    if max_spatial_order >= 2:
        derivs[2] = diff_x(u, dx=dx, order=2)
        bank["u_xx"] = derivs[2]
    # Higher orders by iterated first derivative for stability
    for k in range(3, max_spatial_order + 1):
        derivs[k] = diff_x(derivs[k-1], dx=dx, order=1)
        bank[f"u_x{k}"] = derivs[k]
    # Powers of u
    for p in include_powers:
        bank[f"u_pow{p}"] = u.astype(np.float64) ** p
    # Pairwise products of (u, u^k) × (u_x, u_xx, ...)
    if include_products:
        for k in range(1, max_spatial_order + 1):
            d = derivs[k]
            bank[f"u_times_u_x{k}"] = u.astype(np.float64) * d
            for p in include_powers:
                bank[f"u_pow{p}_times_u_x{k}"] = (u.astype(np.float64) ** p) * d
    return bank


def flatten_bank(bank: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Flatten each (T, N) feature to a (T*N,) 1D vector for PySR consumption.

    PySR doesn't know about the 2D field structure — each (t, x) location
    is treated as an independent sample.
    """
    return {name: arr.flatten().astype(np.float64) for name, arr in bank.items()}


# ----------------- Synthetic PDE generators (for the demo benchmark) -----------------

def solve_heat_1d(*, nx: int = 64, nt: int = 200, dx: float = 0.1, dt: float = 0.001,
                    kappa: float = 1.0, ic: np.ndarray | None = None) -> tuple[np.ndarray, float, float]:
    """Solve the 1D heat equation u_t = κ·u_xx via explicit Euler.

    Stability requires κ·dt/dx² < 1/2.

    Returns
    -------
    u : (nt, nx) — trajectory snapshots
    dx, dt : grid parameters
    """
    if ic is None:
        x = np.arange(nx) * dx
        ic = np.exp(-((x - nx * dx / 2.0) ** 2) / (nx * dx * 0.05))
    u = np.empty((nt, nx), dtype=np.float64)
    u[0] = ic.astype(np.float64)
    for t in range(nt - 1):
        u_xx = diff_x(u[t], dx=dx, order=2)
        u[t+1] = u[t] + dt * kappa * u_xx
    return u, dx, dt


def solve_burgers_1d(*, nx: int = 128, nt: int = 400, dx: float = 0.05, dt: float = 0.0005,
                       nu: float = 0.1, ic: np.ndarray | None = None) -> tuple[np.ndarray, float, float]:
    """Solve Burgers' equation u_t = -u·u_x + ν·u_xx via explicit Euler.

    Returns
    -------
    u : (nt, nx), dx, dt
    """
    if ic is None:
        x = np.arange(nx) * dx
        L = nx * dx
        ic = np.sin(2.0 * np.pi * x / L)
    u = np.empty((nt, nx), dtype=np.float64)
    u[0] = ic.astype(np.float64)
    for t in range(nt - 1):
        u_x = diff_x(u[t], dx=dx, order=1)
        u_xx = diff_x(u[t], dx=dx, order=2)
        u[t+1] = u[t] + dt * (-u[t] * u_x + nu * u_xx)
    return u, dx, dt


def solve_kdv_1d(*, nx: int = 256, nt: int = 400, dx: float = 0.1, dt: float = 0.0001,
                    ic: np.ndarray | None = None) -> tuple[np.ndarray, float, float]:
    """Solve the KdV equation u_t = -6u·u_x − u_xxx via explicit Euler.

    KdV is stiff; small dt required for stability. Returns (nt, nx).
    """
    if ic is None:
        x = np.arange(nx) * dx
        L = nx * dx
        # Soliton-like initial condition
        ic = 0.5 / np.cosh(0.5 * (x - L / 3.0)) ** 2
    u = np.empty((nt, nx), dtype=np.float64)
    u[0] = ic.astype(np.float64)
    for t in range(nt - 1):
        u_x = diff_x(u[t], dx=dx, order=1)
        u_xxx = diff_x(u[t], dx=dx, order=3)
        u[t+1] = u[t] + dt * (-6.0 * u[t] * u_x - u_xxx)
    return u, dx, dt


__all__ = [
    "diff_x", "diff_t", "laplacian_1d", "laplacian_2d",
    "pde_feature_bank_1d", "flatten_bank",
    "solve_heat_1d", "solve_burgers_1d", "solve_kdv_1d",
]
