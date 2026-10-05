"""Finite-difference sensitivity analysis and Gauss-Newton parameter covariance.

The sensitivity matrix S stacks d[V/v_scale; x/x_scale]/d(theta_k) columns from
central differences through the forward RK4 model. It powers:
- the identifiability report (condition number, pairwise correlations),
- the Laplace/Gauss-Newton covariance used for *correlated* Monte-Carlo UQ
  (replaces the audited independent hand-sigma sampling).
"""
import numpy as np

from .config import ParamSpace, PhysicsConfig
from .data import integrate_rk4

_BOUNDS_KEYS = {"mu": "mu_bounds", "cd_scale": "cd_bounds",
                "cl_scale": "cl_bounds", "eta": "eta_bounds"}


def lumped_drag_coefficient(cd_scale, cl_scale, mu, phys: PhysicsConfig):
    """kappa = CD0*cd - mu*CL0*cl: the only data-identifiable combination of
    (cd, cl) while N > 0, because the resistance is mu*W*cos(g) + q*S*kappa.
    With cl fixed, the reported cd is conditional on that choice; kappa is not."""
    return phys.CD0 * cd_scale - mu * phys.CL0 * cl_scale


def free_parameter_names(space: ParamSpace) -> list[str]:
    names = ["mu", "cd_scale", "eta"]
    if space.estimate_cl:
        names.insert(2, "cl_scale")
    return names


def _simulate(theta: dict, t_eval: np.ndarray, phys: PhysicsConfig, scales: dict) -> np.ndarray:
    # Fine-grid integration + interpolation at the evaluation times keeps the
    # forward model accurate regardless of how gappy the data grid is.
    t_fine = np.union1d(t_eval, np.linspace(0.0, phys.TF, 401))
    x, v = integrate_rk4(t_fine, theta["mu"], theta["cd_scale"],
                         theta["cl_scale"], theta["eta"], phys)
    vs = np.interp(t_eval, t_fine, v[0].numpy())
    xs = np.interp(t_eval, t_fine, x[0].numpy())
    return np.concatenate([vs / scales["v"], xs / scales["x"]])


def sensitivity_matrix(theta: dict, t_data: np.ndarray, phys: PhysicsConfig,
                       space: ParamSpace, scales: dict):
    t_sim = np.asarray(t_data, dtype=float)
    names = free_parameter_names(space)
    cols = []
    for name in names:
        lo, hi = getattr(space, _BOUNDS_KEYS[name])
        h = 0.01 * (hi - lo)
        tp, tm = dict(theta), dict(theta)
        tp[name] = float(np.clip(theta[name] + h, lo + 1e-7, hi - 1e-7))
        tm[name] = float(np.clip(theta[name] - h, lo + 1e-7, hi - 1e-7))
        cols.append((_simulate(tp, t_sim, phys, scales)
                     - _simulate(tm, t_sim, phys, scales)) / (tp[name] - tm[name]))
    return np.column_stack(cols), names


def identifiability_report(S: np.ndarray, names: list[str]) -> dict:
    svals = np.linalg.svd(S, compute_uv=False)
    corr = np.corrcoef(S.T) if S.shape[1] > 1 else np.ones((1, 1))
    off = np.abs(corr - np.eye(len(names)))
    return {
        "names": names,
        "condition_number": float(svals[0] / max(svals[-1], 1e-15)),
        "corr": corr.tolist(),
        "max_corr": float(off.max()),
    }


def gauss_newton_covariance(S: np.ndarray, residual: np.ndarray) -> np.ndarray:
    """cov = sigma^2 (S^T S)^-1 with sigma^2 from the normalized residual SSR."""
    n, p = S.shape
    dof = max(n - p, 1)
    sigma2 = float(residual @ residual) / dof
    return sigma2 * np.linalg.pinv(S.T @ S)
