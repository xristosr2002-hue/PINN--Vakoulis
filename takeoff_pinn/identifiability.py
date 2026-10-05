"""Finite-difference sensitivity analysis and Gauss-Newton parameter covariance.

The sensitivity matrix S stacks d[V/v_scale; x/x_scale]/d(theta_k) columns from
central differences through the forward RK4 model. It powers:
- the identifiability report (condition number, pairwise correlations),
- the Laplace/Gauss-Newton covariance used for *correlated* Monte-Carlo UQ
  (replaces the audited independent hand-sigma sampling).
"""
import numpy as np
import torch

from .config import ParamSpace, PhysicsConfig
from .data import fine_grid, integrate_rk4

_BOUNDS_KEYS = {"mu": "mu_bounds", "cd_scale": "cd_bounds",
                "cl_scale": "cl_bounds", "eta": "eta_bounds"}


def lumped_drag_coefficient(cd_scale, cl_scale, mu, phys: PhysicsConfig):
    """kappa = CD0*cd - mu*CL0*cl: the only data-identifiable combination of
    (cd, cl) while N > 0, because the resistance is mu*W*cos(g) + q*S*kappa.
    With cl fixed, the reported cd is conditional on that choice; kappa is not."""
    return phys.CD0 * cd_scale - mu * phys.CL0 * cl_scale


def kappa_gradient(theta: dict, names: list[str], phys: PhysicsConfig) -> np.ndarray:
    """d(kappa)/d(theta) over the free parameters, for linear sigma propagation."""
    d = {"mu": -phys.CL0 * theta["cl_scale"], "cd_scale": phys.CD0,
         "cl_scale": -theta["mu"] * phys.CL0, "eta": 0.0}
    return np.array([d[n] for n in names])


def free_parameter_names(space: ParamSpace) -> list[str]:
    names = ["mu", "cd_scale", "eta"]
    if space.estimate_cl:
        names.insert(2, "cl_scale")
    return names


def _simulate(thetas: list[dict], t_eval: np.ndarray, phys: PhysicsConfig,
              scales: dict) -> np.ndarray:
    """Normalized [V; x] outputs for a list of parameter sets, one batched RK4.
    Fine-grid integration + interpolation at the evaluation times keeps the
    forward model accurate regardless of how gappy the data grid is."""
    t_fine = fine_grid(t_eval, phys.TF)
    batch = [torch.tensor([th[k] for th in thetas])
             for k in ("mu", "cd_scale", "cl_scale", "eta")]
    x, v = integrate_rk4(t_fine, *batch, phys)
    return np.array([np.concatenate([np.interp(t_eval, t_fine, vb) / scales["v"],
                                     np.interp(t_eval, t_fine, xb) / scales["x"]])
                     for xb, vb in zip(x.numpy(), v.numpy())])


def sensitivity_matrix(theta: dict, t_data: np.ndarray, phys: PhysicsConfig,
                       space: ParamSpace, scales: dict):
    """Central-difference d(outputs)/d(theta_k), all 2p perturbations in one batch."""
    t_sim = np.asarray(t_data, dtype=float)
    names = free_parameter_names(space)
    perturbed, steps = [], []
    for name in names:
        lo, hi = getattr(space, _BOUNDS_KEYS[name])
        h = 0.01 * (hi - lo)
        tp, tm = dict(theta), dict(theta)
        tp[name] = float(np.clip(theta[name] + h, lo + 1e-7, hi - 1e-7))
        tm[name] = float(np.clip(theta[name] - h, lo + 1e-7, hi - 1e-7))
        perturbed += [tp, tm]
        steps.append(tp[name] - tm[name])
    out = _simulate(perturbed, t_sim, phys, scales)
    cols = [(out[2 * j] - out[2 * j + 1]) / steps[j] for j in range(len(names))]
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


def parameter_covariance(S: np.ndarray, residual: np.ndarray, n_channels: int = 2,
                         weights: np.ndarray | None = None) -> np.ndarray:
    """Sandwich covariance of a (weighted) least-squares estimate.

    Rows of S and `residual` are equal-length channel blocks ([V; x]), aligned
    row by row. Noise variance is estimated *per channel*: the normalized V and
    x residuals differ ~5x in scale, so a pooled sigma^2 misstates the
    uncertainty. With weights=None (the unweighted OE estimator) this is
    A^-1 S^T Sigma S A^-1, A = S^T S; with weights = 1/variance it reduces to
    the inverse Fisher information (S^T Sigma^-1 S)^-1.
    """
    n, p = S.shape
    if residual.shape[0] != n:
        raise ValueError(f"residual length {residual.shape[0]} != S rows {n}: "
                         "sensitivities and residuals must use the same samples")
    dof_scale = n / max(n - p, 1)
    var = np.concatenate([np.full(len(b), np.mean(b**2) * dof_scale)
                          for b in np.split(residual, n_channels)])
    w = np.ones(n) if weights is None else np.asarray(weights, dtype=float)
    A_inv = np.linalg.pinv((S.T * w) @ S)
    meat = (S.T * (w**2 * var)) @ S
    return A_inv @ meat @ A_inv


def gauss_newton_covariance(S: np.ndarray, residual: np.ndarray) -> np.ndarray:
    """cov = sigma^2 (S^T S)^-1 with sigma^2 from the normalized residual SSR."""
    n, p = S.shape
    dof = max(n - p, 1)
    sigma2 = float(residual @ residual) / dof
    return sigma2 * np.linalg.pinv(S.T @ S)
