"""Correlated parametric UQ: sample from the Gauss-Newton covariance and
propagate through the batched RK4 in one vectorized call.

Replaces the audited scheme (independent hand-picked sigmas), which ignored the
strong parameter correlations of this inverse problem (mu-eta corr ~ -0.9999).
Epistemic network uncertainty is still NOT included; intervals are
parametric-only lower bounds.
"""
import numpy as np
import torch

from .config import ParamSpace, PhysicsConfig
from .data import integrate_rk4
from .evaluate import first_crossing
from .identifiability import _BOUNDS_KEYS


def sample_correlated_params(theta: dict, cov: np.ndarray, names: list[str],
                             space: ParamSpace, n: int, rng: np.random.Generator) -> np.ndarray:
    mean = np.array([theta[k] for k in names])
    cov = 0.5 * (cov + cov.T) + 1e-12 * np.eye(len(names))
    samples = rng.multivariate_normal(mean, cov, size=n)
    for j, name in enumerate(names):
        lo, hi = getattr(space, _BOUNDS_KEYS[name])
        samples[:, j] = np.clip(samples[:, j], lo, hi)
    return samples


def propagate_uncertainty(theta: dict, cov: np.ndarray, names: list[str],
                          space: ParamSpace, phys: PhysicsConfig,
                          n_samples: int = 250, seed: int = 0,
                          sigma_rpm: float = 40.0, sigma_headwind: float = 0.08) -> dict:
    rng = np.random.default_rng(seed)
    samples = sample_correlated_params(theta, cov, names, space, n_samples, rng)
    full = {k: np.full(n_samples, theta[k]) for k in ("mu", "cd_scale", "cl_scale", "eta")}
    for j, name in enumerate(names):
        full[name] = samples[:, j]

    rpm_off = rng.normal(0.0, sigma_rpm, n_samples) if sigma_rpm > 0 else np.zeros(n_samples)
    hw_off = rng.normal(0.0, sigma_headwind, n_samples) if sigma_headwind > 0 else np.zeros(n_samples)

    t_grid = np.linspace(0.0, phys.TF, 600)
    x, v = integrate_rk4(t_grid,
                         torch.as_tensor(full["mu"]), torch.as_tensor(full["cd_scale"]),
                         torch.as_tensor(full["cl_scale"]), torch.as_tensor(full["eta"]),
                         phys,
                         rpm_offset=torch.as_tensor(rpm_off),
                         headwind_offset=torch.as_tensor(hw_off))
    x_np, v_np = x.numpy(), v.numpy()

    t_lift = np.full(n_samples, np.nan)
    x_lift = np.full(n_samples, np.nan)
    for i in range(n_samples):
        t_lift[i], x_lift[i] = first_crossing(t_grid, x_np[i], v_np[i], phys.V_LIFTOFF)

    return {"params": samples, "names": names,
            "x_final": x_np[:, -1], "t_liftoff": t_lift, "x_liftoff": x_lift}
