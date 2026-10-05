"""Correlated parametric UQ: sample from the Gauss-Newton covariance and
propagate through the batched RK4 in one vectorized call.

Replaces the audited scheme (independent hand-picked sigmas), which ignored the
strong parameter correlations of this inverse problem (mu-eta corr ~ -0.9999).
Epistemic network uncertainty is NOT included (the liftoff prediction is a
physics rollout with the identified parameters, so the network does not
enter it).
"""
import numpy as np

from .config import ParamSpace, PhysicsConfig
from .evaluate import predict_liftoff
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
    """Two propagations of the SAME parameter samples (one batched RK4 call):
    - *_param: parameter uncertainty only (what the data told us),
    - total:   plus run-to-run input variability (rpm, headwind offsets), i.e.
               a prediction interval for a repeat takeoff.
    v1 reported only the mixture, which hid how little of the spread is
    parametric in the slide scenario."""
    rng = np.random.default_rng(seed)
    samples = sample_correlated_params(theta, cov, names, space, n_samples, rng)
    full = {k: np.full(n_samples, theta[k]) for k in ("mu", "cd_scale", "cl_scale", "eta")}
    for j, name in enumerate(names):
        full[name] = samples[:, j]

    rpm_off = rng.normal(0.0, sigma_rpm, n_samples) if sigma_rpm > 0 else np.zeros(n_samples)
    hw_off = rng.normal(0.0, sigma_headwind, n_samples) if sigma_headwind > 0 else np.zeros(n_samples)

    twice = {k: np.concatenate([v, v]) for k, v in full.items()}
    zeros = np.zeros(n_samples)
    ev = predict_liftoff(twice["mu"], twice["cd_scale"], twice["cl_scale"], twice["eta"], phys,
                         rpm_offset=np.concatenate([zeros, rpm_off]),
                         headwind_offset=np.concatenate([zeros, hw_off]))
    n = n_samples
    return {"params": samples, "names": names,
            "t_liftoff_param": ev["t"][:n], "x_liftoff_param": ev["x"][:n],
            "t_liftoff": ev["t"][n:], "x_liftoff": ev["x"][n:], "x_final": ev["x_TF"][n:]}
