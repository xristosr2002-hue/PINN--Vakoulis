"""Stage-0 output-error fit: classical flight-test SID through forward RK4."""
import numpy as np
import torch
from scipy.optimize import least_squares

from .config import ParamSpace, PhysicsConfig, TrainConfig
from .data import fine_grid, integrate_rk4
from .identifiability import _BOUNDS_KEYS, free_parameter_names

_PRIOR_KEYS = {"mu": "prior_mu", "cd_scale": "prior_cd",
               "cl_scale": "prior_cl", "eta": "prior_eta"}


def output_error_fit(t_data, x_data, v_data, phys: PhysicsConfig, space: ParamSpace,
                     tc: TrainConfig, seed: int = 0):
    t_np = np.asarray(t_data, dtype=float)
    order = np.argsort(t_np)
    t_np = t_np[order]
    x_np = np.asarray(x_data, dtype=float)[order]
    v_np = np.asarray(v_data, dtype=float)[order]
    # Integrate on a fine uniform grid and interpolate at the data times.
    # Integrating on the raw (gappy) data grid -- as the legacy OE stage did --
    # injects O(dt^4) RK4 error larger than the parameter effects being fitted,
    # which biases the estimate along the mu-eta valley.
    t_fine = fine_grid(t_np, phys.TF)

    v_scale = max(1.0, float(np.nanmax(np.abs(v_np))))
    x_scale = max(1.0, float(np.nanmax(np.abs(x_np))))

    names = free_parameter_names(space)
    lows = np.array([getattr(space, _BOUNDS_KEYS[n])[0] for n in names])
    highs = np.array([getattr(space, _BOUNDS_KEYS[n])[1] for n in names])
    priors = np.array([getattr(space, _PRIOR_KEYS[n]) for n in names])

    def to_bounded(z):
        return lows + (highs - lows) / (1.0 + np.exp(-z))

    def theta_of(z):
        vals = to_bounded(z)
        theta = {"cl_scale": float(space.prior_cl)}
        theta.update({n: float(v) for n, v in zip(names, vals)})
        return theta

    def resid_batch(Z):
        """Residual rows for a batch of unconstrained parameter vectors (B, p)."""
        vals = to_bounded(np.atleast_2d(Z))
        th = {"cl_scale": np.full(len(vals), float(space.prior_cl))}
        th.update({n: vals[:, j] for j, n in enumerate(names)})
        x_sim, v_sim = integrate_rk4(t_fine, *(torch.as_tensor(th[k]) for k in
                                               ("mu", "cd_scale", "cl_scale", "eta")), phys)
        rows = []
        for xb, vb in zip(x_sim.numpy(), v_sim.numpy()):
            rv = (np.interp(t_np, t_fine, vb) - v_np) / v_scale
            rx = (np.interp(t_np, t_fine, xb) - x_np) / x_scale
            rows.append(np.concatenate([rv, rx]))
        # Pure maximum-likelihood: no prior residual. The legacy OE prior
        # (weight 2e-3) was ~100x stiffer than the data curvature along the
        # mu-eta valley, so it silently dictated the estimate there. Bounds
        # via the sigmoid transform are the only regularization; the valley
        # uncertainty is reported honestly through the GN covariance.
        return np.array(rows)

    def resid(z):
        return resid_batch(z)[0]

    def jac(z):
        # scipy's default '2-point' forward difference, but all p perturbed
        # trajectories go through ONE batched RK4 call (cost ~ one solve).
        sign = np.where(z >= 0, 1.0, -1.0)
        h = np.sqrt(np.finfo(float).eps) * sign * np.maximum(1.0, np.abs(z))
        Z = np.vstack([z, z + np.diag(h)])
        R = resid_batch(Z)
        dx = (z + h) - z
        return ((R[1:] - R[0]) / dx[:, None]).T

    p0 = np.clip((priors - lows) / (highs - lows), 1e-6, 1 - 1e-6)
    jitter = np.array([1.5, -2.0, 1.0]) if len(names) == 3 else np.array([1.5, -2.0, -0.8, 1.0])
    z0 = np.log(p0 / (1.0 - p0)) + 0.1 * seed * jitter
    def solve(z_start, row_w):
        sw = np.sqrt(row_w)
        return least_squares(lambda z: sw * resid(z), z_start,
                             jac=lambda z: sw[:, None] * jac(z),
                             max_nfev=tc.oe_max_nfev, xtol=1e-10, ftol=1e-10, gtol=1e-10).x

    n_rows = 2 * len(t_np)
    weights = np.ones(n_rows)
    z = solve(z0, weights)
    if tc.oe_weighted:
        # Output-error relaxation (Klein & Morelli): estimate the per-channel
        # noise variance R from the residuals, refit minimizing r^T R^-1 r,
        # repeat. The normalized V and x noise differ ~5x, so equal weights
        # waste the information in the cleaner channel.
        for _ in range(tc.oe_relax_iters):
            r = resid(z)
            var = np.concatenate([np.full(len(b), max(np.mean(b**2), 1e-300))
                                  for b in np.split(r, 2)])
            weights = 1.0 / var
            z = solve(z, weights)
    theta = theta_of(z)
    fvec = resid(z)
    info = {"loss": float(np.mean(fvec**2)),
            "residual_data": fvec[:n_rows],
            "weights": weights if tc.oe_weighted else None,
            "t": t_np,  # sample times of residual_data rows ([V; x] blocks, sorted)
            "scales": {"v": v_scale, "x": x_scale}}
    return theta, info
