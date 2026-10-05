"""Stage-0 output-error fit: classical flight-test SID through forward RK4."""
import numpy as np
from scipy.optimize import least_squares

from .config import ParamSpace, PhysicsConfig, TrainConfig
from .data import integrate_rk4
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
    t_fine = np.union1d(t_np, np.linspace(0.0, phys.TF, 401))

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

    def resid(z):
        th = theta_of(z)
        x_sim, v_sim = integrate_rk4(t_fine, th["mu"], th["cd_scale"],
                                     th["cl_scale"], th["eta"], phys)
        xs = np.interp(t_np, t_fine, x_sim[0].numpy())
        vs = np.interp(t_np, t_fine, v_sim[0].numpy())
        rv = (vs - v_np) / v_scale
        rx = (xs - x_np) / x_scale
        # Pure maximum-likelihood: no prior residual. The legacy OE prior
        # (weight 2e-3) was ~100x stiffer than the data curvature along the
        # mu-eta valley, so it silently dictated the estimate there. Bounds
        # via the sigmoid transform are the only regularization; the valley
        # uncertainty is reported honestly through the GN covariance.
        return np.concatenate([rv, rx])

    p0 = np.clip((priors - lows) / (highs - lows), 1e-6, 1 - 1e-6)
    jitter = np.array([1.5, -2.0, 1.0]) if len(names) == 3 else np.array([1.5, -2.0, -0.8, 1.0])
    z0 = np.log(p0 / (1.0 - p0)) + 0.1 * seed * jitter
    res = least_squares(resid, z0, max_nfev=tc.oe_max_nfev, xtol=1e-10, ftol=1e-10, gtol=1e-10)
    theta = theta_of(res.x)
    fvec = resid(res.x)
    info = {"loss": float(np.mean(fvec**2)),
            "residual_data": fvec[: 2 * len(t_np)],
            "scales": {"v": v_scale, "x": x_scale}}
    return theta, info
