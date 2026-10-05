import numpy as np

from takeoff_pinn.config import ParamSpace, PhysicsConfig
from takeoff_pinn.uq import propagate_uncertainty, sample_correlated_params

PHYS = PhysicsConfig()
THETA = {"mu": 0.08, "cd_scale": 1.15, "cl_scale": 1.0, "eta": 0.91}


def _tiny_cov(names):
    return np.diag([1e-6] * len(names))


def test_sampling_respects_bounds_and_mean():
    sp = ParamSpace()
    names = ["mu", "cd_scale", "eta"]
    cov = np.diag([0.005**2, 0.05**2, 0.01**2])
    rng = np.random.default_rng(0)
    samples = sample_correlated_params(THETA, cov, names, sp, 400, rng)
    assert samples.shape == (400, 3)
    assert samples[:, 0].min() >= sp.mu_bounds[0] and samples[:, 0].max() <= sp.mu_bounds[1]
    assert abs(samples[:, 0].mean() - THETA["mu"]) < 2e-3


def test_propagation_outputs_distributions():
    sp = ParamSpace()
    names = ["mu", "cd_scale", "eta"]
    out = propagate_uncertainty(THETA, _tiny_cov(names), names, sp, PHYS,
                                n_samples=32, seed=1, sigma_rpm=0.0, sigma_headwind=0.0)
    assert out["x_final"].shape == (32,)
    assert np.isfinite(out["x_final"]).all()
    assert np.isfinite(out["x_liftoff"]).any()
    # near-zero covariance and no input noise -> near-deterministic outputs
    assert out["x_final"].std() < 0.5


def test_parametric_and_total_spreads_are_separated():
    """Same parameter samples; input variability can only add spread."""
    sp = ParamSpace()
    names = ["mu", "cd_scale", "eta"]
    out = propagate_uncertainty(THETA, _tiny_cov(names), names, sp, PHYS,
                                n_samples=32, seed=1, sigma_rpm=40.0, sigma_headwind=0.08)
    assert out["x_liftoff_param"].shape == out["x_liftoff"].shape == (32,)
    assert np.nanstd(out["x_liftoff_param"]) < np.nanstd(out["x_liftoff"])
