import numpy as np

from takeoff_pinn.config import DataConfig, ParamSpace, PhysicsConfig, TrainConfig, TrueParams
from takeoff_pinn.data import make_synthetic_data
from takeoff_pinn.oe import output_error_fit

PHYS, TRUTH = PhysicsConfig(), TrueParams()


def test_oe_recovers_truth_on_clean_data():
    """With zero noise and fixed cl the OE stage must nail the parameters."""
    d = make_synthetic_data(PHYS, TRUTH, DataConfig(n_data=80, sigma_v=0.0, sigma_x=0.0), seed=5)
    theta, info = output_error_fit(d["t_data"], d["x_data"], d["v_data"],
                                   PHYS, ParamSpace(), TrainConfig(), seed=0)
    assert abs(theta["mu"] - TRUTH.mu) < 5e-3
    assert abs(theta["cd_scale"] - TRUTH.cd_scale) < 5e-2
    assert abs(theta["eta"] - TRUTH.eta) < 5e-3
    assert theta["cl_scale"] == ParamSpace().prior_cl  # fixed, not estimated
    assert np.isfinite(info["loss"])


def test_oe_stays_within_bounds_on_noisy_data():
    d = make_synthetic_data(PHYS, TRUTH, DataConfig(n_data=80), seed=6)
    theta, _ = output_error_fit(d["t_data"], d["x_data"], d["v_data"],
                                PHYS, ParamSpace(), TrainConfig(), seed=1)
    sp = ParamSpace()
    assert sp.mu_bounds[0] <= theta["mu"] <= sp.mu_bounds[1]
    assert sp.eta_bounds[0] <= theta["eta"] <= sp.eta_bounds[1]
