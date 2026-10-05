import numpy as np
import torch

from takeoff_pinn.config import ParamSpace, PhysicsConfig
from takeoff_pinn.evaluate import evaluate_model, first_crossing
from takeoff_pinn.model import TakeoffPINN

torch.set_default_dtype(torch.float64)


def test_first_crossing_linear_interpolation_exact():
    t = np.array([0.0, 1.0, 2.0])
    v = np.array([0.0, 10.0, 20.0])
    x = np.array([0.0, 5.0, 20.0])
    tc, xc = first_crossing(t, x, v, threshold=15.0)
    assert abs(tc - 1.5) < 1e-12 and abs(xc - 12.5) < 1e-12


def test_first_crossing_no_event_is_nan():
    t = np.array([0.0, 1.0])
    v = np.array([0.0, 1.0])
    x = np.array([0.0, 0.5])
    tc, xc = first_crossing(t, x, v, threshold=99.0)
    assert np.isnan(tc) and np.isnan(xc)


def test_evaluate_model_shapes_and_keys():
    phys = PhysicsConfig()
    model = TakeoffPINN(phys, ParamSpace())
    fit = evaluate_model(model, phys, n=128)
    for k in ("t", "x", "V", "dVdt", "r_dyn", "r_kin", "mu", "cd_scale", "cl_scale", "eta"):
        assert k in fit
    assert len(fit["t"]) == 128 and np.isfinite(fit["r_dyn"]).all()
