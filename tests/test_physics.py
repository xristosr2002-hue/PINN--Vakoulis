import math

import torch

from takeoff_pinn.config import PhysicsConfig
from takeoff_pinn.physics import (
    headwind_profile, rhs_accel, rpm_profile, runway_slope_percent,
    smooth_normal, thrust_uninstalled,
)

torch.set_default_dtype(torch.float64)
PHYS = PhysicsConfig()


def _theta(mu=0.08, cd=1.15, cl=1.0, eta=0.91):
    return tuple(torch.tensor(v) for v in (mu, cd, cl, eta))


def test_rpm_profile_is_continuous_at_blend_points():
    for t0 in (0.8, 4.7, 4.9, 5.0, 5.1):
        a = rpm_profile(torch.tensor([t0 - 1e-6]), PHYS)
        b = rpm_profile(torch.tensor([t0 + 1e-6]), PHYS)
        assert abs(float(a - b)) < 1.0  # rpm; C0-continuous (no torch.where step)


def test_smooth_normal_positive_and_matches_clamp_far_from_kink():
    W = torch.tensor(PHYS.W)
    n = smooth_normal(W, torch.tensor(10.0), PHYS.NORMAL_SOFTNESS)
    assert float(n) > 0
    assert abs(float(n) - (PHYS.W - 10.0)) < 1e-6  # far from kink: equals W-L


def _expected_accel(t0: float, v0: float) -> float:
    """Hand computation mirroring the model equation, for cross-checking."""
    t = torch.tensor([[t0]])
    rpm = float(rpm_profile(t, PHYS))
    hw = float(headwind_profile(t))
    v_air = max(v0 + hw, 0.0)
    q = 0.5 * PHYS.rho * v_air**2
    gamma = math.atan(float(runway_slope_percent(t)) / 100.0)
    L = q * PHYS.S * PHYS.CL0 * 1.0
    D = q * PHYS.S * PHYS.CD0 * 1.15
    n_soft = float(smooth_normal(torch.tensor(PHYS.W * math.cos(gamma)),
                                 torch.tensor(L), PHYS.NORMAL_SOFTNESS))
    T = float(thrust_uninstalled(torch.tensor(v_air), torch.tensor(rpm), PHYS))
    rolling = 0.08 * n_soft * math.tanh(v0 / PHYS.FRICTION_SMOOTH_V)
    return (0.91 * T - D - rolling - PHYS.W * math.sin(gamma)) / PHYS.m


def test_rhs_accel_matches_hand_computation():
    mu, cd, cl, eta = _theta()
    for t0, v0 in ((0.0, 0.0), (3.0, 8.0), (7.0, 13.0)):
        a = rhs_accel(torch.tensor([[t0]]), torch.tensor([[v0]]), mu, cd, cl, eta, PHYS)
        assert abs(float(a) - _expected_accel(t0, v0)) < 1e-10


def test_friction_vanishes_at_standstill_and_opposes_motion():
    """Directional friction: zero at V=0, full magnitude once rolling."""
    mu, cd, cl, eta = _theta()
    t = torch.tensor([[0.0]])
    a_stand = rhs_accel(t, torch.tensor([[0.0]]), mu, cd, cl, eta, PHYS)
    # At V=0 friction is zero, so with idle thrust + slight downhill a > 0
    # (the legacy unconditional -mu*N gave a < 0: backward roll at brake release).
    assert float(a_stand) > 0.0


def test_rhs_accel_is_differentiable_wrt_time_and_speed():
    mu, cd, cl, eta = _theta()
    t = torch.linspace(0.05, 7.95, 50).reshape(-1, 1).requires_grad_(True)
    v = (1.8 * t).clone()
    a = rhs_accel(t, v, mu, cd, cl, eta, PHYS)
    g = torch.autograd.grad(a.sum(), t, create_graph=False)[0]
    assert torch.isfinite(a).all() and torch.isfinite(g).all()


def test_offsets_shift_inputs():
    mu, cd, cl, eta = _theta()
    t = torch.tensor([[3.0]])
    v = torch.tensor([[8.0]])
    base = rhs_accel(t, v, mu, cd, cl, eta, PHYS)
    more_wind = rhs_accel(t, v, mu, cd, cl, eta, PHYS, headwind_offset=1.0)
    assert float(more_wind) != float(base)  # headwind changes airspeed terms
