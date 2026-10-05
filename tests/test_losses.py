import torch
import torch.nn as nn

from takeoff_pinn.config import ParamSpace, PhysicsConfig, TrainConfig
from takeoff_pinn.losses import compute_loss, physics_ramp, prior_weight, residuals
from takeoff_pinn.model import TakeoffPINN
from takeoff_pinn.physics import rhs_accel

torch.set_default_dtype(torch.float64)
PHYS = PhysicsConfig()


class AnalyticModel(nn.Module):
    """x = t^3/3, V = t^2 -> r_kin must be exactly 0; r_dyn = 2t - rhs."""

    def forward(self, t):
        return t**3 / 3.0, t**2

    def params_physical(self):
        return (torch.tensor(0.08), torch.tensor(1.15),
                torch.tensor(1.0), torch.tensor(0.91))


def test_residuals_against_analytic_model():
    m = AnalyticModel()
    t = torch.linspace(0.1, 7.9, 40).reshape(-1, 1)
    r_dyn, r_kin = residuals(m, t, PHYS)
    assert torch.allclose(r_kin, torch.zeros_like(r_kin), atol=1e-10)
    t_ref = t.clone()
    expected = 2.0 * t_ref - rhs_accel(t_ref, t_ref**2, *m.params_physical(), PHYS)
    assert torch.allclose(r_dyn.detach(), expected, atol=1e-10)


def test_curriculum_monotone_and_final_weights():
    tc = TrainConfig()
    assert physics_ramp(0, tc) < 0.1 < physics_ramp(tc.epochs_adam, tc)
    assert physics_ramp(None, tc) == 1.0           # epoch=None -> final weights
    assert prior_weight(None, tc) == tc.w_prior_end
    # Priors are OFF by default: even decayed weights bias cd_scale to the
    # prior mean (~3.4 sigma) because the residual carries little cd info.
    assert tc.w_prior_start == 0.0 and tc.w_prior_end == 0.0
    tc_real = TrainConfig(w_prior_start=0.04, w_prior_end=0.001)
    assert prior_weight(0, tc_real) > prior_weight(tc_real.epochs_adam, tc_real)


def test_compute_loss_finite_and_keyed():
    model = TakeoffPINN(PHYS, ParamSpace())
    t = torch.linspace(0.0, 8.0, 30).reshape(-1, 1)
    x = 0.5 * t**2
    v = t.clone()
    t_phys = torch.linspace(0.05, 7.95, 16).reshape(-1, 1).requires_grad_(True)
    scales = {"v": 14.0, "x": 70.0, "a": PHYS.a_scale}
    total, parts = compute_loss(model, t, x, v, t_phys, 100, PHYS, ParamSpace(),
                                TrainConfig(), scales)
    assert torch.isfinite(total)
    for key in ("v_data", "x_data", "dyn", "kin", "prior", "total"):
        assert key in parts
