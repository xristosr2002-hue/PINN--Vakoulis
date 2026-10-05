import torch

from takeoff_pinn.config import ParamSpace, PhysicsConfig
from takeoff_pinn.model import BoundedScalar, TakeoffPINN

torch.set_default_dtype(torch.float64)


def test_bounded_scalar_stays_in_bounds_and_inits_at_value():
    b = BoundedScalar(0.09, 0.02, 0.16)
    assert abs(float(b()) - 0.09) < 1e-9
    with torch.no_grad():
        b.raw.fill_(100.0)
    assert float(b()) <= 0.16 + 1e-12


def test_hard_initial_conditions_exact():
    phys = PhysicsConfig(V0=2.0, X0=1.5)
    model = TakeoffPINN(phys, ParamSpace())
    t0 = torch.zeros(1, 1, requires_grad=True)
    x0, v0 = model(t0)
    assert abs(float(x0) - 1.5) < 1e-12
    assert abs(float(v0) - 2.0) < 1e-12
    dxdt0 = torch.autograd.grad(x0.sum(), t0)[0]
    assert abs(float(dxdt0) - 2.0) < 1e-9  # dx/dt(0) = V0 by construction


def test_cl_fixed_by_default_but_estimable_on_request():
    phys = PhysicsConfig()
    fixed = TakeoffPINN(phys, ParamSpace(estimate_cl=False))
    _, _, cl, _ = fixed.params_physical()
    assert not cl.requires_grad and abs(float(cl) - 1.0) < 1e-12

    free = TakeoffPINN(phys, ParamSpace(estimate_cl=True))
    _, _, cl2, _ = free.params_physical()
    assert cl2.requires_grad


def test_gradients_reach_physical_parameters():
    phys = PhysicsConfig()
    model = TakeoffPINN(phys, ParamSpace())
    mu, cd, cl, eta = model.params_physical()
    (mu + cd + eta).backward()
    assert model.mu.raw.grad is not None and model.eta.raw.grad is not None
