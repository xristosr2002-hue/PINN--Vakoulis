import numpy as np
import torch

from takeoff_pinn.config import ParamSpace, PhysicsConfig, TrainConfig
from takeoff_pinn.collocation import latin_hypercube, make_collocation, topk_residual_points
from takeoff_pinn.model import TakeoffPINN

torch.set_default_dtype(torch.float64)


def test_latin_hypercube_stratified():
    rng = np.random.default_rng(0)
    pts = latin_hypercube(50, 0.0, 1.0, rng)
    assert pts.min() >= 0.0 and pts.max() <= 1.0
    counts, _ = np.histogram(pts, bins=50, range=(0.0, 1.0))
    assert (counts == 1).all()  # exactly one point per stratum


def test_make_collocation_in_domain_with_extras():
    phys, tc = PhysicsConfig(), TrainConfig(n_collocation=40)
    rng = np.random.default_rng(1)
    t = make_collocation(phys, tc, rng, extra=np.array([1.0, 2.0]))
    t_np = t.detach().numpy().ravel()
    assert t.requires_grad
    assert t_np.min() >= 0.04 and t_np.max() <= phys.TF
    assert np.isin([1.0, 2.0], t_np).all()


def test_topk_residual_points_within_domain():
    phys, tc = PhysicsConfig(), TrainConfig()
    model = TakeoffPINN(phys, ParamSpace())
    rng = np.random.default_rng(2)
    scales = {"v": 14.0, "a": phys.a_scale}
    pts = topk_residual_points(model, phys, scales, n_add=10, rng=rng)
    assert len(pts) <= 10 and pts.min() >= 0.04 and pts.max() <= phys.TF
