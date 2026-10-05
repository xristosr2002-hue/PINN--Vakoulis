"""Normalized Huber data/residual losses with curriculum and weak priors.

Design notes (audit-driven):
- No event loss (it leaked truth in synthetic mode) and no feasibility/jerk
  penalties (heuristics that bias the fit); hard ICs + physics carry the load.
- epoch=None means "final weights": used for validation scoring so checkpoint
  selection is not biased by the curriculum ramp.
"""
import math

import torch
import torch.nn.functional as F

from .config import ParamSpace, PhysicsConfig, TrainConfig
from .physics import rhs_accel


def grad_wrt(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return torch.autograd.grad(y, x, grad_outputs=torch.ones_like(y),
                               create_graph=True, retain_graph=True)[0]


def huber_mean(z: torch.Tensor, delta: float = 1.0) -> torch.Tensor:
    return torch.mean(F.huber_loss(z, torch.zeros_like(z), delta=delta, reduction="none"))


def residuals(model, t_phys: torch.Tensor, phys: PhysicsConfig):
    if not t_phys.requires_grad:
        t_phys = t_phys.clone().requires_grad_(True)
    x, V = model(t_phys)
    dVdt = grad_wrt(V, t_phys)
    dxdt = grad_wrt(x, t_phys)
    mu, cd, cl, eta = model.params_physical()
    r_dyn = dVdt - rhs_accel(t_phys, V, mu, cd, cl, eta, phys)
    r_kin = dxdt - V
    return r_dyn, r_kin


def physics_ramp(epoch: int | None, tc: TrainConfig) -> float:
    if epoch is None:
        return 1.0
    return 1.0 / (1.0 + math.exp(-(epoch - tc.ramp_center) / tc.ramp_width))


def prior_weight(epoch: int | None, tc: TrainConfig) -> float:
    if epoch is None:
        return tc.w_prior_end
    return tc.w_prior_end + (tc.w_prior_start - tc.w_prior_end) * math.exp(-epoch / tc.prior_decay_tau)


def compute_loss(model, t_data, x_data, v_data, t_phys, epoch,
                 phys: PhysicsConfig, space: ParamSpace, tc: TrainConfig,
                 scales: dict):
    x_pred, v_pred = model(t_data)
    loss_v = huber_mean((v_pred - v_data) / scales["v"])
    loss_x = huber_mean((x_pred - x_data) / scales["x"])

    r_dyn, r_kin = residuals(model, t_phys, phys)
    loss_dyn = huber_mean(r_dyn / scales["a"])
    loss_kin = huber_mean(r_kin / scales["v"])

    mu, cd, cl, eta = model.params_physical()
    loss_prior = (((mu - space.prior_mu) / space.prior_sigma_mu) ** 2
                  + ((cd - space.prior_cd) / space.prior_sigma_cd) ** 2
                  + ((eta - space.prior_eta) / space.prior_sigma_eta) ** 2)
    if space.estimate_cl:
        loss_prior = loss_prior + ((cl - space.prior_cl) / space.prior_sigma_cl) ** 2

    ramp = physics_ramp(epoch, tc)
    w_pr = prior_weight(epoch, tc)
    total = (tc.w_v * loss_v + tc.w_x * loss_x
             + tc.w_dyn * ramp * loss_dyn + tc.w_kin * ramp * loss_kin
             + w_pr * loss_prior)

    parts = {
        "total": float(total.detach()), "v_data": float(loss_v.detach()),
        "x_data": float(loss_x.detach()), "dyn": float(loss_dyn.detach()),
        "kin": float(loss_kin.detach()), "prior": float(loss_prior.detach()),
        "ramp": ramp, "w_prior": w_pr,
    }
    return total, parts
