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
from .physics import normal_force_margin, rhs_accel


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

    # Observed liftoff (C9): g = W*cos(gamma) - L must vanish at t_obs. Weighted
    # like one more data point of the V channel, i.e. 0.5*s_v^2/n per unit
    # (g/sigma_g)^2, so it carries exactly its likelihood share of information.
    ev = scales.get("event")
    if ev is not None:
        t_e = torch.full((1, 1), ev["t"], dtype=t_data.dtype)
        _, v_e = model(t_e)
        g = normal_force_margin(t_e, v_e, cl, phys)
        loss_event = ev["weight"] * ((g / ev["sigma_g"]) ** 2).sum()
    else:
        loss_event = torch.zeros((), dtype=t_data.dtype)

    ramp = physics_ramp(epoch, tc)
    w_pr = prior_weight(epoch, tc)
    total = (tc.w_v * loss_v + tc.w_x * scales.get("w_x_rel", 1.0) * loss_x
             + tc.w_dyn * ramp * loss_dyn + tc.w_kin * ramp * loss_kin
             + w_pr * loss_prior + loss_event)

    parts = {
        "total": float(total.detach()), "v_data": float(loss_v.detach()),
        "x_data": float(loss_x.detach()), "dyn": float(loss_dyn.detach()),
        "kin": float(loss_kin.detach()), "prior": float(loss_prior.detach()),
        "event": float(loss_event.detach()),
        "ramp": ramp, "w_prior": w_pr,
    }
    return total, parts
