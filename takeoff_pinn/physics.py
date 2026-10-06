"""Single source of truth for the ground-roll dynamics (torch, float64).

Every consumer (data generator, output-error initializer, PINN residual,
UQ rollout) calls rhs_accel; there is deliberately no numpy twin.
All input profiles are C1-smooth so autograd residuals are well behaved.
"""
import math

import torch
import torch.nn.functional as F

from .config import PhysicsConfig


def smoothblend(z: torch.Tensor) -> torch.Tensor:
    """C-infinity 0->1 blend over z in [0, 1] (sigmoid; ~0.25% tails outside).

    The legacy cubic smoothstep is only C1: its second-derivative jumps degrade
    RK4 from 4th to ~2nd order and roughen autograd residuals at the blend
    boundaries. A sigmoid blend is smooth everywhere.
    """
    return torch.sigmoid(12.0 * (z - 0.5))


def rpm_profile(t: torch.Tensor, cfg: PhysicsConfig) -> torch.Tensor:
    """Spool-up + persistent excitation + smooth blend to RPM_MAX (no steps)."""
    ramp = smoothblend(t / 1.10)
    base = 1500.0 + (cfg.RPM_MAX - 1500.0) * ramp
    on = smoothblend((t - 0.8) / 0.8)
    off = 1.0 - smoothblend((t - 4.7) / 0.6)
    excite = 160.0 * torch.sin(2.0 * math.pi * (t - 0.5) / 4.6) * on * off
    rpm = base + excite
    fade = smoothblend((t - 4.9) / 0.2)
    rpm = rpm * (1.0 - fade) + cfg.RPM_MAX * fade
    return torch.clamp(rpm, 900.0, cfg.RPM_MAX * 1.05)


def headwind_profile(t: torch.Tensor) -> torch.Tensor:
    """Positive = headwind: V_air = V_ground + headwind."""
    return 0.05 + 0.10 * torch.sin(2.0 * math.pi * (t + 0.3) / 7.5)


def runway_slope_percent(t: torch.Tensor) -> torch.Tensor:
    """Slope in percent; positive = uphill."""
    return -0.50 + 0.20 * torch.sin(2.0 * math.pi * t / 8.0)


def thrust_uninstalled(v_air: torch.Tensor, rpm: torch.Tensor, cfg: PhysicsConfig) -> torch.Tensor:
    """Propeller surrogate: T ~ rpm^2 with mild advance-ratio loss."""
    rpm_ratio = torch.clamp(rpm / cfg.RPM_MAX, 0.0, 1.20)
    advance_loss = torch.clamp(1.0 - 0.010 * v_air, 0.70, 1.05)
    return cfg.T_MAX * rpm_ratio**2 * advance_loss


def smooth_normal(w_eff: torch.Tensor, lift: torch.Tensor, softness: float) -> torch.Tensor:
    """Smooth max(w_eff - lift, 0); softness in Newtons."""
    return F.softplus((w_eff - lift) / softness) * softness


def normal_force_margin(
    t: torch.Tensor,
    V: torch.Tensor,
    cl_scale: torch.Tensor | float,
    cfg: PhysicsConfig,
    headwind_offset: torch.Tensor | float = 0.0,
) -> torch.Tensor:
    """W*cos(gamma) - L at ground attitude: > 0 while the wheels carry load.

    Liftoff is the first time this reaches 0 (no rotation manoeuvre is
    modelled, so lift uses the ground-attitude C_L0). Beyond it the 1-D
    ground-roll equation no longer describes the vehicle."""
    _, headwind, gamma = exogenous_inputs(t, cfg)
    v_air = torch.clamp(V + headwind + headwind_offset, min=0.0)
    lift = 0.5 * cfg.rho * v_air**2 * cfg.S * cfg.CL0 * cl_scale
    return cfg.W * torch.cos(gamma) - lift


def exogenous_inputs(t: torch.Tensor, cfg: PhysicsConfig):
    """Time-only inputs (rpm, headwind, runway angle gamma). Separated from the
    state-dependent forces so integrators can evaluate them once per grid."""
    rpm = rpm_profile(t, cfg)
    headwind = headwind_profile(t)
    gamma = torch.atan(runway_slope_percent(t) / 100.0)
    return rpm, headwind, gamma


def rhs_accel(
    t: torch.Tensor,
    V: torch.Tensor,
    mu: torch.Tensor,
    cd_scale: torch.Tensor,
    cl_scale: torch.Tensor,
    eta: torch.Tensor,
    cfg: PhysicsConfig,
    rpm_offset: torch.Tensor | float = 0.0,
    headwind_offset: torch.Tensor | float = 0.0,
) -> torch.Tensor:
    """dV/dt = [eta*T - D - mu*N - W*sin(gamma)] / m  (broadcasts over batch)."""
    rpm, headwind, gamma = exogenous_inputs(t, cfg)
    return accel_from_inputs(V, rpm + rpm_offset, headwind + headwind_offset, gamma,
                             mu, cd_scale, cl_scale, eta, cfg)


def accel_from_inputs(
    V: torch.Tensor,
    rpm: torch.Tensor,
    headwind: torch.Tensor,
    gamma: torch.Tensor,
    mu: torch.Tensor,
    cd_scale: torch.Tensor,
    cl_scale: torch.Tensor,
    eta: torch.Tensor,
    cfg: PhysicsConfig,
    mu_speed_slope: float = 0.0,
) -> torch.Tensor:
    """The ground-roll force balance given already-evaluated inputs; the one
    place the equation of motion is written down.

    mu_speed_slope != 0 makes rolling friction grow with ground speed,
    mu*(1 + k*V). The identification model never uses it: it exists only to
    inject a known model-form error into synthetic truth (experiment E10)."""
    v_air = torch.clamp(V + headwind, min=0.0)
    q = 0.5 * cfg.rho * v_air**2
    D = q * cfg.S * cfg.CD0 * cd_scale
    L = q * cfg.S * cfg.CL0 * cl_scale

    N = smooth_normal(cfg.W * torch.cos(gamma), L, cfg.NORMAL_SOFTNESS)
    T = eta * thrust_uninstalled(v_air, rpm, cfg)

    # Rolling friction opposes the direction of motion (smooth sign via tanh on
    # GROUND speed). The legacy scripts applied -mu*N unconditionally, which
    # made the aircraft roll backwards at idle thrust before spool-up; their
    # integrators hid this with a V >= -0.2 clamp that also broke RK4 accuracy.
    if mu_speed_slope:
        mu = mu * (1.0 + mu_speed_slope * V)
    rolling = mu * N * torch.tanh(V / cfg.FRICTION_SMOOTH_V)

    return (T - D - rolling - cfg.W * torch.sin(gamma)) / cfg.m
