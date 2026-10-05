"""Dense model evaluation, event metrics, and summary statistics."""
import numpy as np
import torch

from .config import PhysicsConfig, TrueParams
from .identifiability import lumped_drag_coefficient
from .losses import grad_wrt
from .physics import rhs_accel


def first_crossing(t, x, v, threshold: float):
    t, x, v = (np.asarray(a, dtype=float) for a in (t, x, v))
    idx = np.where(v >= threshold)[0]
    if len(idx) == 0:
        return float("nan"), float("nan")
    j = int(idx[0])
    if j == 0:
        return float(t[0]), float(x[0])
    frac = float((threshold - v[j - 1]) / (v[j] - v[j - 1] + 1e-15))
    frac = float(np.clip(frac, 0.0, 1.0))
    return (float(t[j - 1] + frac * (t[j] - t[j - 1])),
            float(x[j - 1] + frac * (x[j] - x[j - 1])))


def evaluate_model(model, phys: PhysicsConfig, n: int = 1000) -> dict:
    t_np = np.linspace(0.0, phys.TF, n)
    t = torch.as_tensor(t_np, dtype=torch.get_default_dtype()).reshape(-1, 1).requires_grad_(True)
    x, V = model(t)
    dVdt = grad_wrt(V, t)
    dxdt = grad_wrt(x, t)
    mu, cd, cl, eta = model.params_physical()
    a_rhs = rhs_accel(t, V, mu, cd, cl, eta, phys)
    out = {
        "t": t_np,
        "x": x.detach().numpy().ravel(),
        "V": V.detach().numpy().ravel(),
        "dVdt": dVdt.detach().numpy().ravel(),
        "a_rhs": a_rhs.detach().numpy().ravel(),
        "r_dyn": (dVdt - a_rhs).detach().numpy().ravel(),
        "r_kin": (dxdt - V).detach().numpy().ravel(),
        "mu": float(mu.detach()), "cd_scale": float(cd.detach()),
        "cl_scale": float(cl.detach()), "eta": float(eta.detach()),
    }
    return out


def compute_metrics(fit: dict, data: dict, truth: TrueParams | None, phys: PhysicsConfig) -> dict:
    m = {}
    v_pred = np.interp(data["t_data"], fit["t"], fit["V"])
    x_pred = np.interp(data["t_data"], fit["t"], fit["x"])
    m["v_rmse_data"] = float(np.sqrt(np.mean((v_pred - data["v_data"]) ** 2)))
    m["x_rmse_data"] = float(np.sqrt(np.mean((x_pred - data["x_data"]) ** 2)))
    if data.get("v_true") is not None:
        vt = np.interp(fit["t"], data["t_dense"], data["v_true"])
        xt = np.interp(fit["t"], data["t_dense"], data["x_true"])
        m["v_rmse_true"] = float(np.sqrt(np.mean((fit["V"] - vt) ** 2)))
        m["x_rmse_true"] = float(np.sqrt(np.mean((fit["x"] - xt) ** 2)))
    m["r_dyn_rms"] = float(np.sqrt(np.mean(fit["r_dyn"] ** 2)))
    m["r_kin_rms"] = float(np.sqrt(np.mean(fit["r_kin"] ** 2)))

    tl, xl = first_crossing(fit["t"], fit["x"], fit["V"], phys.V_LIFTOFF)
    m["t_liftoff_pred"], m["x_liftoff_pred"] = tl, xl
    if data.get("v_true") is not None:
        tlt, xlt = first_crossing(data["t_dense"], data["x_true"], data["v_true"], phys.V_LIFTOFF)
        m["t_liftoff_true"], m["x_liftoff_true"] = tlt, xlt
        if np.isfinite(tl) and np.isfinite(tlt):
            m["liftoff_time_abs_err"] = abs(tl - tlt)
            m["ground_roll_abs_err"] = abs(xl - xlt)
            m["ground_roll_pct_err"] = 100.0 * abs(xl - xlt) / max(abs(xlt), 1e-9)
    for name in ("mu", "cd_scale", "cl_scale", "eta"):
        m[name] = fit[name]
        if truth is not None:
            m[f"{name}_abs_err"] = abs(fit[name] - getattr(truth, name))
    m["kappa"] = lumped_drag_coefficient(fit["cd_scale"], fit["cl_scale"], fit["mu"], phys)
    if truth is not None:
        m["kappa_true"] = lumped_drag_coefficient(truth.cd_scale, truth.cl_scale, truth.mu, phys)
        m["kappa_rel_err"] = abs(m["kappa"] - m["kappa_true"]) / abs(m["kappa_true"])
    return m
