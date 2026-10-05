"""Dense model evaluation, event metrics, and summary statistics."""
import numpy as np
import torch

from .config import PhysicsConfig, TrueParams
from .identifiability import lumped_drag_coefficient
from .data import integrate_rk4
from .losses import grad_wrt
from .physics import normal_force_margin, rhs_accel


def first_event(t, x, g):
    """First time the event function g drops to <= 0 (from > 0), with linear
    interpolation of t and x; (nan, nan) if it never does on the grid."""
    t, x, g = (np.asarray(a, dtype=float) for a in (t, x, g))
    idx = np.where(g <= 0.0)[0]
    if len(idx) == 0:
        return float("nan"), float("nan")
    j = int(idx[0])
    if j == 0:
        return float(t[0]), float(x[0])
    frac = float(np.clip(g[j - 1] / (g[j - 1] - g[j]), 0.0, 1.0))
    return (float(t[j - 1] + frac * (t[j] - t[j - 1])),
            float(x[j - 1] + frac * (x[j] - x[j - 1])))


def predict_liftoff(mu, cd_scale, cl_scale, eta, phys: PhysicsConfig,
                    rpm_offset=0.0, headwind_offset=0.0, horizon: float = 2.0,
                    dt: float = 0.01) -> dict:
    """Liftoff (L = W*cos(gamma)) by forward RK4 with the given parameters,
    integrated up to horizon*TF so events after the record end are still
    found. Batched over parameter sets; returns arrays of t, x, V at liftoff
    and x at TF."""
    t = np.linspace(0.0, horizon * phys.TF, int(round(horizon * phys.TF / dt)) + 1)
    batch = [torch.as_tensor(np.atleast_1d(np.asarray(a, dtype=float)))
             for a in (mu, cd_scale, cl_scale, eta, rpm_offset, headwind_offset)]
    B = max(b.numel() for b in batch)
    mu_b, cd_b, cl_b, eta_b, rpm_b, hw_b = (b.expand(B) if b.numel() == 1 else b for b in batch)
    x, v = integrate_rk4(t, mu_b, cd_b, cl_b, eta_b, phys,
                         rpm_offset=rpm_b, headwind_offset=hw_b)
    with torch.no_grad():
        margin = normal_force_margin(torch.as_tensor(t).reshape(1, -1), v,
                                     cl_b.reshape(-1, 1), phys,
                                     headwind_offset=hw_b.reshape(-1, 1)).numpy()
    x_np, v_np = x.numpy(), v.numpy()
    out = {"t": np.full(B, np.nan), "x": np.full(B, np.nan), "V": np.full(B, np.nan)}
    for i in range(B):
        out["t"][i], out["x"][i] = first_event(t, x_np[i], margin[i])
        if np.isfinite(out["t"][i]):
            out["V"][i] = np.interp(out["t"][i], t, v_np[i])
    out["x_TF"] = np.array([np.interp(phys.TF, t, xi) for xi in x_np])
    return out


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

    # Liftoff = first L >= W*cos(gamma). Predicted by forward integration with
    # the identified parameters (the network only spans the record [0, TF]);
    # the network-state event inside the record is reported as a diagnostic.
    with torch.no_grad():
        margin_net = normal_force_margin(torch.as_tensor(fit["t"]), torch.as_tensor(fit["V"]),
                                         fit["cl_scale"], phys).numpy()
    m["t_liftoff_net"] = first_event(fit["t"], fit["x"], margin_net)[0]
    pred = predict_liftoff(fit["mu"], fit["cd_scale"], fit["cl_scale"], fit["eta"], phys)
    m["t_liftoff_pred"], m["x_liftoff_pred"], m["v_liftoff_pred"] = (
        float(pred[k][0]) for k in ("t", "x", "V"))
    if truth is not None:
        ref = predict_liftoff(truth.mu, truth.cd_scale, truth.cl_scale, truth.eta, phys)
        m["t_liftoff_true"], m["x_liftoff_true"], m["v_liftoff_true"] = (
            float(ref[k][0]) for k in ("t", "x", "V"))
        if np.isfinite(m["t_liftoff_pred"]) and np.isfinite(m["t_liftoff_true"]):
            m["liftoff_time_abs_err"] = abs(m["t_liftoff_pred"] - m["t_liftoff_true"])
            m["ground_roll_abs_err"] = abs(m["x_liftoff_pred"] - m["x_liftoff_true"])
            m["ground_roll_pct_err"] = 100.0 * m["ground_roll_abs_err"] / abs(m["x_liftoff_true"])
    for name in ("mu", "cd_scale", "cl_scale", "eta"):
        m[name] = fit[name]
        if truth is not None:
            m[f"{name}_abs_err"] = abs(fit[name] - getattr(truth, name))
    m["kappa"] = lumped_drag_coefficient(fit["cd_scale"], fit["cl_scale"], fit["mu"], phys)
    if truth is not None:
        m["kappa_true"] = lumped_drag_coefficient(truth.cd_scale, truth.cl_scale, truth.mu, phys)
        m["kappa_rel_err"] = abs(m["kappa"] - m["kappa_true"]) / abs(m["kappa_true"])
    return m
