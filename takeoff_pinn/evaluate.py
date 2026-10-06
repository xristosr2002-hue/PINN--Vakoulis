"""Dense model evaluation, event metrics, and summary statistics."""
import numpy as np
import torch

from .config import PhysicsConfig, TrueParams
from .identifiability import lumped_drag_coefficient
from .data import first_event, predict_liftoff  # noqa: F401  (re-exported)
from .losses import grad_wrt
from .physics import normal_force_margin, rhs_accel

GEN_HEADWIND = 2.0  # m/s, generalization case (steady headwind)


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
    # Generalization: the same vehicle taking off into a steady +GEN_HEADWIND
    # m/s headwind, a condition the record never saw. Liftoff is set by
    # airspeed, so only correct aerodynamics (not merely a good fit) predicts it.
    gen = predict_liftoff(fit["mu"], fit["cd_scale"], fit["cl_scale"], fit["eta"], phys,
                          headwind_offset=GEN_HEADWIND)
    m["x_liftoff_gen_pred"] = float(gen["x"][0])
    if truth is not None:
        gref = predict_liftoff(truth.mu, truth.cd_scale, truth.cl_scale, truth.eta, phys,
                               headwind_offset=GEN_HEADWIND)
        m["x_liftoff_gen_true"] = float(gref["x"][0])
        m["gen_ground_roll_pct_err"] = (100.0 * abs(m["x_liftoff_gen_pred"] - m["x_liftoff_gen_true"])
                                        / m["x_liftoff_gen_true"])
    for name in ("mu", "cd_scale", "cl_scale", "eta"):
        m[name] = fit[name]
        if truth is not None:
            m[f"{name}_abs_err"] = abs(fit[name] - getattr(truth, name))
    m["kappa"] = lumped_drag_coefficient(fit["cd_scale"], fit["cl_scale"], fit["mu"], phys)
    if truth is not None:
        m["kappa_true"] = lumped_drag_coefficient(truth.cd_scale, truth.cl_scale, truth.mu, phys)
        m["kappa_rel_err"] = abs(m["kappa"] - m["kappa_true"]) / abs(m["kappa_true"])
    return m
